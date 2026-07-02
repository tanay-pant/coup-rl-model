import os
import sys
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from scripts.train_lstm import CoupActionMaskLSTM
from scripts.train_rllib import CoupActionMaskModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
envs_dir = os.path.join(base_dir, "envs")
scripts_dir = os.path.join(base_dir, "scripts")
agents_dir = os.path.join(base_dir, "agents")

ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)
ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)

loaded_policy = None

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    highest_idx = -1
    checkpoint_path = None
    for root, dirs, files in os.walk(checkpoint_dir):
        for d in dirs:
            if d.startswith("checkpoint_"):
                try:
                    idx = int(d.split("_")[1])
                    if idx > highest_idx:
                        highest_idx = idx
                        checkpoint_path = os.path.join(root, d)
                except ValueError:
                    continue
    return checkpoint_path

@app.on_event("startup")
def startup_event():
    global loaded_policy
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True, runtime_env={"py_modules": [envs_dir, scripts_dir, agents_dir]})
    
    checkpoint_dir = os.path.join(base_dir, "checkpoints_lstm_advanced")
    load_dir = get_latest_checkpoint(checkpoint_dir)
    if load_dir:
        policy_dir = os.path.join(load_dir, "policies", "main_policy")
        if os.path.exists(policy_dir):
            print(f"Loading Policy from: {policy_dir}")
            loaded_policy = Policy.from_checkpoint(policy_dir)
        else:
            print(f"WARNING: No policy found at {policy_dir}")
    else:
        print("WARNING: No checkpoint found!")

@app.get("/health")
def health_check():
    return {"status": "ok"}

def get_target_name(target_idx):
    if target_idx == 0:
        return "You"
    return f"AI {target_idx}"

def get_action_name(idx, agent_idx):
    if idx == 0: return "Income"
    if idx == 1: return "Foreign Aid"
    if idx == 2: return "Tax"
    if idx == 3: return "Exchange"
    
    MAX_PLAYERS = 6
    if 4 <= idx <= 9: return f"Steal from {get_target_name((agent_idx + (idx - 4 + 1)) % MAX_PLAYERS)}"
    if 10 <= idx <= 15: return f"Assassinate {get_target_name((agent_idx + (idx - 10 + 1)) % MAX_PLAYERS)}"
    if 16 <= idx <= 21: return f"Coup {get_target_name((agent_idx + (idx - 16 + 1)) % MAX_PLAYERS)}"
    if idx == 22: return "Challenge"
    if idx == 23: return "Allow"
    
    roles = {0: "Duke", 1: "Assassin", 2: "Captain", 3: "Ambassador", 4: "Contessa"}
    if idx == 24: return "Block with Duke"
    if idx == 25: return "Block with Captain"
    if idx == 27: return "Block with Contessa"
    if idx == 28: return "Block with Ambassador"
    if 29 <= idx <= 33: return f"Reveal {roles[idx - 29]}"
    if 34 <= idx <= 38: return f"Exchange return pool index {idx - 34}"
    
    return f"Unknown Action {idx}"

def serialize_state(env, log_messages, valid_actions_mask=None, active_agent=None):
    state = env.state
    
    players = []
    for i in range(state.num_players):
        p = state.players[i]
        cards = []
        for inf in p.influence:
            if inf.revealed:
                cards.append({"role": inf.role.name, "revealed": True})
            elif i == 0:
                cards.append({"role": inf.role.name, "revealed": False})
            else:
                cards.append({"role": "HIDDEN", "revealed": False})
        
        players.append({
            "id": i,
            "name": "You" if i == 0 else f"AI {i}",
            "cash": p.cash,
            "cards": cards,
            "alive": p.influence_count > 0
        })
        
    pool = []
    if state.turn.phase.name == "EXCHANGE" and state.turn.active_player == 0:
        pool = [{"id": i, "role": r.name} for i, r in enumerate(state.turn.exchange_pool) if r.value != -1]
        
    valid_actions = []
    if valid_actions_mask is not None and active_agent == 0:
        for i, m in enumerate(valid_actions_mask):
            if m == 1:
                valid_actions.append({"id": i, "name": get_action_name(i, 0)})
                
    winner = None
    alive_count = sum(1 for p in players if p["alive"])
    if alive_count <= 1:
        for p in players:
            if p["alive"]:
                winner = p["name"]

    return {
        "phase": state.turn.phase.name,
        "active_player": state.turn.active_player,
        "players": players,
        "exchange_pool": pool,
        "valid_actions": valid_actions,
        "log": log_messages[-15:], # Keep last 15 log messages
        "winner": winner
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if loaded_policy is None:
        await websocket.send_json({"type": "error", "message": "AI policy not loaded"})
        await websocket.close()
        return

    env = CoupEnv()
    env.reset()
    has_state = hasattr(loaded_policy, "get_initial_state") and len(loaded_policy.get_initial_state()) > 0
    state_map = {agent: loaded_policy.get_initial_state() if has_state else [] for agent in env.agents if agent != "player_0"}
    
    log_messages = ["Game Started!"]
    
    try:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
            
            action_mask = observation["action_mask"]
            agent_idx = int(agent.split('_')[1])
            
            state_data = serialize_state(env, log_messages, action_mask, agent_idx)
            await websocket.send_json({
                "type": "state_update",
                "data": state_data
            })
            
            if state_data["winner"] is not None:
                log_messages.append(f"Game Over! Winner: {state_data['winner']}")
                await websocket.send_json({
                    "type": "state_update",
                    "data": serialize_state(env, log_messages, None, None)
                })
                break
                
            if agent == "player_0":
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "action":
                        action = data.get("action_id")
                        if action_mask[action] == 1:
                            log_messages.append(f"You chose: {get_action_name(action, 0)}")
                            env.step(action)
                            break
                        else:
                            await websocket.send_json({"type": "error", "message": "Invalid action"})
            else:
                await asyncio.sleep(0.5)
                if has_state:
                    action, state_out, info = await asyncio.to_thread(
                        loaded_policy.compute_single_action,
                        obs=observation,
                        state=state_map[agent],
                        explore=True
                    )
                    state_map[agent] = state_out
                else:
                    action = await asyncio.to_thread(
                        loaded_policy.compute_single_action,
                        obs=observation,
                        explore=True
                    )
                
                # Check if action is a numpy array or tuple and extract the integer
                # Sometimes compute_single_action returns a tuple like (action, state, info) but we already unpacked it
                # If the action itself is a list/array with one element, extract it.
                if isinstance(action, tuple):
                    action = int(action[0])
                else:
                    try:
                        action = int(action)
                    except TypeError:
                        action = action.item()

                agent_name = "You" if agent == "player_0" else f"AI {agent_idx}"
                log_messages.append(f"{agent_name} chose: {get_action_name(action, agent_idx)}")
                env.step(action)
                
        # Game is over, keep connection open if client wants
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        print("Client disconnected gracefully")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass
