import os
import sys
import uuid
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from agents.ismcts import ISMCTSBot

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

@app.on_event("startup")
def startup_event():
    print("Backend started with ISMCTS!")

@app.get("/health")
def health_check():
    return {"status": "ok"}

NAMES = ["You", "Abe", "Bart", "Charlie", "Dave", "Eve"]

def possessive(name):
    if name == "You":
        return "Your"
    if name.endswith("s"):
        return f"{name}'"
    return f"{name}'s"

def get_target_name(target_idx):
    if 0 <= target_idx < len(NAMES):
        return NAMES[target_idx]
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

def generate_contextual_log(env, action, agent_idx):
    agent_name = get_target_name(agent_idx)
    action_name = get_action_name(action, agent_idx)
    
    # Intercept ANY block action globally so it can never fall through to the fallback
    if action in [24, 25, 27, 28]:
        # During a block, the active_player is the one whose action is being blocked
        active_player = get_target_name(env.state.turn.active_player)
        original_action = get_action_name(env.state.turn.action, env.state.turn.active_player)
        return f"{agent_name} blocked {possessive(active_player)} {original_action} with {action_name.replace('Block with ', '')}"
        
    phase = env.state.turn.phase.name
    
    if phase == "START_OF_TURN":
        return f"{agent_name} decided to {action_name}"
    
    if phase == "ACTION_CHALLENGE":
        active_player = get_target_name(env.state.turn.active_player)
        original_action = get_action_name(env.state.turn.action, env.state.turn.active_player)
        
        if action == 23: # Allow
            return f"{agent_name} allowed ({active_player} {original_action})"
        elif action == 22: # Challenge
            return f"{agent_name} challenged {possessive(active_player)} {original_action}"
            
    if phase == "ACTION_BLOCK":
        active_player = get_target_name(env.state.turn.active_player)
        original_action = get_action_name(env.state.turn.action, env.state.turn.active_player)
        
        if action == 23: # Allow
            return f"{agent_name} allowed ({active_player} {original_action})"
            
    if phase == "BLOCK_RESPONSE":
        blocker = get_target_name(env.state.turn.target)
        active_player = get_target_name(env.state.turn.active_player)
        original_action = get_action_name(env.state.turn.action, env.state.turn.active_player)
        if action == 23:
            return f"{agent_name} accepted ({blocker} block {active_player} {original_action})"
        elif action == 22:
            return f"{agent_name} challenged {possessive(blocker)} block ({active_player} {original_action})"
            
    if phase == "REVEAL_INFLUENCE":
        roles = {29: "Duke", 30: "Assassin", 31: "Captain", 32: "Ambassador", 33: "Contessa"}
        revealed_role = roles.get(action, "Unknown")
        return f"{agent_name} revealed a {revealed_role}"
        
    if phase == "EXCHANGE":
        return None

    return f"{agent_name} chose: {action_name}"

def serialize_state(env, log_messages, player_0_placement, valid_actions_mask=None, active_agent=None):
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
            "name": get_target_name(i),
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
        "log": log_messages,
        "winner": winner,
        "deck_count": len(env.state.deck),
        "player_0_placement": player_0_placement
    }

active_sessions = {}

class GameSession:
    def __init__(self, bot_count):
        self.env = CoupEnv()
        self.env.reset(options={"num_players": bot_count + 1})
        self.log_messages = ["Game Started!"]
        self.game_active = True
        self.player_0_placement = None
        self.player_0_was_alive = True
        self.action_queue = asyncio.Queue()
        self.active_websocket = None
        self.last_state_data = None
        
        self.bots = {}
        for agent in self.env.agents:
            if agent != "player_0":
                agent_idx = int(agent.split("_")[1])
                self.bots[agent] = ISMCTSBot(agent_id=agent_idx, num_simulations=10000, max_time=0.5)
        
    async def send_json(self, data):
        if self.active_websocket:
            try:
                await self.active_websocket.send_json(data)
            except Exception:
                pass

async def game_engine_loop(session_id: str):
    session = active_sessions.get(session_id)
    if not session: return
    env = session.env
    
    while session.game_active:
        for agent in env.agent_iter():
            if not session.game_active:
                break
                
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                state_data = serialize_state(env, session.log_messages, session.player_0_placement, None, None)
                if state_data["winner"] is not None:
                    if session.player_0_was_alive:
                        if state_data["players"][0]["alive"]:
                            session.player_0_placement = 1
                        else:
                            session.player_0_placement = 2
                            session.player_0_was_alive = False
                    session.log_messages.append(f"Game Over! Winner: {state_data['winner']}")
                    final_state = serialize_state(env, session.log_messages, session.player_0_placement, None, None)
                    session.last_state_data = final_state
                    await session.send_json({
                        "type": "state_update",
                        "data": final_state
                    })
                    session.game_active = False
                    break
                env.step(None)
                continue
            
            action_mask = observation["action_mask"]
            agent_idx = int(agent.split('_')[1])
            
            state_data = serialize_state(env, session.log_messages, session.player_0_placement, action_mask, agent_idx)
            
            if session.player_0_was_alive and not state_data["players"][0]["alive"]:
                alive_count = sum(1 for p in state_data["players"] if p["alive"])
                session.player_0_placement = alive_count + 1
                session.player_0_was_alive = False
                state_data["player_0_placement"] = session.player_0_placement
                
            session.last_state_data = state_data
            await session.send_json({
                "type": "state_update",
                "data": state_data
            })
            
            if agent == "player_0":
                while session.game_active:
                    data = await session.action_queue.get()
                    if data.get("type") == "action":
                        action = data.get("action_id")
                        if action_mask[action] == 1:
                            log_msg = generate_contextual_log(env, action, 0)
                            was_challenge = (action == 22)
                            env.step(action)
                            if was_challenge:
                                loser = env.state.turn.player_to_reveal
                                result = "WRONG!" if loser == 0 else "RIGHT!"
                                if log_msg:
                                    log_msg = f"{log_msg}, {result}"
                                    
                            if log_msg:
                                session.log_messages.append(log_msg)
                            break
                        else:
                            await session.send_json({"type": "error", "message": "Invalid action"})
                    elif data.get("type") == "restart":
                        session.game_active = False
                        break
            else:
                await asyncio.sleep(0.5)
                await asyncio.sleep(0.1)
                
                bot = session.bots[agent]
                action = await asyncio.to_thread(bot.compute_action, env)

                log_msg = generate_contextual_log(env, action, agent_idx)
                was_challenge = (action == 22)
                env.step(action)
                if was_challenge:
                    loser = env.state.turn.player_to_reveal
                    result = "WRONG!" if loser == agent_idx else "RIGHT!"
                    if log_msg:
                        log_msg = f"{log_msg}, {result}"
                        
                if log_msg:
                    session.log_messages.append(log_msg)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    await websocket.accept()

    if not session_id:
        session_id = str(uuid.uuid4())
        await websocket.send_json({"type": "session_id", "session_id": session_id})
        
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session.active_websocket = websocket
        if session.last_state_data:
            await websocket.send_json({"type": "state_update", "data": session.last_state_data})
    else:
        # Lobby Phase
        await websocket.send_json({"type": "lobby_state"})

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "start_game":
                bot_count = int(data.get("bot_count", 2))
                session = GameSession(bot_count)
                session.active_websocket = websocket
                active_sessions[session_id] = session
                asyncio.create_task(game_engine_loop(session_id))
            elif session_id in active_sessions:
                await active_sessions[session_id].action_queue.put(data)
    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected gracefully")
        if session_id in active_sessions:
            active_sessions[session_id].active_websocket = None
    except Exception as e:
        print(f"Error in websocket loop: {e}")
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass

frontend_dist = os.path.join(base_dir, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Serve exact file if it exists (e.g. favicon.ico, rules_complete.pdf)
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html (SPA routing)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
