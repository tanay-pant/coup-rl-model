import os
import sys
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from scripts.train_lstm import CoupActionMaskLSTM

# ======================================================================
# PLAY CONFIGURATION
# ======================================================================
# Change this variable to choose which AI model to play against.
# Valid options: "rllib", "rllib_pbt", "lstm", "lstm_pbt"
PLAY_MODE = "lstm"
# ======================================================================

def get_action_name(idx):
    if idx == 0: return "Income"
    if idx == 1: return "Foreign Aid"
    if idx == 2: return "Tax"
    if idx == 3: return "Exchange"
    if 4 <= idx <= 9: return f"Steal from Player {idx - 4}"
    if 10 <= idx <= 15: return f"Assassinate Player {idx - 10}"
    if 16 <= idx <= 21: return f"Coup Player {idx - 16}"
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

def print_board(env):
    state = env.state
    print("\n" + "="*45)
    print(f"   --- COUP TERMINAL INTERFACE ---")
    print(f"Phase: {state.turn.phase.name} | Active Player: {state.turn.active_player}")
    print("="*45)
    for i in range(state.num_players):
        p = state.players[i]
        cards = []
        for inf in p.influence:
            if inf.revealed:
                cards.append(f"[{inf.role.name} (DEAD)]")
            elif i == 0:
                cards.append(f"[{inf.role.name}]")
            else:
                cards.append("[HIDDEN]")
        prefix = "YOU" if i == 0 else f"AI {i}"
        print(f"{prefix:6} | {p.cash} Coins | Cards: {' '.join(cards)}")
        
    if state.turn.phase.name == "EXCHANGE" and state.turn.active_player == 0:
        pool_cards = [f"[{i}: {r.name}]" for i, r in enumerate(state.turn.exchange_pool) if r.value != -1]
        print("-" * 45)
        print(f"EXCHANGE POOL: {' '.join(pool_cards)}")
        
    print("="*45 + "\n")

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

def main():
    ray.init(ignore_reinit_error=True)
    from scripts.train_lstm import CoupActionMaskLSTM
    from scripts.train_rllib import CoupActionMaskModel
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)
    ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    directories = {
        "rllib": os.path.join(base_dir, "checkpoints_rllib"),
        "rllib_pbt": os.path.join(base_dir, "checkpoints_pbt"),
        "lstm": os.path.join(base_dir, "checkpoints_lstm"),
        "lstm_pbt": os.path.join(base_dir, "checkpoints_lstm_pbt")
    }
    
    if PLAY_MODE not in directories:
        print(f"Invalid PLAY_MODE: {PLAY_MODE}")
        sys.exit(1)

    checkpoint_dir = directories[PLAY_MODE]
    load_dir = get_latest_checkpoint(checkpoint_dir)
                            
    if not load_dir:
        print(f"No valid checkpoints found for {PLAY_MODE} in {checkpoint_dir}.")
        print(f"Please train the {PLAY_MODE} model first!")
        sys.exit(1)
        
    print(f"Loading {PLAY_MODE} AI Brain from: {load_dir}")
    algo = Algorithm.from_checkpoint(load_dir)
    
    env = CoupEnv()
    env.reset()
    
    policy = algo.get_policy("main_policy")
    has_state = hasattr(policy, "get_initial_state") and len(policy.get_initial_state()) > 0
    state_map = {agent: policy.get_initial_state() if has_state else [] for agent in env.agents if agent != "player_0"}
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
        
        action_mask = observation["action_mask"]
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        
        if agent == "player_0":
            print_board(env)
            print(f"--- IT IS YOUR TURN ---")
            
            for v in valid_actions:
                print(f"[{v:2d}] {get_action_name(v)}")
            
            action = None
            while action not in valid_actions:
                try:
                    choice = input("Enter Action Number: ")
                    action = int(choice)
                    if action not in valid_actions:
                        print("Invalid choice. Please select a valid number from the menu.")
                except ValueError:
                    print("Please enter a valid number.")
            
            env.step(action)
        else:
            if has_state:
                action, state_out, info = algo.compute_single_action(
                    observation=observation,
                    state=state_map[agent],
                    policy_id="main_policy",
                    explore=True
                )
                state_map[agent] = state_out
            else:
                action = algo.compute_single_action(
                    observation=observation,
                    policy_id="main_policy",
                    explore=True
                )
                
            print(f">>> Player {agent} chose: {get_action_name(action)}")
            env.step(action)
            
    print("\nGAME OVER!")
    for i in range(env.MAX_PLAYERS):
        if env.state.players[i].influence_count > 0:
            winner = "YOU" if i == 0 else f"AI {i}"
            print(f"🏆 WINNER: {winner} 🏆")
            break
            
    ray.shutdown()

if __name__ == "__main__":
    main()
