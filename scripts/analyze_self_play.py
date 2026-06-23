import os
import sys
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from scripts.train_rllib import CoupActionMaskModel
from scripts.train_lstm import CoupActionMaskLSTM

PLAY_MODE = "lstm"
NUM_GAMES = 3

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
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)
    ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    directories = {
        "rllib": os.path.join(base_dir, "checkpoints_rllib"),
        "lstm": os.path.join(base_dir, "checkpoints_lstm"),
    }
    
    load_dir = get_latest_checkpoint(directories[PLAY_MODE])
    if not load_dir:
        print(f"No checkpoint found for {PLAY_MODE}")
        sys.exit(1)
        
    print(f"Loading {PLAY_MODE} AI Brain from: {load_dir}")
    algo = Algorithm.from_checkpoint(load_dir)
    
    env = CoupEnv()
    policy = algo.get_policy("main_policy")
    has_state = hasattr(policy, "get_initial_state") and len(policy.get_initial_state()) > 0
    
    for game in range(NUM_GAMES):
        print(f"\n================ GAME {game+1} ================\n")
        env.reset()
        state_map = {agent: policy.get_initial_state() if has_state else [] for agent in env.agents}
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
            
            if has_state:
                action, state_out, _ = algo.compute_single_action(
                    observation=obs,
                    state=state_map[agent],
                    policy_id="main_policy",
                    explore=True
                )
                state_map[agent] = state_out
            else:
                action = algo.compute_single_action(
                    observation=obs,
                    policy_id="main_policy",
                    explore=True
                )
                
            print(f"Player {agent} chose: {get_action_name(action)}")
            env.step(action)
            
        print("\nGAME OVER!")
        for i in range(env.MAX_PLAYERS):
            if i in env.state.players and env.state.players[i].influence_count > 0:
                print(f"🏆 WINNER: Player {i} 🏆")
                break

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
