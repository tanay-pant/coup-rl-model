import os
import sys
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from scripts.train_lstm import setup_rllib_config, CoupActionMaskLSTM

# ======================================================================
# EVALUATION CONFIGURATION
# ======================================================================
# Change this variable to evaluate a specific model or compare all four.
# Valid options: "rllib", "rllib_pbt", "lstm", "lstm_pbt", "all"
EVAL_MODE = "lstm"
NUM_GAMES = 1000
# ======================================================================

def get_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    return os.path.join(ckpt_dir, checkpoints[-1])

def evaluate_single_model(name, checkpoint_dir):
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        print(f"[{name}] Error: No checkpoint found in {checkpoint_dir}. Skipping.")
        return None

    print(f"\n[{name}] Loading checkpoint from {checkpoint_path}...")
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"[{name}] Failed to load checkpoint: {e}")
        return None
    
    env = CoupEnv()
    wins = 0
    total_turns = 0
    
    print(f"[{name}] Starting Tournament ({NUM_GAMES} games)...")
    
    for game in range(NUM_GAMES):
        env.reset()
        policy_map = {"player_0": "main_policy"}
        
        # Check if the loaded policy uses RNN states
        policy = algo.get_policy("main_policy")
        has_state = hasattr(policy, "get_initial_state") and len(policy.get_initial_state()) > 0
        state_map = {"player_0": policy.get_initial_state() if has_state else []}
        
        for p in env.agents:
            if p != "player_0":
                policy_map[p] = random.choice(["random_policy", "honest_policy", "aggressive_policy"])
                
        turns = 0
        agent_rewards = {p: 0 for p in env.agents}
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                policy_id = policy_map[agent]
                if policy_id == "main_policy":
                    if has_state:
                        action, state_out, _ = algo.compute_single_action(
                            observation=obs,
                            state=state_map[agent],
                            policy_id=policy_id,
                            explore=True 
                        )
                        state_map[agent] = state_out
                    else:
                        action = algo.compute_single_action(
                            observation=obs,
                            policy_id=policy_id,
                            explore=True 
                        )
                else:
                    action = algo.compute_single_action(
                        observation=obs,
                        policy_id=policy_id,
                        explore=True 
                    )
                
            env.step(action)
            agent_rewards[agent] += reward
            if not (termination or truncation) and env.state.turn.phase.value == 1:
                turns += 1
                
        if agent_rewards.get("player_0", 0) > 0:
            wins += 1
            
        total_turns += turns
        
        if (game + 1) % int(NUM_GAMES/5) == 0:
            print(f"[{name}] Played {game+1}/{NUM_GAMES} games... Current Win Rate: {wins/(game+1)*100:.2f}%")
            
    win_rate = (wins/NUM_GAMES)*100
    avg_turns = total_turns/NUM_GAMES
    algo.stop()
    return win_rate, avg_turns

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
        "lstm": os.path.join(base_dir, "checkpoints_lstm_shaped"),
        "lstm_pbt": os.path.join(base_dir, "checkpoints_lstm_pbt"),
        "lstm_shaped": os.path.join(base_dir, "checkpoints_lstm_shaped")
    }
    
    if EVAL_MODE == "all":
        modes_to_run = ["lstm", "lstm_shaped"]
    elif EVAL_MODE in directories:
        modes_to_run = [EVAL_MODE]
    else:
        print(f"Invalid EVAL_MODE: {EVAL_MODE}")
        ray.shutdown()
        return

    results = {}
    for mode in modes_to_run:
        res = evaluate_single_model(mode, directories[mode])
        if res:
            results[mode] = res

    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Model Type':<15} | {'Win Rate':<10} | {'Avg Turns'}")
    print("-" * 50)
    for mode, (win_rate, avg_turns) in results.items():
        print(f"{mode:<15} | {win_rate:>8.2f}% | {avg_turns:>9.1f}")
    print("=" * 50)
    
    ray.shutdown()

if __name__ == "__main__":
    main()
