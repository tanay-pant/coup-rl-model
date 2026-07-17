import os
import sys
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
import numpy as np
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from scripts.train_lstm_advanced import CoupActionMaskLSTM
from agents.neural_mcts import NeuralMCTSBot

# ======================================================================
# EVALUATION CONFIGURATION
# ======================================================================
NUM_GAMES_PER_GAUNTLET = 200
MCTS_SIMULATIONS = 600
MCTS_MAX_TIME = 0.6
# ======================================================================

def get_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    return os.path.join(ckpt_dir, checkpoints[-1])

def evaluate_mcts():
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(base_dir, "checkpoints_lstm_advanced_v2")
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    
    if not checkpoint_path:
        print(f"Error: No checkpoint found in {checkpoint_dir}. Skipping.")
        return None

    print(f"\nLoading checkpoint from {checkpoint_path}...")
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
        policy_dir = os.path.join(checkpoint_path, "policies", "main_policy")
        loaded_policy = Policy.from_checkpoint(policy_dir)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None
    
    def get_valid_policy(pool):
        valid = []
        for p in pool:
            try:
                if algo.get_policy(p) is not None:
                    valid.append(p)
            except Exception:
                pass
        return valid if valid else ["honest_policy"]

    # We test MCTS against these heuristic opponents
    gauntlets = {
        "Random": ["random_policy"],
        "Rational": ["rational_policy"],
        "Aggressive": ["terminator_policy", "aggressive_policy"],
        "Tracker": ["tracker_policy"],
        "Honest": ["honest_policy"],
        "Mixed": ["rational_policy", "tracker_policy", "honest_policy", "random_policy"]
    }
    
    results = {}
    env = CoupEnv()
    
    # Initialize the MCTS bot for player_0
    mcts_bot = NeuralMCTSBot(agent_id=0, loaded_policy=loaded_policy, num_simulations=MCTS_SIMULATIONS, max_time=MCTS_MAX_TIME)
    
    for gauntlet_name, pool in gauntlets.items():
        valid_pool = get_valid_policy(pool)
        print(f"\nStarting '{gauntlet_name}' Gauntlet ({NUM_GAMES_PER_GAUNTLET} games) vs {valid_pool}...")
        wins = 0
        total_turns = 0
        
        for game in range(NUM_GAMES_PER_GAUNTLET):
            env.reset()
            policy_map = {}
            
            # Setup MCTS initial LSTM state
            has_state = hasattr(loaded_policy, "get_initial_state") and len(loaded_policy.get_initial_state()) > 0
            global_lstm_states = {}
            if has_state:
                for agent in env.agents:
                    global_lstm_states[agent] = [torch.tensor([s], dtype=torch.float32) for s in loaded_policy.get_initial_state()]
            
            for p in env.agents:
                if p != "player_0":
                    policy_map[p] = random.choice(valid_pool)
                    
            turns = 0
            agent_rewards = {p: 0 for p in env.agents}
            
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                
                if termination or truncation:
                    action = None
                else:
                    if agent == "player_0":
                        # Run full MCTS for our agent
                        action = mcts_bot.compute_action(env, current_lstm_states=global_lstm_states)
                    else:
                        # Fast heuristic policies for opponents
                        policy_id = policy_map[agent]
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
            
            if (game + 1) % max(1, int(NUM_GAMES_PER_GAUNTLET/10)) == 0:
                print(f"[{gauntlet_name}] Played {game+1}/{NUM_GAMES_PER_GAUNTLET} games... Current Win Rate: {wins/(game+1)*100:.2f}%")
                
        win_rate = (wins/NUM_GAMES_PER_GAUNTLET)*100
        avg_turns = total_turns/NUM_GAMES_PER_GAUNTLET
        results[gauntlet_name] = (win_rate, avg_turns)
        
    algo.stop()
    ray.shutdown()
    
    
    report = []
    report.append("=" * 65)
    report.append("FINAL MCTS EVALUATION RESULTS")
    report.append("=" * 65)
    report.append(f"{'Gauntlet':<15} | {'Win Rate':<10} | {'Avg Turns'}")
    report.append("-" * 65)
    for gauntlet, (win_rate, avg_turns) in results.items():
        report.append(f"{gauntlet:<15} | {win_rate:>8.2f}% | {avg_turns:>9.1f}")
    report.append("=" * 65)
    
    report_str = "\n".join(report)
    print("\n" + report_str)
    
    out_file = os.path.join(base_dir, "mcts_evaluation_results.txt")
    with open(out_file, "w") as f:
        f.write(report_str + "\n")
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    evaluate_mcts()
