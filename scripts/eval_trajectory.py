import os
import sys
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from scripts.train_rllib import CoupActionMaskModel
import random

NUM_GAMES = 500

def evaluate_checkpoint(checkpoint_path):
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None
    
    env = CoupEnv()
    wins = 0
    total_turns = 0
    
    for game in range(NUM_GAMES):
        env.reset()
        policy_map = {"player_0": "main_policy"}
        
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
                action = algo.compute_single_action(
                    observation=obs,
                    policy_id=policy_id,
                    explore=True 
                )
                
            env.step(action)
            agent_rewards[agent] += reward
            turns += 1
                
        if agent_rewards.get("player_0", 0) > 0:
            wins += 1
            
        total_turns += turns
            
    win_rate = (wins/NUM_GAMES)*100
    avg_turns = total_turns/NUM_GAMES
    algo.stop()
    return win_rate, avg_turns

def main():
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)
    
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v2'))
    
    checkpoints = ["checkpoint_1000", "checkpoint_2000", "checkpoint_3000", "checkpoint_4000"]
    
    results = {}
    for cp in checkpoints:
        cp_path = os.path.join(checkpoint_dir, cp)
        if os.path.exists(cp_path):
            res = evaluate_checkpoint(cp_path)
            if res:
                results[cp] = res

    print("\n" + "=" * 40)
    print("BASE RLLIB IMPROVEMENT TRAJECTORY")
    print("=" * 40)
    print(f"{'Checkpoint':<20} | {'Win Rate':<10} | {'Avg Turns'}")
    print("-" * 40)
    for cp, (win_rate, avg_turns) in results.items():
        print(f"{cp:<20} | {win_rate:>8.2f}% | {avg_turns:>9.1f}")
    print("=" * 40)
    
    ray.shutdown()

if __name__ == "__main__":
    main()
