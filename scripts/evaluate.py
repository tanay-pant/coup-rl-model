import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from envs.coup.coup_env import CoupEnv
from scripts.train_rllib import setup_rllib_config
import numpy as np
import random

def evaluate_model():
    ray.init()
    
    # Load config and algorithm
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v2'))
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_10000")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        ray.shutdown()
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    env = CoupEnv()
    
    num_games = 1000
    wins = 0
    total_turns = 0
    
    print(f"Starting Evaluation Tournament ({num_games} games)...")
    
    for game in range(num_games):
        env.reset()
        
        # Force player_0 to be the AI, everyone else is a heuristic
        policy_map = {"player_0": "main_policy"}
        for p in env.agents:
            if p != "player_0":
                policy_map[p] = random.choice(["random_policy", "honest_policy", "aggressive_policy"])
                
        turns = 0
        agent_rewards = {p: 0 for p in env.agents}
        
        # PettingZoo AEC loop
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
                
        # Game over, check if player_0 survived/won
        if agent_rewards.get("player_0", 0) > 0:
            wins += 1
            
        total_turns += turns
        
        if (game + 1) % 100 == 0:
            print(f"Played {game+1} games... Current Win Rate: {wins/(game+1)*100:.2f}%")
            
    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Games Played: {num_games}")
    print(f"Main Policy Wins:   {wins}")
    print(f"Main Policy Win Rate: {(wins/num_games)*100:.2f}%")
    print(f"Average Game Length (turns): {total_turns/num_games:.1f}")
    print("=" * 40)
    
    ray.shutdown()

if __name__ == "__main__":
    evaluate_model()
