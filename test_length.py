import os
import sys
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from envs.coup.coup_env import CoupEnv
from ray.rllib.models import ModelCatalog
from envs.coup.game_logic import CoupActionMaskLSTM

ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)

def test_game_lengths(num_games=100):
    load_dir = "/Users/tanay-pant/Downloads/projects/coup_rl_project/checkpoints_lstm_advanced/checkpoint_50000"
    algo = Algorithm.from_checkpoint(load_dir)
    policy = algo.get_policy("main_policy")
    
    env = CoupEnv(max_moves=2000)
    
    lengths = []
    truncations_count = 0
    
    for _ in range(num_games):
        env.reset()
        has_state = hasattr(policy, "get_initial_state") and len(policy.get_initial_state()) > 0
        state_map = {agent: policy.get_initial_state() if has_state else [] for agent in env.agents}
        
        step_count = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
                
            out = policy.compute_single_action(observation["observation"], state=state_map[agent], explore=False)
            action = out[0]
            state_map[agent] = out[1]
            
            env.step(action)
            step_count += 1
            
        lengths.append(step_count)
        if step_count >= 2000:
            truncations_count += 1
            
    print(f"Tested {num_games} games.")
    print(f"Average Steps: {np.mean(lengths):.1f}")
    print(f"Max Steps: {np.max(lengths)}")
    print(f"Min Steps: {np.min(lengths)}")
    print(f"90th Percentile: {np.percentile(lengths, 90)}")
    print(f"Games > 200 steps: {sum(1 for l in lengths if l >= 200)} ({sum(1 for l in lengths if l >= 200) / num_games * 100}%)")

if __name__ == "__main__":
    test_game_lengths(100)
