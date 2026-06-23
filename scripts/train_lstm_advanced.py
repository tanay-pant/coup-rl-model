import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import random
import csv

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from envs.coup.coup_env import CoupEnv

from agents.league_policies import RandomHeuristicPolicy, HonestHeuristicPolicy, AggressiveHeuristicPolicy, CowardHeuristicPolicy, TerminatorHeuristicPolicy, TrackerHeuristicPolicy
from scripts.train_lstm import CoupActionMaskLSTM

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "player_0":
        return "main_policy"
    r = random.random()
    if r < 0.30: return "main_policy"
    elif r < 0.36: return "past_policy_1"
    elif r < 0.42: return "past_policy_2"
    elif r < 0.48: return "past_policy_3"
    elif r < 0.54: return "past_policy_4"
    elif r < 0.60: return "past_policy_5"
    elif r < 0.68: return "honest_policy"
    elif r < 0.76: return "aggressive_policy"
    elif r < 0.84: return "coward_policy"
    elif r < 0.92: return "terminator_policy"
    else: return "tracker_policy"

def env_creator(config):
    env = CoupEnv()
    return PettingZooEnv(env)

register_env("coup_parallel_v0", env_creator)
ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)

def setup_rllib_config(env_name="coup_parallel_v0", num_workers=6, use_pbt=False):
    dummy_env = env_creator({})
    obs_space = dummy_env.observation_space["player_0"]
    act_space = dummy_env.action_space["player_0"]

    config = (
        PPOConfig()
        .environment(env=env_name)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(num_env_runners=num_workers, num_envs_per_env_runner=10)
        .training(
            train_batch_size=6000,
            minibatch_size=600,
            # Decaying entropy from 0.05 down to 0.01 over the 60M timesteps (10,000 iterations)
            entropy_coeff_schedule=[[0, 0.05], [60000000, 0.01]],
            model={
                "custom_model": "coup_mask_lstm",
                "max_seq_len": 30, 
            }
        )
        .multi_agent(
            policies={
                "main_policy": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_1": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_2": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_3": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_4": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_5": PolicySpec(observation_space=obs_space, action_space=act_space),
                "random_policy": PolicySpec(policy_class=RandomHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "honest_policy": PolicySpec(policy_class=HonestHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "aggressive_policy": PolicySpec(policy_class=AggressiveHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "coward_policy": PolicySpec(policy_class=CowardHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "terminator_policy": PolicySpec(policy_class=TerminatorHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "tracker_policy": PolicySpec(policy_class=TrackerHeuristicPolicy, observation_space=obs_space, action_space=act_space),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main_policy"]
        )
    )
    return config


def train_coup():
    ray.init()

    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_lstm_advanced'))
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = setup_rllib_config()
    
    def get_latest_checkpoint(ckpt_dir):
        if not os.path.exists(ckpt_dir): return None
        highest_idx = -1
        ckpt_path = None
        for cp in os.listdir(ckpt_dir):
            if cp.startswith("checkpoint_"):
                try:
                    idx = int(cp.split("_")[1])
                    if idx > highest_idx:
                        highest_idx = idx
                        ckpt_path = os.path.join(ckpt_dir, cp)
                except ValueError:
                    pass
        return ckpt_path

    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    is_new_run = latest_ckpt is None

    from ray.rllib.algorithms.algorithm import Algorithm

    if latest_ckpt:
        print(f"Resuming training from {latest_ckpt}...")
        # To apply our updated config (specifically higher entropy), we instantiate algo from config, then restore weights
        algo = config.build_algo()
        algo.restore(latest_ckpt)
    else:
        print("No checkpoint found. Starting from scratch...")
        algo = config.build_algo()

    mode = "a" if not is_new_run and latest_ckpt else "w"
    log_file = open("training_lstm_advanced_log.csv", mode, newline="")
    csv_writer = csv.writer(log_file)
    if mode == "w":
        csv_writer.writerow(["Iteration", "Mean Reward", "Policy Loss", "Value Loss", "Entropy"])

    print("Starting Multi-Agent PPO LSTM Training on Coup from Scratch...")
    
    start_iter = algo.iteration if hasattr(algo, 'iteration') else 0
    target_iter = start_iter + 10000
    
    print(f"Starting at iteration {start_iter}, targeting {target_iter}")

    for i in range(start_iter + 1, target_iter + 1):
        result = algo.train()
        
        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", 
                      result.get("episode_reward_mean", 0.0))
        
        learner_stats = result.get("info", {}).get("learner", {}).get("main_policy", {}).get("learner_stats", {})
        policy_loss = learner_stats.get("policy_loss", 0.0)
        vf_loss = learner_stats.get("vf_loss", 0.0)
        entropy = learner_stats.get("entropy", 0.0)
        
        print(f"Iteration {i:03d} | Reward: {mean_reward:7.4f} | Policy Loss: {policy_loss:7.4f} | VF Loss: {vf_loss:7.4f} | Entropy: {entropy:7.4f}")
        csv_writer.writerow([i, mean_reward, policy_loss, vf_loss, entropy])
        log_file.flush()

        if i % 1000 == 0:
            current_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{algo.iteration}")
            checkpoint_path = algo.save(current_checkpoint_dir)
            print(f"=== Saved Checkpoint at Iteration {algo.iteration} to: {checkpoint_path} ===")

        if i % 50 == 0:
            print(f"=== Rotating Policies (Fictitious Self-Play) ===")
            main_weights = algo.get_policy("main_policy").get_weights()
            if random.random() < 0.2:
                past_4_weights = algo.get_policy("past_policy_4").get_weights()
                algo.get_policy("past_policy_5").set_weights(past_4_weights)
                past_3_weights = algo.get_policy("past_policy_3").get_weights()
                algo.get_policy("past_policy_4").set_weights(past_3_weights)
                past_2_weights = algo.get_policy("past_policy_2").get_weights()
                algo.get_policy("past_policy_3").set_weights(past_2_weights)
                past_1_weights = algo.get_policy("past_policy_1").get_weights()
                algo.get_policy("past_policy_2").set_weights(past_1_weights)
            algo.get_policy("past_policy_1").set_weights(main_weights)

    log_file.close()
    print("Training Complete.")
    ray.shutdown()

if __name__ == "__main__":
    train_coup()
