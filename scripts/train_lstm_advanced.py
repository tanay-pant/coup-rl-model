import os
os.environ["OMP_NUM_THREADS"] = "1"  # Fix M1 CPU thread contention
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

from agents.league_policies import RandomHeuristicPolicy, HonestHeuristicPolicy, AggressiveHeuristicPolicy, CowardHeuristicPolicy, TerminatorHeuristicPolicy, TrackerHeuristicPolicy, RationalHeuristicPolicy
from scripts.train_lstm import CoupActionMaskLSTM

class CoupTrainingCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result["training_iteration"]
        
        # 1. Base Entropy Schedule (Piecewise interpolation)
        if iteration <= 2000:
            base_entropy = 0.20
        elif iteration <= 30000:
            fraction = (iteration - 2000) / 28000.0
            base_entropy = 0.20 - (0.15 * fraction)
        else:
            fraction = (iteration - 30000) / 20000.0
            base_entropy = 0.05 - (0.04 * fraction)
            base_entropy = max(0.01, base_entropy)
            
        # 2. Entropy Micro-bumps (FSP Rotation boundaries)
        bump = 0.0
        if iteration > 30000:
            cycles_since_rotation = iteration % 100
            if cycles_since_rotation < 25:
                bump_fraction = 1.0 - (cycles_since_rotation / 25.0)
                bump = 0.007 * bump_fraction
                
        final_entropy = base_entropy + bump
        
        # 3. Clip Param schedule
        clip_val = 0.3 if iteration <= 30000 else 0.15
        if bump > 0.0:
            clip_val = 0.25
        
        # Apply config updates securely
        was_frozen = getattr(algorithm.config, "_is_frozen", False)
        if was_frozen:
            algorithm.config._is_frozen = False
            
        if hasattr(algorithm.config, "entropy_coeff"):
            algorithm.config.entropy_coeff = final_entropy
            algorithm.config.clip_param = clip_val
        else:
            algorithm.config["entropy_coeff"] = final_entropy
            algorithm.config["clip_param"] = clip_val
            
        if was_frozen:
            algorithm.config._is_frozen = True
            
        def _update_worker(w):
            if not hasattr(w, "global_vars"): w.global_vars = {}
            w.global_vars["training_iteration"] = iteration
            if w.get_policy("main_policy"):
                w.get_policy("main_policy").config.update({
                    "entropy_coeff": final_entropy,
                    "clip_param": clip_val
                })
                
        algorithm.env_runner_group.foreach_env_runner(_update_worker)
        
        # CRITICAL FIX: Push the updates directly into the Learner's live policy
        learner_policy = algorithm.get_policy("main_policy")
        if learner_policy:
            if hasattr(learner_policy, "config"):
                learner_policy.config["entropy_coeff"] = final_entropy
                learner_policy.config["clip_param"] = clip_val
            # PPOTorchPolicy caches these as attributes, so we MUST overwrite them directly:
            if hasattr(learner_policy, "entropy_coeff"):
                learner_policy.entropy_coeff = final_entropy
        
        # 4. Gradient Norm Logging (FSP Cycle Variance)
        learner_stats = result.get("info", {}).get("learner", {}).get("main_policy", {}).get("learner_stats", {})
        if "grad_gnorm" in learner_stats:
            if "custom_metrics" not in result:
                result["custom_metrics"] = {}
            result["custom_metrics"]["fsp_cycle"] = iteration % (100 if iteration > 30000 else 50)
            result["custom_metrics"]["grad_gnorm"] = learner_stats["grad_gnorm"]

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "player_0":
        return "main_policy"
        
    iteration = worker.global_vars.get("training_iteration", 0) if worker and hasattr(worker, "global_vars") else 0
    
    # Decay heuristics from 20% to 5% by iteration 25000
    if iteration <= 25000:
        heuristic_prob = 0.20 - (0.15 * (iteration / 25000.0))
    else:
        heuristic_prob = 0.05
        
    fsp_prob = 1.0 - heuristic_prob
    
    r = random.random()
    if r > fsp_prob:
        # Heuristic bot
        h = random.random()
        if h < 1/6: return "rational_policy"
        elif h < 2/6: return "honest_policy"
        elif h < 3/6: return "aggressive_policy"
        elif h < 4/6: return "coward_policy"
        elif h < 5/6: return "terminator_policy"
        else: return "tracker_policy"
    else:
        # FSP bot (normalize r to [0, 1) within the FSP block)
        fsp_r = r / fsp_prob
        if fsp_r < 0.20: return "main_policy"
        elif fsp_r < 0.28: return "past_policy_1"
        elif fsp_r < 0.36: return "past_policy_2"
        elif fsp_r < 0.44: return "past_policy_3"
        elif fsp_r < 0.52: return "past_policy_4"
        elif fsp_r < 0.60: return "past_policy_5"
        elif fsp_r < 0.68: return "past_policy_6"
        elif fsp_r < 0.76: return "past_policy_7"
        elif fsp_r < 0.84: return "past_policy_8"
        elif fsp_r < 0.92: return "past_policy_9"
        else: return "past_policy_10"

def env_creator(config):
    env = CoupEnv()
    return PettingZooEnv(env)

register_env("coup_parallel_v0", env_creator)
ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)

def setup_rllib_config(env_name="coup_parallel_v0", num_workers=6, use_pbt=False, start_iter=0):
    dummy_env = env_creator({})
    obs_space = dummy_env.observation_space["player_0"]
    act_space = dummy_env.action_space["player_0"]

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={"max_moves": 200}
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(num_env_runners=num_workers, num_envs_per_env_runner=10)
        .training(
            num_sgd_iter=5,
            gamma=0.995,
            lambda_=0.99,
            train_batch_size=6000,
            minibatch_size=1200,
            grad_clip=1.0,
            lr_schedule=[
                [0, 1e-4],
                [6000000, 1e-4],
                [150000000, 5e-5],
                [300000000, 1e-5]
            ],
            model={
                "custom_model": "coup_mask_lstm",
                "max_seq_len": 150, 
            }
        )
        .callbacks(CoupTrainingCallback)
        .multi_agent(
            policies={
                "main_policy": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_1": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_2": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_3": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_4": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_5": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_6": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_7": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_8": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_9": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_10": PolicySpec(observation_space=obs_space, action_space=act_space),
                "random_policy": PolicySpec(policy_class=RandomHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "honest_policy": PolicySpec(policy_class=HonestHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "rational_policy": PolicySpec(policy_class=RationalHeuristicPolicy, observation_space=obs_space, action_space=act_space),
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

    start_iter = 0
    if latest_ckpt:
        start_iter = int(latest_ckpt.split("_")[-1])

    config = setup_rllib_config(start_iter=start_iter)
    
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
    target_iter = 50000
    
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

        if i % 2000 == 0:
            current_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{algo.iteration}")
            checkpoint_path = algo.save(current_checkpoint_dir)
            print(f"=== Saved Checkpoint at Iteration {algo.iteration} to: {checkpoint_path} ===")

        rotation_period = 100 if i > 30000 else 50
        if i % rotation_period == 0:
            print(f"=== Rotating Policies (Fictitious Self-Play) ===")
            main_weights = algo.get_policy("main_policy").get_weights()
            
            if i <= 2000:
                # Warmup phase: Pure trailing window to seed anchors without garbage
                algo.get_policy("past_policy_10").set_weights(algo.get_policy("past_policy_9").get_weights())
                algo.get_policy("past_policy_9").set_weights(algo.get_policy("past_policy_8").get_weights())
                algo.get_policy("past_policy_8").set_weights(algo.get_policy("past_policy_7").get_weights())
                algo.get_policy("past_policy_7").set_weights(algo.get_policy("past_policy_6").get_weights())
                algo.get_policy("past_policy_6").set_weights(algo.get_policy("past_policy_5").get_weights())
                algo.get_policy("past_policy_5").set_weights(algo.get_policy("past_policy_4").get_weights())
            else:
                # Post-warmup: Immutable Anchors and Probabilistic Medium Anchors
                # Freeze specific anchors permanently at exact milestones
                if i == 10000:
                    algo.get_policy("past_policy_8").set_weights(main_weights)
                if i == 25000:
                    algo.get_policy("past_policy_9").set_weights(main_weights)
                if i == 40000:
                    algo.get_policy("past_policy_10").set_weights(main_weights)
                    
                # Medium-term probabilistic anchors
                if random.random() < 0.20:
                    algo.get_policy("past_policy_7").set_weights(algo.get_policy("past_policy_6").get_weights())
                    algo.get_policy("past_policy_6").set_weights(algo.get_policy("past_policy_5").get_weights())
                    algo.get_policy("past_policy_5").set_weights(algo.get_policy("past_policy_4").get_weights())
                    
            # Always rotate recent snapshots
            algo.get_policy("past_policy_4").set_weights(algo.get_policy("past_policy_3").get_weights())
            algo.get_policy("past_policy_3").set_weights(algo.get_policy("past_policy_2").get_weights())
            algo.get_policy("past_policy_2").set_weights(algo.get_policy("past_policy_1").get_weights())
            algo.get_policy("past_policy_1").set_weights(main_weights)

    log_file.close()
    print("Training Complete.")
    ray.shutdown()

if __name__ == "__main__":
    train_coup()
