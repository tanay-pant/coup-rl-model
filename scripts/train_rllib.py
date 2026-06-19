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
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import random

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from envs.coup.coup_env import CoupEnv

from agents.league_policies import RandomHeuristicPolicy, HonestHeuristicPolicy, AggressiveHeuristicPolicy

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "player_0":
        return "main_policy"
    r = random.random()
    if r < 0.4:
        return "main_policy"
    elif r < 0.6:
        return "past_policy_1"
    elif r < 0.8:
        return "past_policy_2"
    elif r < 0.866:
        return "honest_policy"
    elif r < 0.933:
        return "aggressive_policy"
    else:
        return "random_policy"

def env_creator(config):
    """
    RLlib needs a factory function that creates and returns our environment.
    We pass our dynamic AEC env directly to Ray's native PettingZoo wrapper.
    """
    env = CoupEnv()
    return PettingZooEnv(env)

# Register the environment with Ray's global registry so RLlib can find it by name
register_env("coup_parallel_v0", env_creator)


def setup_rllib_config(env_name="coup_parallel_v0", num_workers=6, use_pbt=False):
    """
    Configures the PPO algorithm, hardware scaling, and Multi-Agent Policy mapping.
    """
    # 1. Create a dummy environment just to extract the tensor shapes
    dummy_env = env_creator({})
    
    # In PettingZoo, action/obs spaces are dicts keyed by agent. 
    # We just grab the shapes from player_0 to define our neural network layout.
    obs_space = dummy_env.observation_space["player_0"]
    act_space = dummy_env.action_space["player_0"]

    # 2. Build the PPO Configuration
    config = (
        PPOConfig()
        .environment(env=env_name)

        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        
        # CPU Scaling: How many parallel environment copies to run
        .env_runners(num_env_runners=num_workers, num_envs_per_env_runner=10)
        
        # GPU / Tensor batching constraints
        .training(
            train_batch_size=6000,
            minibatch_size=600,
            entropy_coeff_schedule=None if use_pbt else [[0, 0.2], [36000000, 0.01]],
            entropy_coeff=0.2 if use_pbt else 0.0,
            model={
                "custom_model": "coup_mask_model",
            }
        )
        
        .multi_agent(
            policies={
                "main_policy": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_1": PolicySpec(observation_space=obs_space, action_space=act_space),
                "past_policy_2": PolicySpec(observation_space=obs_space, action_space=act_space),
                "random_policy": PolicySpec(policy_class=RandomHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "honest_policy": PolicySpec(policy_class=HonestHeuristicPolicy, observation_space=obs_space, action_space=act_space),
                "aggressive_policy": PolicySpec(policy_class=AggressiveHeuristicPolicy, observation_space=obs_space, action_space=act_space),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main_policy"]
        )
    )
    return config

class CoupActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_dim = 209 

        self.core_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )
        
        # The Critic (Value) Network
        self.value_branch = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._current_value = None

    def forward(self, input_dict, state, seq_lens):
        # A. Extract the tensors (RLlib handles the batching automatically)
        # Shapes will be [batch_size, 131] and [batch_size, 38]
        obs = input_dict["obs"]["observation"]
        action_mask = input_dict["obs"]["action_mask"]

        # B. Forward pass through the Actor network
        raw_logits = self.core_network(obs)

        # C. Forward pass through the Critic network
        self._current_value = self.value_branch(obs).squeeze(1)

        # D. The Masking Math
        # Convert valid actions (1) to 0.0
        # Convert invalid actions (0) to -inf, clamped to -1,000,000,000
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e9)
        
        # Add the mask to the raw logits
        masked_logits = raw_logits + inf_mask

        # Prevent NaN if action_mask is entirely zeros (e.g. dead players)
        all_masked = (action_mask.sum(dim=-1, keepdim=True) == 0.0)
        masked_logits = torch.where(
            all_masked,
            torch.zeros_like(masked_logits),
            masked_logits
        )

        return masked_logits, state

    def value_function(self):
        return self._current_value

# Register the model with RLlib
ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)

def train_coup():
    ray.init()

    # Setup checkpoint directory
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_rllib'))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build the algorithm from our config (from Chunk 2)
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
    if latest_ckpt:
        from ray.rllib.algorithms.algorithm import Algorithm
        algo = Algorithm.from_checkpoint(latest_ckpt)
        print(f"Resuming training from {latest_ckpt}...")
    else:
        algo = config.build_algo()

    import csv
    mode = "a" if latest_ckpt else "w"
    log_file = open("training_log.csv", mode, newline="")
    csv_writer = csv.writer(log_file)
    if mode == "w":
        csv_writer.writerow(["Iteration", "Mean Reward", "Policy Loss", "Value Loss", "Entropy"])

    print("Starting Multi-Agent PPO Training on Coup...")
    
    start_iter = algo.iteration if hasattr(algo, 'iteration') else 0
    for i in range(start_iter + 1, 6001):
        # algo.train() triggers the rollout workers to play games, 
        # gather batches, and run the PyTorch backpropagation.
        result = algo.train()
        
        # Extract the mean reward across all agents for this iteration
        # Note: In Ray 2.x, metrics are often nested under 'env_runners'
        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", 
                      result.get("episode_reward_mean", 0.0))
        
        # Extract advanced metrics
        learner_stats = result.get("info", {}).get("learner", {}).get("main_policy", {}).get("learner_stats", {})
        # Note: If RLlib uses a different dict structure based on Ray version, these might default to 0.0
        policy_loss = learner_stats.get("policy_loss", 0.0)
        vf_loss = learner_stats.get("vf_loss", 0.0)
        entropy = learner_stats.get("entropy", 0.0)
        
        print(f"Iteration {i:03d} | Reward: {mean_reward:7.4f} | Policy Loss: {policy_loss:7.4f} | VF Loss: {vf_loss:7.4f} | Entropy: {entropy:7.4f}")
        csv_writer.writerow([i, mean_reward, policy_loss, vf_loss, entropy])
        log_file.flush()

        # Save a checkpoint every 1000 iterations
        if i % 1000 == 0:
            current_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{algo.iteration}")
            checkpoint_path = algo.save(current_checkpoint_dir)
            print(f"=== Saved Checkpoint at Iteration {algo.iteration} to: {checkpoint_path} ===")

        if i % 50 == 0:
            print(f"=== Rotating Policies (Fictitious Self-Play) ===")
            main_weights = algo.get_policy("main_policy").get_weights()
            if random.random() < 0.2:
                past_1_weights = algo.get_policy("past_policy_1").get_weights()
                algo.get_policy("past_policy_2").set_weights(past_1_weights)
            algo.get_policy("past_policy_1").set_weights(main_weights)

    log_file.close()
    print("Training Complete.")
    ray.shutdown()

if __name__ == "__main__":
    train_coup()