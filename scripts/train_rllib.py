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

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from envs.coup.coup_env import CoupEnv

def env_creator(config):
    """
    RLlib needs a factory function that creates and returns our environment.
    We pass our dynamic AEC env directly to Ray's native PettingZoo wrapper.
    """
    env = CoupEnv()
    return PettingZooEnv(env)

# Register the environment with Ray's global registry so RLlib can find it by name
register_env("coup_parallel_v0", env_creator)


def setup_rllib_config(env_name="coup_parallel_v0", num_workers=4):
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
        .env_runners(num_env_runners=num_workers)
        
        # GPU / Tensor batching constraints
        .training(
            train_batch_size=4000,
            minibatch_size=512,
            entropy_coeff=0.05,
            model={
                "custom_model": "coup_mask_model",
            }
        )
        
        # 3. Multi-Agent Policy Setup
        .multi_agent(
            policies={
                # We define a single brain called "shared_policy"
                "shared_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                )
            },
            # We map every player in the game to use this exact same brain
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy"
        )
    )
    return config

class CoupActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_dim = 131 

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

        return masked_logits, state

    def value_function(self):
        return self._current_value

# Register the model with RLlib
ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)

def train_coup():
    ray.init()

    # Build the algorithm from our config (from Chunk 2)
    config = setup_rllib_config()
    algo = config.build()

    # Setup checkpoint directory
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Starting Multi-Agent PPO Training on Coup...")
    
    # The Training Loop
    for i in range(1, 101):  # Running for 100 iterations as a test
        # algo.train() triggers the rollout workers to play games, 
        # gather batches, and run the PyTorch backpropagation.
        result = algo.train()
        
        # Extract the mean reward across all agents for this iteration
        # Note: In Ray 2.x, metrics are often nested under 'env_runners'
        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", 
                      result.get("episode_reward_mean", 0.0))
        
        print(f"Iteration {i}: Mean Reward = {mean_reward:.4f}")

        # Save Model Checkpoint every 20 iterations
        if i % 20 == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"=== Saved Checkpoint at Iteration {i} to: {checkpoint_path} ===")

    print("Training Complete.")
    ray.shutdown()

if __name__ == "__main__":
    train_coup()