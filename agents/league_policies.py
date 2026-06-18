import numpy as np
from ray.rllib.policy.policy import Policy

class RandomHeuristicPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        actions = []
        for obs in obs_batch:
            # If obs is a dict, extract it directly. If flattened array, action_mask is the first 38 elements (alphabetical sorting).
            if isinstance(obs, dict):
                action_mask = obs["action_mask"]
            else:
                action_mask = obs[:38]
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
            else:
                actions.append(int(np.random.choice(legal_actions)))
        return np.array(actions), [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

class HonestHeuristicPolicy(RandomHeuristicPolicy):
    def compute_actions(self, obs_batch, *args, **kwargs):
        actions = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                action_mask = obs["action_mask"]
            else:
                action_mask = obs[:38]
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
            else:
                # Prefer safe moves: Income(0), FA(1), Tax(2), Exchange(3), Allow(23), Reveal(29-33), Exchange Return(34-37)
                safe_moves = [a for a in legal_actions if a in [0, 1, 2, 3, 23] or a >= 29]
                if len(safe_moves) > 0:
                    actions.append(int(np.random.choice(safe_moves)))
                else:
                    actions.append(int(np.random.choice(legal_actions)))
        return np.array(actions), [], {}

class AggressiveHeuristicPolicy(RandomHeuristicPolicy):
    def compute_actions(self, obs_batch, *args, **kwargs):
        actions = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                action_mask = obs["action_mask"]
            else:
                action_mask = obs[:38]
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
            else:
                # Prefer Coup, Assassinate, Steal, Challenge
                aggro_moves = [a for a in legal_actions if (4 <= a <= 22)]
                if len(aggro_moves) > 0:
                    actions.append(int(np.random.choice(aggro_moves)))
                else:
                    actions.append(int(np.random.choice(legal_actions)))
        return np.array(actions), [], {}
