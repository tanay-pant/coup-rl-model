import numpy as np


class RandomBot:
    def act(self, obs_dict):
        # Extract the 38-value binary mask
        action_mask = obs_dict["action_mask"]

        # Get an array of indices where the mask equals 1
        legal_actions = np.where(action_mask == 1)[0]

        # Uniformly sample one of the legal indices
        return np.random.choice(legal_actions)
