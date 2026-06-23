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
                flat_obs = obs["observation"]
            else:
                action_mask = obs[:38]
                flat_obs = obs[38:]
                
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
                continue
                
            # Determine held roles from observation (Indices 2..6 and 7..11)
            has_duke = flat_obs[2] > 0.5 or flat_obs[7] > 0.5
            has_assassin = flat_obs[3] > 0.5 or flat_obs[8] > 0.5
            has_captain = flat_obs[4] > 0.5 or flat_obs[9] > 0.5
            has_ambassador = flat_obs[5] > 0.5 or flat_obs[10] > 0.5
            has_contessa = flat_obs[6] > 0.5 or flat_obs[11] > 0.5
            
            honest_moves = []
            for a in legal_actions:
                # Guaranteed honest actions
                if a in [0, 1, 23] or (34 <= a <= 48):
                    honest_moves.append(a)
                # Tax
                elif a == 2 and has_duke:
                    honest_moves.append(a)
                # Exchange
                elif a == 3 and has_ambassador:
                    honest_moves.append(a)
                # Steal
                elif 4 <= a <= 9 and has_captain:
                    honest_moves.append(a)
                # Assassinate
                elif 10 <= a <= 15 and has_assassin:
                    honest_moves.append(a)
                # Coup
                elif 16 <= a <= 21:
                    honest_moves.append(a)
                # Block(Duke)
                elif a == 24 and has_duke:
                    honest_moves.append(a)
                # Block(Captain)
                elif a == 25 and has_captain:
                    honest_moves.append(a)
                # Block(Assassin) - Wait, nobody blocks with Assassin. Action 26 is Block(Assassin)? No, Contessa is 27
                elif a == 26: 
                    pass # Not a real block
                elif a == 27 and has_contessa:
                    honest_moves.append(a)
                elif a == 28 and has_ambassador:
                    honest_moves.append(a)
                # Reveals
                elif a == 29 and has_duke: honest_moves.append(a)
                elif a == 30 and has_assassin: honest_moves.append(a)
                elif a == 31 and has_captain: honest_moves.append(a)
                elif a == 32 and has_ambassador: honest_moves.append(a)
                elif a == 33 and has_contessa: honest_moves.append(a)
                # The Honest bot NEVER challenges unless it's impossible for the opponent to have the card
                # But to keep it simple, it just passes challenges instead of tracking card counts.
                # So we omit Challenge (22).
                
            if len(honest_moves) > 0:
                actions.append(int(np.random.choice(honest_moves)))
            else:
                actions.append(int(np.random.choice(legal_actions)))
                
        return np.array(actions), [], {}

class AggressiveHeuristicPolicy(RandomHeuristicPolicy):
    # This acts as the "Calling Station" - it ALWAYS challenges if it can, 
    # and ALWAYS takes aggressive actions (Assassinate/Steal) if possible.
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
                continue
                
            if 22 in legal_actions:
                # ALWAYS CHALLENGE
                actions.append(22)
            else:
                aggro_moves = [a for a in legal_actions if (4 <= a <= 21)]
                if len(aggro_moves) > 0:
                    actions.append(int(np.random.choice(aggro_moves)))
                else:
                    actions.append(int(np.random.choice(legal_actions)))
                    
        return np.array(actions), [], {}

class CowardHeuristicPolicy(RandomHeuristicPolicy):
    def compute_actions(self, obs_batch, *args, **kwargs):
        actions = []
        for obs in obs_batch:
            if isinstance(obs, dict): action_mask = obs["action_mask"]
            else: action_mask = obs[:38]
                
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
                continue
                
            # Coward only takes safe actions and never blocks/challenges unless forced
            safe_moves = [a for a in legal_actions if a in [0, 1, 2] or (34 <= a <= 48) or a in [23] or (29 <= a <= 33)]
            if len(safe_moves) > 0:
                actions.append(int(np.random.choice(safe_moves)))
            else:
                actions.append(int(np.random.choice(legal_actions)))
        return np.array(actions), [], {}

class TerminatorHeuristicPolicy(RandomHeuristicPolicy):
    def compute_actions(self, obs_batch, *args, **kwargs):
        actions = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                action_mask = obs["action_mask"]
                flat_obs = obs["observation"]
            else:
                action_mask = obs[:38]
                flat_obs = obs[38:]
                
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
                continue
                
            has_assassin = flat_obs[3] > 0.5 or flat_obs[8] > 0.5
            has_captain = flat_obs[4] > 0.5 or flat_obs[9] > 0.5
            
            # Terminator does not challenge, it only attacks
            if 23 in legal_actions: # Pass block/challenge
                actions.append(23)
                continue
            
            # Forced Reveal or Exchange Return
            if any(a >= 29 for a in legal_actions):
                actions.append(int(np.random.choice([a for a in legal_actions if a >= 29])))
                continue
            
            aggro_moves = []
            if has_assassin:
                aggro_moves.extend([a for a in legal_actions if 10 <= a <= 14])
            if has_captain:
                aggro_moves.extend([a for a in legal_actions if 4 <= a <= 8])
            
            if len(aggro_moves) > 0:
                actions.append(int(np.random.choice(aggro_moves)))
            elif 3 in legal_actions:
                actions.append(3) # Exchange to find a weapon
            else:
                actions.append(int(np.random.choice(legal_actions)))
                
        return np.array(actions), [], {}

class TrackerHeuristicPolicy(RandomHeuristicPolicy):
    def compute_actions(self, obs_batch, *args, **kwargs):
        actions = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                action_mask = obs["action_mask"]
                flat_obs = obs["observation"]
            else:
                action_mask = obs[:38]
                flat_obs = obs[38:]
                
            legal_actions = np.where(action_mask > 0.5)[0]
            if len(legal_actions) == 0:
                actions.append(0)
                continue
                
            filtered_legal = []
            for a in legal_actions:
                # Steal Action is 4..8 mapping to offset 1..5
                if 4 <= a <= 8:
                    offset = a - 4 + 1
                    # claims are at 107 + (offset-1)*5
                    # Wait, no. ego_claims[0] is ego. ego_claims[1] is offset 1.
                    # so offset 1 is index 107 + 5*1.
                    captain_claim = flat_obs[107 + offset*5 + 2]
                    amb_claim = flat_obs[107 + offset*5 + 3]
                    if captain_claim > 0 or amb_claim > 0:
                        continue # DO NOT STEAL FROM BLOCKER
                filtered_legal.append(a)
                
            if len(filtered_legal) == 0:
                filtered_legal = legal_actions
                
            actions.append(int(np.random.choice(filtered_legal)))
        return np.array(actions), [], {}
