import copy
import random
import time
import math
import numpy as np
import torch
from envs.coup.game_logic import Role

class NeuralNode:
    def __init__(self, parent=None, move=None, active_player=None, prior_prob=0.0):
        self.parent = parent
        self.move = move
        self.active_player = active_player
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.prior_prob = prior_prob
        self.untried_moves = None
        self.lstm_states = {} # Map agent_str -> lstm_state
        
class NeuralMCTSBot:
    def __init__(self, agent_id, loaded_policy, num_simulations=800, max_time=0.8, c_puct=2.5):
        self.agent_id = agent_id
        self.loaded_policy = loaded_policy
        self.model = loaded_policy.model if loaded_policy else None
        self.num_simulations = num_simulations
        self.max_time = max_time
        self.c_puct = c_puct
        self.seq_lens = torch.tensor([1])
        
    def _forward_to_active(self, env):
        while not self.is_terminal(env):
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                break
                
    def _get_neural_eval(self, env, lstm_state=None):
        agent_str = env.agent_selection
        obs = env.observe(agent_str)
        
        input_dict = {
            "obs": {
                "observation": torch.tensor([obs["observation"]], dtype=torch.float32),
                "action_mask": torch.tensor([obs["action_mask"]], dtype=torch.float32)
            }
        }
        
        if lstm_state is None:
            state = [torch.tensor([s], dtype=torch.float32) for s in self.loaded_policy.get_initial_state()]
        else:
            state = lstm_state
            
        with torch.no_grad():
            logits, state_out = self.model(input_dict, state, self.seq_lens)
            value = self.model.value_function()[0].item()
            
        mask = obs["action_mask"]
        legal_moves = [i for i, m in enumerate(mask) if m == 1]
        
        logits = logits[0].numpy()
        logits[mask == 0] = -1e9
        
        # Softmax over legal moves only to get prior probabilities
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        action_probs = {m: probs[m] for m in legal_moves}
        
        return action_probs, value, state_out

    def compute_action(self, env, current_lstm_states=None):
        agent_str = f"player_{self.agent_id}"
        
        obs = env.observe(agent_str)
        action_mask = obs["action_mask"]
        legal_moves = [i for i, m in enumerate(action_mask) if m == 1]
        
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        root = NeuralNode(active_player=agent_str)
        if current_lstm_states is not None:
            root.lstm_states = current_lstm_states
            
        start_time = time.time()
        sims = 0
        
        while sims < self.num_simulations and (time.time() - start_time) < self.max_time:
            # 1. Determinize
            sim_env = self.clone_and_determinize(env, self.agent_id)
            self._forward_to_active(sim_env)
            
            # 2. Select (PUCT)
            node = root
            while not self.is_terminal(sim_env) and node.untried_moves == [] and node.children != []:
                node = self.select_child_puct(node)
                sim_env.step(node.move)
                self._forward_to_active(sim_env)
                
            # 3. Expand & Neural Evaluate
            if not self.is_terminal(sim_env):
                active_agent = sim_env.agent_selection
                
                # Get neural evaluation for the active agent
                current_state = node.lstm_states.get(active_agent)
                action_probs, value, new_state = self._get_neural_eval(sim_env, current_state)
                
                # Expand all children and assign prior probabilities
                node.untried_moves = []
                for move, prob in action_probs.items():
                    child = NeuralNode(parent=node, move=move, active_player=active_agent, prior_prob=prob)
                    
                    # Copy over all lstm states from parent
                    child.lstm_states = {k: [s.clone() for s in v] for k, v in node.lstm_states.items()}
                    # Update the state for the agent that just acted
                    child.lstm_states[active_agent] = new_state
                    
                    node.children.append(child)
                    
                # In PUCT, we don't randomly pick an untried move during expansion.
                # We immediately evaluate the node and backpropagate the value.
                
            else:
                # Terminal state, backpropagate objective win/loss
                winners = self.get_winners(sim_env)
                value = 1.0 if sim_env.agent_selection in winners else -1.0
                active_agent = sim_env.agent_selection # For terminal, this might be arbitrary, so rely on winners list

            # 4. Backpropagate
            self.backpropagate(node, sim_env, value, active_agent)
            
            sims += 1
            
        print(f"NeuralMCTS Agent {self.agent_id} achieved {sims} simulations in {time.time() - start_time:.2f}s")
        
        if not root.children:
            print(f"WARNING: NeuralMCTS Agent {self.agent_id} root.children is empty! Returning random move.")
            return random.choice(legal_moves)
            
        # Return action with highest visits
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
        
    def select_child_puct(self, node):
        best_score = -float('inf')
        best_children = []
        
        sqrt_N = math.sqrt(node.visits)
        
        for child in node.children:
            Q = child.wins / child.visits if child.visits > 0 else 0.0
            U = self.c_puct * child.prior_prob * sqrt_N / (1 + child.visits)
            
            score = Q + U
            
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
                
        return random.choice(best_children)
        
    def backpropagate(self, node, env, value, evaluator_agent):
        is_terminal = self.is_terminal(env)
        if is_terminal:
            winners = self.get_winners(env)
            
        curr = node
        discount = 0.99
        steps_up = 0
        
        winner_reward = env.unwrapped.winner_pot if hasattr(env, 'unwrapped') else 1.0
        if winner_reward == 0: winner_reward = 1.0
        
        alive_players = sum(1 for i, p in env.unwrapped.state.players.items() if p.influence_count > 0) if hasattr(env, 'unwrapped') else 2
        opponents = max(1, alive_players - 1)
        
        while curr is not None:
            curr.visits += 1
            discounted_val = discount ** steps_up
            
            if is_terminal:
                if curr.active_player in winners:
                    curr.wins += winner_reward * discounted_val
                else:
                    curr.wins -= 1.0 * discounted_val
            elif env.terminations.get(curr.active_player, False):
                curr.wins -= 1.0 * discounted_val
            else:
                if curr.active_player == evaluator_agent:
                    curr.wins += value
                else:
                    curr.wins -= (value / opponents)
                    
            curr = curr.parent
            steps_up += 1

    def is_terminal(self, env):
        return all(env.terminations.values())

    def get_winners(self, env):
        winners = []
        max_reward = max(env.rewards.values()) if env.rewards else 0
        if max_reward > 0:
             for a, r in env.rewards.items():
                  if r == max_reward:
                       winners.append(a)
        else:
            for a in env.agents:
                idx = int(a.split("_")[1])
                if env.state.players[idx].influence_count > 0:
                    winners.append(a)
        return winners

    def clone_and_determinize(self, env, observer_idx):
        sim_env = copy.deepcopy(env)
        
        roles = [Role.DUKE, Role.CAPTAIN, Role.ASSASSIN, Role.CONTESSA, Role.AMBASSADOR]
        full_deck = roles * 3
        
        for i, p in sim_env.state.players.items():
            for inf in p.influence:
                if inf.revealed and inf.role != Role.NONE:
                    try:
                        full_deck.remove(inf.role)
                    except ValueError:
                        pass
                    
        is_observer_exchanging = (sim_env.state.turn.phase == 5 and sim_env.state.turn.active_player == observer_idx)
        
        if is_observer_exchanging:
            for r in sim_env.state.turn.exchange_pool:
                if r != Role.NONE:
                    try:
                        full_deck.remove(r)
                    except ValueError:
                        pass
        else:
            observer_state = sim_env.state.players[observer_idx]
            for inf in observer_state.influence:
                if not inf.revealed and inf.role != Role.NONE:
                    try:
                        full_deck.remove(inf.role)
                    except ValueError:
                        pass
                    
        unknown_cards = full_deck
        random.shuffle(unknown_cards)
        
        opponents = [ (i, p) for i, p in sim_env.state.players.items() if i != observer_idx ]
        random.shuffle(opponents)
        
        for i, p in opponents:
            is_opp_exchanging = (sim_env.state.turn.phase == 5 and sim_env.state.turn.active_player == i)
            
            if is_opp_exchanging:
                active_idx = 0
                for j in range(len(sim_env.state.turn.exchange_pool)):
                    if sim_env.state.turn.exchange_pool[j] != Role.NONE:
                        if len(unknown_cards) > 0:
                            card = unknown_cards.pop()
                        else:
                            card = random.choice(roles)
                        sim_env.state.turn.exchange_pool[j] = card
                        
                        if active_idx < len(p.influence) and not p.influence[active_idx].revealed:
                            p.influence[active_idx].role = card
                            active_idx += 1
            else:
                claims = sim_env.active_claims[i].copy()
                proven_not = sim_env.proven_not_to_have[i]
                
                for inf in p.influence:
                    if not inf.revealed and inf.role != Role.NONE:
                        assigned = False
                        for role_val in range(5):
                            if claims[role_val] == 1 and proven_not[role_val] == 0:
                                target_role = Role(role_val)
                                if target_role in unknown_cards:
                                    inf.role = target_role
                                    unknown_cards.remove(target_role)
                                    assigned = True
                                    claims[role_val] = 0 
                                    break
                                    
                        if not assigned:
                            valid_cards = [c for c in unknown_cards if proven_not[c.value] == 0]
                            if valid_cards:
                                chosen = random.choice(valid_cards)
                                inf.role = chosen
                                unknown_cards.remove(chosen)
                            elif len(unknown_cards) > 0:
                                inf.role = unknown_cards.pop()
                            else:
                                inf.role = random.choice(roles)
                    
        sim_env.state.deck = unknown_cards
        return sim_env
