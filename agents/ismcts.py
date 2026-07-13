import copy
import random
import time
import math
import numpy as np
from envs.coup.game_logic import Role

class Node:
    def __init__(self, parent=None, move=None, active_player=None):
        self.parent = parent
        self.move = move
        self.active_player = active_player
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None

class ISMCTSBot:
    def __init__(self, agent_id, num_simulations=1000, max_time=1.0):
        self.agent_id = agent_id
        self.num_simulations = num_simulations
        self.max_time = max_time

    def _forward_to_active(self, env):
        while not self.is_terminal(env):
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                break

    def compute_action(self, env):
        agent_str = f"player_{self.agent_id}"
        
        # Get legal moves from observation
        obs = env.observe(agent_str)
        action_mask = obs["action_mask"]
        legal_moves = [i for i, m in enumerate(action_mask) if m == 1]
        
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        root = Node(active_player=agent_str)
        
        start_time = time.time()
        sims = 0
        
        while sims < self.num_simulations and (time.time() - start_time) < self.max_time:
            # 1. Determinize
            sim_env = self.clone_and_determinize(env, self.agent_id)
            self._forward_to_active(sim_env)
            
            # 2. Select
            node = root
            while not self.is_terminal(sim_env) and node.untried_moves == [] and node.children != []:
                node = self.select_child(node, sim_env)
                sim_env.step(node.move)
                self._forward_to_active(sim_env)
                
            # 3. Expand
            if not self.is_terminal(sim_env):
                if node.untried_moves is None:
                    node.untried_moves = self.get_legal_moves(sim_env)
                
                if node.untried_moves:
                    move = random.choice(node.untried_moves)
                    node.untried_moves.remove(move)
                    
                    active = sim_env.agent_selection
                    sim_env.step(move)
                    self._forward_to_active(sim_env)
                    
                    child = Node(parent=node, move=move, active_player=active)
                    node.children.append(child)
                    node = child
                    
            # 4. Simulate
            while not self.is_terminal(sim_env):
                self._forward_to_active(sim_env)
                if self.is_terminal(sim_env):
                    break
                moves = self.get_legal_moves(sim_env)
                move = random.choice(moves)
                sim_env.step(move)
                
            # 5. Backpropagate
            winners = self.get_winners(sim_env)
            while node is not None:
                node.visits += 1
                if node.active_player in winners:
                    node.wins += 1
                node = node.parent
                
            sims += 1
            
        print(f"Agent {self.agent_id} achieved {sims} simulations in {time.time() - start_time:.2f}s")
        # Select best move
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
        
    def select_child(self, node, sim_env):
        # UCB1
        best_score = -float('inf')
        best_children = []
        for child in node.children:
            exploit = child.wins / child.visits
            explore = math.sqrt(2 * math.log(node.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return random.choice(best_children)

    def is_terminal(self, env):
        # Check if all agents are terminated
        return all(env.terminations.values())

    def get_legal_moves(self, env):
        agent = env.agent_selection
        mask = env.observe(agent)["action_mask"]
        return [i for i, m in enumerate(mask) if m == 1]

    def get_winners(self, env):
        # Returns list of agent strings that won (usually just 1)
        winners = []
        max_reward = max(env.rewards.values()) if env.rewards else 0
        if max_reward > 0:
             for a, r in env.rewards.items():
                  if r == max_reward:
                       winners.append(a)
        else:
            # If no clear winner by reward, check who is alive
            for a in env.agents:
                idx = int(a.split("_")[1])
                if env.state.players[idx].influence_count > 0:
                    winners.append(a)
        return winners

    def clone_and_determinize(self, env, observer_idx):
        sim_env = copy.deepcopy(env)
        
        # 1. Count all roles in a fresh deck
        roles = [Role.DUKE, Role.CAPTAIN, Role.ASSASSIN, Role.CONTESSA, Role.AMBASSADOR]
        full_deck = roles * 3
        
        # 2. Remove cards that the observer CAN see
        # - Revealed dead cards of all players
        for i, p in sim_env.state.players.items():
            for inf in p.influence:
                if inf.revealed and inf.role != Role.NONE:
                    try:
                        full_deck.remove(inf.role)
                    except ValueError:
                        pass # Should not happen unless there's a bug in game logic
                    
        # - Observer's hidden cards
        is_observer_exchanging = (sim_env.state.turn.phase == 5 and sim_env.state.turn.active_player == observer_idx)
        
        if is_observer_exchanging:
            # Observer's active cards are in the exchange pool, along with drawn cards
            for r in sim_env.state.turn.exchange_pool:
                if r != Role.NONE:
                    try:
                        full_deck.remove(r)
                    except ValueError:
                        pass
        else:
            # Observer is not exchanging, just remove their own hidden cards
            observer_state = sim_env.state.players[observer_idx]
            for inf in observer_state.influence:
                if not inf.revealed and inf.role != Role.NONE:
                    try:
                        full_deck.remove(inf.role)
                    except ValueError:
                        pass
                    
        # full_deck now contains exactly the pool of UNKNOWN cards
        unknown_cards = full_deck
        random.shuffle(unknown_cards)
        
        # 3. Re-assign unknown cards to where they belong
        for i, p in sim_env.state.players.items():
            if i == observer_idx:
                continue
                
            is_opp_exchanging = (sim_env.state.turn.phase == 5 and sim_env.state.turn.active_player == i)
            
            if is_opp_exchanging:
                # Opponent is exchanging. Fill their exchange pool, and sync their influence to match
                active_idx = 0
                for j in range(len(sim_env.state.turn.exchange_pool)):
                    if sim_env.state.turn.exchange_pool[j] != Role.NONE:
                        if len(unknown_cards) > 0:
                            card = unknown_cards.pop()
                        else:
                            card = random.choice(roles) # Fallback just in case
                        
                        sim_env.state.turn.exchange_pool[j] = card
                        
                        # Sync back to influence for consistency
                        if active_idx < len(p.influence) and not p.influence[active_idx].revealed:
                            p.influence[active_idx].role = card
                            active_idx += 1
            else:
                # Opponent is not exchanging, just fill their hidden cards
                # First, try to honor active_claims
                claims = sim_env.active_claims[i]
                proven_not = sim_env.proven_not_to_have[i]
                
                for inf in p.influence:
                    if not inf.revealed and inf.role != Role.NONE:
                        assigned = False
                        
                        # Try to give them a card they claimed
                        for role_val in range(5):
                            if claims[role_val] == 1 and proven_not[role_val] == 0:
                                target_role = Role(role_val)
                                if target_role in unknown_cards:
                                    inf.role = target_role
                                    unknown_cards.remove(target_role)
                                    assigned = True
                                    # Temporarily unset the claim so we don't assign two of the same if they only claimed one
                                    claims[role_val] = 0 
                                    break
                                    
                        if not assigned:
                            # Try to give them a random card they are NOT proven to lack
                            valid_cards = [c for c in unknown_cards if proven_not[c.value] == 0]
                            if valid_cards:
                                chosen = random.choice(valid_cards)
                                inf.role = chosen
                                unknown_cards.remove(chosen)
                            elif len(unknown_cards) > 0:
                                # Fallback if forced
                                inf.role = unknown_cards.pop()
                            else:
                                inf.role = random.choice(roles)
                    
        # - The rest go into the deck
        sim_env.state.deck = unknown_cards
        
        return sim_env
