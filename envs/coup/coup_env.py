from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import random

from .game_logic import GameState, PlayerState, TurnState, Influence, Phase, Role, Action


def env(render_mode=None):
    env = CoupEnv(render_mode=render_mode)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class CoupEnv(AECEnv):
    metadata = {'render_modes': ['human'], "name": "coup_v0", "is_parallelizable": True}

    def __init__(self, render_mode=None, max_moves=200):
        super().__init__()
        self.render_mode = render_mode
        self.max_moves = max_moves
        self.MAX_PLAYERS = 6
        self.num_players = 3  # Default value, gets randomized in reset()

        self.possible_agents = [f"player_{i}" for i in range(self.MAX_PLAYERS)]

        # 38 Total Discrete Actions
        self.action_spaces = {
            agent: Discrete(38) for agent in self.possible_agents
        }

        # 214-value Observation Array (includes 10-turn event log and global dead counts) and 38-value Action Mask
        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=-np.inf, high=np.inf, shape=(184,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(38,), dtype=np.int8)
            })
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_players = random.randint(3, self.MAX_PLAYERS)
        self.agents = self.possible_agents[:self.num_players]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.active_claims = np.zeros((self.MAX_PLAYERS, 5), dtype=np.int32)
        self.proven_not_to_have = np.zeros((self.MAX_PLAYERS, 5), dtype=np.int32)
        self.grudges = np.zeros((self.MAX_PLAYERS, self.MAX_PLAYERS), dtype=np.int32)
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.players_eliminated = 0
        self.winner_pot = 0.0
        self.elimination_step = 0.9 / max(1, self.num_players - 2)
        self.num_moves = 0

        # Initialize Game State
        self.state = GameState(num_players=self.num_players)
        self.state.setup_base_game()
        random.shuffle(self.state.deck)

        # Deal Cards
        for i in range(self.num_players):
            role1 = self.state.deck.pop()
            role2 = self.state.deck.pop()
            self.state.players[i].influence = [
                Influence(role=role1),
                Influence(role=role2)
            ]

        # Set initial turn phase
        self.state.turn.phase = Phase.START_OF_TURN
        self.state.turn.active_player = random.randint(0, self.num_players - 1)
        self.state.turn.exchange_pool = []
        self.state.turn.exchange_returns_left = 0

        self.agent_selection = f"player_{self.state.turn.active_player}"
        if hasattr(self, "_stored_agent_selection"):
            del self._stored_agent_selection

    def observe(self, agent):
        """
        Translates the GameState into a flat 184-value NumPy array for the neural network,
        and generates the 38-value binary Action Mask.
        """
        agent_idx = int(agent.split("_")[1])
        my_state = self.state.players.get(agent_idx, PlayerState(cash=0, influence_count=0))
        obs = []

        obs.append(my_state.cash)
        obs.append(my_state.influence_count)

        for i in range(2):
            card_encoding = [0] * 5
            if i < len(
                    my_state.influence) and not my_state.influence[i].revealed:
                role_idx = my_state.influence[i].role.value
                if role_idx != -1:
                    card_encoding[role_idx] = 1
            obs.extend(card_encoding)

        for offset in range(1, self.MAX_PLAYERS):
            i = (agent_idx + offset) % self.MAX_PLAYERS
            opp = self.state.players.get(
                i, PlayerState(cash=0, influence_count=0))
            obs.append(opp.cash)
            obs.append(opp.influence_count)

            for j in range(2):
                card_encoding = [0] * 5
                if j < len(opp.influence) and opp.influence[j].revealed:
                    role_idx = opp.influence[j].role.value
                    if role_idx != -1:
                        card_encoding[role_idx] = 1
                obs.extend(card_encoding)

        # Global State (9 values)
        phase_encoding = [0] * 7
        current_phase_idx = self.state.turn.phase.value
        phase_encoding[current_phase_idx] = 1
        obs.extend(phase_encoding)

        if self.state.turn.target is not None and self.state.turn.target != -1:
            ego_target = (self.state.turn.target - agent_idx) % self.MAX_PLAYERS
            obs.append(ego_target)
        else:
            obs.append(-1)
        obs.append(len(self.state.deck))

        # Active Player (6 values)
        active_player_encoding = [0] * self.MAX_PLAYERS
        if self.state.turn.active_player != -1:
            ego_active = (self.state.turn.active_player - agent_idx) % self.MAX_PLAYERS
            active_player_encoding[ego_active] = 1
        obs.extend(active_player_encoding)

        # Exchange Pool (20 values)
        can_see_exchange = (self.state.turn.active_player == agent_idx) and (self.state.turn.phase == Phase.EXCHANGE)
        for i in range(4):
            card_encoding = [0] * 5
            if can_see_exchange and i < len(
                    self.state.turn.exchange_pool) and self.state.turn.exchange_pool[i] != Role.NONE:
                role_idx = self.state.turn.exchange_pool[i].value
                if role_idx != -1:
                    card_encoding[role_idx] = 1
            obs.extend(card_encoding)

        # Active Claims (30 values)
        ego_claims = np.roll(self.active_claims, shift=-agent_idx, axis=0)
        obs.extend(ego_claims.flatten().tolist())

        # Is Over-Claiming (6 values)
        is_over_claiming = np.zeros(self.MAX_PLAYERS, dtype=np.int32)
        for i in range(self.MAX_PLAYERS):
            opp_state = self.state.players.get(i, PlayerState(cash=0, influence_count=0))
            if np.sum(self.active_claims[i]) > opp_state.influence_count:
                is_over_claiming[i] = 1
        ego_over_claiming = np.roll(is_over_claiming, shift=-agent_idx, axis=0)
        obs.extend(ego_over_claiming.flatten().tolist())

        # Proven Not To Have (30 values)
        ego_proven_not_to_have = np.roll(self.proven_not_to_have, shift=-agent_idx, axis=0)
        obs.extend(ego_proven_not_to_have.flatten().tolist())

        # Hostile Actions Against Me (Grudges) (6 values)
        my_grudges = self.grudges[agent_idx]
        ego_grudges = np.roll(my_grudges, shift=-agent_idx, axis=0)
        obs.extend(ego_grudges.flatten().tolist())

        # Global Dead (5 values)
        global_dead = [0, 0, 0, 0, 0]
        for p in self.state.players.values():
            for inf in p.influence:
                if inf.revealed and inf.role.value != -1:
                    global_dead[inf.role.value] += 1
        obs.extend(global_dead)

        obs_vector = np.array(obs, dtype=np.float32)
        action_mask = np.zeros(38, dtype=np.int8)

        if my_state.influence_count == 0:
            return {"observation": obs_vector, "action_mask": action_mask}

        # BASE TURN ACTIONS
        if self.state.turn.phase == Phase.START_OF_TURN and self.state.turn.active_player == agent_idx:
            if my_state.cash >= 10:
                # Must Coup
                for offset in range(1, self.MAX_PLAYERS):
                    i = (agent_idx + offset) % self.MAX_PLAYERS
                    if self.state.players.get(i, PlayerState(cash=0, influence_count=0)).influence_count > 0:
                        action_mask[16 + offset - 1] = 1
            else:
                action_mask[0] = 1  # Income
                action_mask[1] = 1  # Foreign Aid
                action_mask[2] = 1  # Tax
                action_mask[3] = 1  # Exchange

                for offset in range(1, self.MAX_PLAYERS):
                    i = (agent_idx + offset) % self.MAX_PLAYERS
                    opp_state = self.state.players.get(i, PlayerState(cash=0, influence_count=0))
                    if opp_state.influence_count > 0:
                        if opp_state.cash > 0:
                            action_mask[4 + offset - 1] = 1  # Steal from offset 1..5 -> Action 4..8
                        if my_state.cash >= 3:
                            action_mask[10 + offset - 1] = 1  # Assassinate -> Action 10..14
                        if my_state.cash >= 7:
                            action_mask[16 + offset - 1] = 1  # Coup -> Action 16..20

        # INTERVENTION ACTIONS (Responses to someone else's move)
        elif self.state.turn.phase in [Phase.ACTION_CHALLENGE, Phase.ACTION_BLOCK, Phase.BLOCK_RESPONSE]:
            claimer = self.state.turn.active_player if self.state.turn.phase in [Phase.ACTION_CHALLENGE, Phase.ACTION_BLOCK] else self.state.turn.target

            if claimer != agent_idx:
                action_mask[23] = 1  # Allow/Pass
                
                if self.state.turn.phase == Phase.ACTION_CHALLENGE:
                    action_mask[22] = 1  # Challenge
                    
                elif self.state.turn.phase == Phase.BLOCK_RESPONSE:
                    action_mask[22] = 1  # Challenge
                    
                elif self.state.turn.phase == Phase.ACTION_BLOCK:
                    if self.state.turn.action == 1:
                        action_mask[24] = 1  # Duke
                    elif self.state.turn.action in range(10, 15) and self.state.turn.target == agent_idx:
                        action_mask[27] = 1  # Contessa
                    elif self.state.turn.action in range(4, 9) and self.state.turn.target == agent_idx:
                        action_mask[25] = 1  # Captain
                        action_mask[28] = 1  # Ambassador

        # FORCED REVEAL
        elif self.state.turn.phase == Phase.REVEAL_INFLUENCE and self.state.turn.player_to_reveal == agent_idx:
            for inf in my_state.influence:
                if not inf.revealed:
                    role_idx = inf.role.value
                    if role_idx != -1:
                        action_mask[29 + role_idx] = 1

        # EXCHANGE PHASE
        elif self.state.turn.phase == Phase.EXCHANGE and self.state.turn.active_player == agent_idx:
            for i in range(len(self.state.turn.exchange_pool)):
                if self.state.turn.exchange_pool[i] != Role.NONE:
                    action_mask[34 + i] = 1

        return {"observation": obs_vector, "action_mask": action_mask}

    def _was_dead_step(self, action):
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")
        
        agent = self.agent_selection
        self.agents.remove(agent)
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        
        if len(self.agents) == 0:
            return
            
        for a in self.agents:
            if self.terminations[a]:
                self.agent_selection = a
                return

        if hasattr(self, "_stored_agent_selection"):
            stored = self._stored_agent_selection
            del self._stored_agent_selection
            if stored in self.agents:
                self.agent_selection = stored
                return
                
        self._next_turn()

    def step(self, action):
        """
        Executes actions, manages state transitions,
        handles player eliminations, and calculates RL rewards.
        """
        if (self.terminations[self.agent_selection] or
                self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return

        self._cumulative_rewards[self.agent_selection] = 0

        agent_idx = int(self.agent_selection.split("_")[1])

        # Log the event
        target = -1
        if action in range(4, 9):
            offset = action - 4 + 1
            target = (agent_idx + offset) % self.MAX_PLAYERS
        elif action in range(10, 15):
            offset = action - 10 + 1
            target = (agent_idx + offset) % self.MAX_PLAYERS
        elif action in range(16, 21):
            offset = action - 16 + 1
            target = (agent_idx + offset) % self.MAX_PLAYERS


        # State Machine Routing
        phase = self.state.turn.phase
        if phase == Phase.START_OF_TURN:
            self._handle_base_action(agent_idx, action)
        elif phase == Phase.ACTION_CHALLENGE:
            self._handle_challenge_response(agent_idx, action)
        elif phase == Phase.ACTION_BLOCK:
            self._handle_block_decision(agent_idx, action)
        elif phase == Phase.BLOCK_RESPONSE:
            self._handle_block_response(agent_idx, action)
        elif phase == Phase.REVEAL_INFLUENCE:
            self._handle_reveal(agent_idx, action)
        elif phase == Phase.EXCHANGE:
            self._handle_exchange(agent_idx, action)

        self._check_eliminations_and_victory()

        self.num_moves += 1
        if self.num_moves >= self.max_moves:
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.rewards[agent] -= 1.0
                    self.terminations[agent] = True
                    self.truncations[agent] = False

        self._accumulate_rewards()

    def _handle_base_action(self, player, action):
        p_state = self.state.players[player]
        target = None
        if action in range(4, 9):
            offset = action - 4 + 1
            target = (player + offset) % self.MAX_PLAYERS
        elif action in range(10, 15):
            offset = action - 10 + 1
            target = (player + offset) % self.MAX_PLAYERS
        elif action in range(16, 21):
            offset = action - 16 + 1
            target = (player + offset) % self.MAX_PLAYERS

        if action == 2:  # Tax -> Duke
            self._record_claim(player, Role.DUKE)
        elif action == 3:  # Exchange -> Ambassador
            self._record_claim(player, Role.AMBASSADOR)
        elif action in range(4, 9):  # Steal -> Captain
            self._record_claim(player, Role.CAPTAIN)
            self._record_grudge(initiator=player, target=target)
        elif action in range(10, 15):  # Assassinate -> Assassin
            self._record_claim(player, Role.ASSASSIN)
            self._record_grudge(initiator=player, target=target)

        if action == 0:  # Income
            p_state.cash += 1
            self._next_turn()
        elif action == 1:  # FA
            self.state.turn.phase = Phase.ACTION_BLOCK
            self.state.turn.action = 1
            self._open_block_window()
        elif action == 2:  # Tax
            self.state.turn.phase = Phase.ACTION_CHALLENGE
            self.state.turn.action = 2
            self._open_challenge_window(initiator=player)
        elif action == 3:  # Exchange
            self.state.turn.phase = Phase.ACTION_CHALLENGE
            self.state.turn.action = 3
            self._open_challenge_window(initiator=player)
        elif action in range(16, 21):  # Coup
            p_state.cash -= 7
            self.state.turn.phase = Phase.REVEAL_INFLUENCE
            self.state.turn.player_to_reveal = target
            self.agent_selection = f"player_{target}"
        elif action in range(10, 15):  # Assassinate
            p_state.cash -= 3
            self.state.turn.phase = Phase.ACTION_CHALLENGE
            self.state.turn.action = action
            self.state.turn.target = target
            self._open_challenge_window(initiator=player)
        elif action in range(4, 9):  # Steal
            self.state.turn.phase = Phase.ACTION_CHALLENGE
            self.state.turn.action = action
            self.state.turn.target = target
            self._open_challenge_window(initiator=player)

    def _handle_challenge_response(self, player, action):
        if action == 23:  # Allow/Pass
            self._advance_intervention_window()
            return
        if action == 22:  # Challenge
            self._resolve_challenge(challenger=player)
            return

    def _handle_block_decision(self, player, action):
        if action == 23:  # Allow/Pass
            self._advance_intervention_window()
            return

        if action in [24, 25, 27, 28]:  # Blocks
            roles = {
                24: Role.DUKE,
                25: Role.CAPTAIN,
                27: Role.CONTESSA,
                28: Role.AMBASSADOR}
            self._record_claim(player, roles[action])
            self._record_grudge(initiator=player, target=self.state.turn.active_player)
            self.state.turn.phase = Phase.BLOCK_RESPONSE
            self.state.turn.blocking_role = roles[action]
            self.state.turn.target = player
            self._open_challenge_window(initiator=player)
            return

    def _handle_block_response(self, player, action):
        if action == 23:  # Allow/Pass
            self._advance_intervention_window(block_accepted=True)
            return
        if action == 22:
            self._resolve_challenge(challenger=player, challenging_block=True)
            return


    def _handle_reveal(self, player, action):
        roles = {
            29: Role.DUKE,
            30: Role.ASSASSIN,
            31: Role.CAPTAIN,
            32: Role.AMBASSADOR,
            33: Role.CONTESSA}
        role_to_kill = roles.get(action, Role.NONE)

        if role_to_kill != Role.NONE and self.active_claims[player][role_to_kill.value] == 1:
            self.active_claims[player][role_to_kill.value] = 0

        p_state = self.state.players[player]
        for inf in p_state.influence:
            if not inf.revealed and inf.role == role_to_kill:
                inf.revealed = True
                p_state.influence_count -= 1
                break

        if self.state.turn.pending_action:
            self.state.turn.pending_action = False
            if self.state.turn.resuming_from_failed_block:
                self.state.turn.resuming_from_failed_block = False
                self._resolve_successful_action()
            elif self.state.turn.action in [1] + list(range(4, 15)):
                self.state.turn.phase = Phase.ACTION_BLOCK
                self._open_block_window()
            else:
                self._resolve_successful_action()
        else:
            self._next_turn()

    def _handle_exchange(self, player, action):
        pool_idx = action - 34
        if 0 <= pool_idx < len(self.state.turn.exchange_pool):
            role_to_return = self.state.turn.exchange_pool[pool_idx]
            if role_to_return != Role.NONE:
                self.state.deck.append(role_to_return)
                self.state.turn.exchange_pool[pool_idx] = Role.NONE
                self.state.turn.exchange_returns_left -= 1
                random.shuffle(self.state.deck)

        if self.state.turn.exchange_returns_left == 0:
            kept_cards = [
                card for card in self.state.turn.exchange_pool if card != Role.NONE]
            p_state = self.state.players[player]
            kept_idx = 0
            for inf in p_state.influence:
                if not inf.revealed:
                    inf.role = kept_cards[kept_idx]
                    kept_idx += 1
            self.state.turn.exchange_pool = [Role.NONE] * 4
            self.active_claims[player].fill(0)
            self.proven_not_to_have[player].fill(0)
            self._next_turn()

    # ==========================================
    # Turn Rotation Helpers
    # ==========================================

    def _next_turn(self):
        # Turn penalty to prevent stalling (much safer than step penalty)
        for agent_str in self.agents:
            i = int(agent_str.split("_")[1])
            if not self.terminations[agent_str] and self.state.players[i].influence_count > 0:
                self.rewards[agent_str] -= 0.005

        self.state.turn.phase = Phase.START_OF_TURN
        self.state.turn.action = -1
        self.state.turn.target = -1
        self.state.turn.blocking_role = Role.NONE
        self.state.turn.pending_action = False
        self.state.turn.resuming_from_failed_block = False

        for agent in self.agents:
            if self.terminations[agent]:
                self.agent_selection = agent
                return

        current = self.state.turn.active_player
        for i in range(1, self.num_players + 1):
            next_p = (current + i) % self.num_players
            if self.state.players[next_p].influence_count > 0:
                self.state.turn.active_player = next_p
                self.agent_selection = f"player_{next_p}"
                return

    def _open_challenge_window(self, initiator):
        self.intervention_queue = []
        for i in range(self.num_players):
            if i != initiator and self.state.players[i].influence_count > 0:
                self.intervention_queue.append(i)
                
        random.shuffle(self.intervention_queue)
        self._advance_intervention_window()

    def _open_block_window(self):
        self.intervention_queue = []
        action = self.state.turn.action
        
        if action == 1:
            for i in range(self.num_players):
                if i != self.state.turn.active_player and self.state.players[i].influence_count > 0:
                    self.intervention_queue.append(i)
        elif action in range(4, 15):
            target = self.state.turn.target
            if self.state.players.get(target, PlayerState(cash=0, influence_count=0)).influence_count > 0:
                self.intervention_queue.append(target)
                
        random.shuffle(self.intervention_queue)
        
        if len(self.intervention_queue) > 0:
            self._advance_intervention_window()
        else:
            self._resolve_successful_action()

    def _advance_intervention_window(self, block_accepted=False):
        if len(self.intervention_queue) > 0:
            next_responder = self.intervention_queue.pop(0)
            self.agent_selection = f"player_{next_responder}"
        else:
            if self.state.turn.phase == Phase.ACTION_CHALLENGE:
                if self.state.turn.action in [1] + list(range(4, 15)):
                    self.state.turn.phase = Phase.ACTION_BLOCK
                    self._open_block_window()
                else:
                    self._resolve_successful_action()
            elif self.state.turn.phase == Phase.ACTION_BLOCK:
                self._resolve_successful_action()
            elif self.state.turn.phase == Phase.BLOCK_RESPONSE:
                if not block_accepted:
                    self._resolve_successful_action()
                else:
                    self._next_turn()

    def _resolve_successful_action(self):
        action = self.state.turn.action
        target = self.state.turn.target
        initiator = self.state.turn.active_player
        p_state = self.state.players[initiator]

        claimed_role = self._get_claimed_role(action)
        if action == 1:
            p_state.cash += 2
            self._next_turn()
        elif action == 2:
            p_state.cash += 3
            self._next_turn()
        elif action == 3:
            active_cards = [
                inf.role for inf in p_state.influence if not inf.revealed]
            drawn_cards = [self.state.deck.pop(), self.state.deck.pop()]
            self.state.turn.exchange_pool = active_cards + drawn_cards
            self.state.turn.exchange_returns_left = 2
            self.state.turn.phase = Phase.EXCHANGE
            self.agent_selection = f"player_{initiator}"
        elif action in range(4, 9):
            t_state = self.state.players.get(target, PlayerState(cash=0, influence_count=0))
            if t_state.influence_count > 0:
                stolen = min(2, t_state.cash)
                t_state.cash -= stolen
                p_state.cash += stolen
            self._next_turn()
        elif action in range(10, 15):
            if self.state.players.get(target, PlayerState(cash=0, influence_count=0)).influence_count > 0:
                self.state.turn.phase = Phase.REVEAL_INFLUENCE
                self.state.turn.player_to_reveal = target
                self.agent_selection = f"player_{target}"
            else:
                self._next_turn()
        elif action in range(16, 21):
            t_state = self.state.players.get(target, PlayerState(cash=0, influence_count=0))
            if t_state.influence_count > 0:
                p_state.cash -= 7
                self.state.turn.phase = Phase.REVEAL_INFLUENCE
                self.state.turn.player_to_reveal = target
                self.agent_selection = f"player_{target}"
            else:
                p_state.cash -= 7
                self._next_turn()

    # ==========================================
    # Challenge & Elimination Handlers
    # ==========================================

    def _resolve_challenge(self, challenger, challenging_block=False):
        if challenging_block:
            challenged = self.state.turn.target
            claimed_role = self.state.turn.blocking_role
        else:
            challenged = self.state.turn.active_player
            claimed_role = self._get_claimed_role(self.state.turn.action)

        challenged_state = self.state.players[challenged]

        has_role = False
        matching_card_idx = -1

        for i, inf in enumerate(challenged_state.influence):
            if not inf.revealed and inf.role == claimed_role:
                has_role = True
                matching_card_idx = i
                break

        if has_role:
            loser = challenger
            # The challenger called a bluff INCORRECTLY.
            
            self.state.deck.append(
                challenged_state.influence[matching_card_idx].role)
            random.shuffle(self.state.deck)
            challenged_state.influence[matching_card_idx].role = self.state.deck.pop(
            )

            if not challenging_block:
                self.state.turn.pending_action = True
        else:
            loser = challenged
            # The challenger called a bluff CORRECTLY.
            self.proven_not_to_have[challenged][claimed_role.value] = 1
            
            # If the bluffed action was an Assassination, refund the 3 coins!
            if not challenging_block and self.state.turn.action in range(10, 15):
                challenged_state.cash += 3
            
            if challenging_block:
                self.state.turn.pending_action = True
                self.state.turn.resuming_from_failed_block = True

        self.state.turn.phase = Phase.REVEAL_INFLUENCE
        self.state.turn.player_to_reveal = loser
        self.agent_selection = f"player_{loser}"

    def _get_claimed_role(self, action):
        if action == 2:
            return Role.DUKE
        if action == 3:
            return Role.AMBASSADOR
        if action in range(4, 9):
            return Role.CAPTAIN
        if action in range(10, 15):
            return Role.ASSASSIN
        return Role.NONE

    def _check_eliminations_and_victory(self):
        alive_count = 0
        winner = None

        for i in range(self.num_players):
            agent = f"player_{i}"
            if self.state.players[i].influence_count == 0:
                self.state.players[i].cash = 0
                if agent in self.agents and not self.terminations[agent]:
                    self.terminations[agent] = True
                    penalty = -1.0 + (self.players_eliminated * self.elimination_step)
                    self.rewards[agent] += penalty
                    self.winner_pot += abs(penalty)
                    self.players_eliminated += 1
            else:
                alive_count += 1
                winner = agent

        if alive_count <= 1:
            if winner in self.agents and not self.terminations[winner]:
                self.rewards[winner] += self.winner_pot

            for agent in self.agents:
                self.terminations[agent] = True
                self.truncations[agent] = False

        for agent in self.agents:
            if self.terminations[agent]:
                if not hasattr(self, "_stored_agent_selection"):
                    self._stored_agent_selection = self.agent_selection
                self.agent_selection = agent
                return

    def _accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.rewards[agent] = 0



    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _record_claim(self, player_idx, role):
        if role != Role.NONE:
            self.active_claims[player_idx][role.value] = 1

    def _record_grudge(self, initiator, target):
        if target != -1 and target is not None:
            self.grudges[target][initiator] = 1