from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import IntEnum


class Phase(IntEnum):
    WAITING_FOR_PLAYERS = 0
    START_OF_TURN = 1
    ACTION_RESPONSE = 2
    BLOCK_RESPONSE = 3
    REVEAL_INFLUENCE = 4
    EXCHANGE = 5


class Role(IntEnum):
    NONE = -1
    DUKE = 0
    ASSASSIN = 1
    CAPTAIN = 2
    AMBASSADOR = 3
    CONTESSA = 4


class Action(IntEnum):
    NONE = -1
    INCOME = 0
    FOREIGN_AID = 1
    TAX = 2
    EXCHANGE = 3
    STEAL = 4
    ASSASSINATE = 5
    COUP = 6


@dataclass
class Influence:
    role: Role = Role.NONE
    revealed: bool = False


@dataclass
class PlayerState:
    cash: int = 2
    influence_count: int = 2
    influence: List[Influence] = field(
        default_factory=lambda: [
            Influence(), Influence()])


@dataclass
class TurnState:
    phase: Phase = Phase.WAITING_FOR_PLAYERS
    active_player: int = -1
    action: Action = Action.NONE
    target: int = -1
    blocking_role: Role = Role.NONE
    player_to_reveal: int = -1

    # handle the Ambassador exchange phase
    exchange_pool: List[Role] = field(default_factory=lambda: [Role.NONE] * 4)
    exchange_returns_left: int = 0


@dataclass
class GameState:
    """
    The master state object representing the absolute truth of the game board.
    """
    num_players: int
    players: Dict[int, PlayerState] = field(default_factory=dict)
    deck: List[Role] = field(default_factory=list)
    turn: TurnState = field(default_factory=TurnState)

    def setup_base_game(self):
        """Helper to mimic the start() function from game.js"""
        roles = [
            Role.DUKE,
            Role.CAPTAIN,
            Role.ASSASSIN,
            Role.CONTESSA,
            Role.AMBASSADOR]

        self.deck = roles * 3

        for i in range(self.num_players):
            self.players[i] = PlayerState()
