import sys
sys.path.append('.')
from backend.main import generate_contextual_log
class DummyRole:
    name = "Duke"
class DummyPhase:
    name = "ACTION_CHALLENGE"
class DummyTurn:
    phase = DummyPhase()
    active_player = 1
    action = 2
class DummyState:
    turn = DummyTurn()
class DummyEnv:
    state = DummyState()
    
env = DummyEnv()
print(generate_contextual_log(env, 23, 2))
