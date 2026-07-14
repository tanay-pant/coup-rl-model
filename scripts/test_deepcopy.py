import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
import copy
import random

env = CoupEnv()
env.reset(options={"num_players": 4})

for i in range(50):
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
            
        action_mask = obs["action_mask"]
        valid = [i for i, m in enumerate(action_mask) if m == 1]
        action = random.choice(valid)
        
        sim_env = copy.deepcopy(env)
        agent_sel = sim_env.agent_selection
        try:
            _ = sim_env.terminations[agent_sel]
        except KeyError as e:
            print(f"FAILED ON DEEPCOPY! Agent: {agent_sel}")
            print(f"sim_env.terminations: {sim_env.terminations}")
            print(f"env.terminations: {env.terminations}")
            sys.exit(1)
            
        env.step(action)
