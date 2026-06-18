import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coup.coup_env import CoupEnv

def run_tests():
    env = CoupEnv()
    num_episodes = 10000
    
    print(f"Running {num_episodes} random games to test CoupEnv for crashes...")
    
    for ep in range(num_episodes):
        env.reset()
        
        steps = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action_mask = obs["action_mask"]
                valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                if not valid_actions:
                    print(f"Error: No valid actions for {agent} in phase {env.state.turn.phase}!")
                    sys.exit(1)
                
                action = random.choice(valid_actions)
                
            env.step(action)
            steps += 1
            
            if steps > 5000:
                print(f"Error: Game {ep} stuck in infinite loop!")
                sys.exit(1)
                
        if (ep + 1) % 2500 == 0:
            print(f"Completed {ep + 1} games without errors.")
            
    print("SUCCESS: 10,000 random games completed with zero faults.")

if __name__ == "__main__":
    run_tests()
