import os
import sys
import time
import asyncio
import traceback

import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from envs.coup.game_logic import Phase, Action, Role
from agents.neural_mcts import NeuralMCTSBot
from scripts.train_lstm_advanced import CoupActionMaskLSTM
from backend.main import generate_contextual_log

def print_board_state(env, agent_id):
    print("\n" + "="*50)
    print("BOARD STATE:")
    
    # My state
    idx = int(agent_id.split("_")[1])
    my_state = env.state.players[idx]
    my_cards = [inf.role.name for inf in my_state.influence if not inf.revealed and inf.role != Role.NONE]
    print(f"YOU: Coins: {my_state.cash}, Cards: {my_cards}")
    
    # Opponents
    for i, p in env.state.players.items():
        if i != idx:
            name = "Abe" if i == 1 else ("Bart" if i == 2 else f"Player {i}")
            print(f"{name}: Coins: {p.cash}, Cards Remaining: {p.influence_count}")
            
    print("="*50)
    print("YOUR TURN! Choose an action:")
    
    obs = env.observe(agent_id)
    mask = obs["action_mask"]
    legal_moves = [i for i, m in enumerate(mask) if m == 1]
    
    for m in legal_moves:
        action_name = Action(m).name if m < len(Action) else str(m)
        if m in range(4, 10):  # Steal
            target = m - 4
            name = "Abe" if target == 1 else ("Bart" if target == 2 else str(target))
            print(f"[{m}] Steal from {name}")
        elif m in range(10, 16): # Assassinate
            target = m - 10
            name = "Abe" if target == 1 else ("Bart" if target == 2 else str(target))
            print(f"[{m}] Assassinate {name}")
        elif m in range(16, 22): # Coup
            target = m - 16
            name = "Abe" if target == 1 else ("Bart" if target == 2 else str(target))
            print(f"[{m}] Coup {name}")
        else:
            print(f"[{m}] {action_name}")
            
    print("INPUT_REQUIRED")
    sys.stdout.flush()

def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ckpt_v2_path = os.path.join(base_dir, 'checkpoints_lstm_advanced_v2', 'checkpoint_50000')
    policy_dir = os.path.join(ckpt_v2_path, "policies", "main_policy")
    loaded_policy = Policy.from_checkpoint(policy_dir)

    env = CoupEnv()
    env.reset(options={"num_players": 3})
    
    bots = {}
    for agent in env.agents:
        agent_idx = int(agent.split("_")[1])
        if agent_idx != 0:
            bots[agent] = NeuralMCTSBot(agent_id=agent_idx, loaded_policy=loaded_policy, num_simulations=50, max_time=0.5)

    print("Game Started!")
    
    while True:
        active_agents = [a for a in env.agents if not (env.terminations[a] or env.truncations[a])]
        if len(active_agents) <= 1:
            break
            
        for agent in env.agent_iter():
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
                
            active_agents = [a for a in env.agents if not (env.terminations[a] or env.truncations[a])]
            if len(active_agents) <= 1:
                env.step(None)
                continue

            agent_idx = int(agent.split("_")[1])
            
            if agent_idx == 0:
                print_board_state(env, agent)
                input_file = '/Users/tanay-pant/.gemini/antigravity/brain/980a8304-90a9-4f35-9006-6f54443feda1/scratch/llm_input.txt'
                while True:
                    if os.path.exists(input_file):
                        with open(input_file, 'r') as f:
                            content = f.read().strip()
                        if content:
                            try:
                                action = int(content)
                                os.remove(input_file)
                                break
                            except ValueError:
                                pass
                    time.sleep(0.5)
            else:
                bot = bots[agent]
                action = bot.compute_action(env)

            log_msg = generate_contextual_log(env, action, agent_idx)
            print(log_msg)
            sys.stdout.flush()
            env.step(action)

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

    winner = winners[0] if winners else "Draw"
    print(f"\nGame Over! Winner: {winner}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
