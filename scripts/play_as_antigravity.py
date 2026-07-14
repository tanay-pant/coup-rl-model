import os
import sys
import time
import random
import traceback

import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from envs.coup.game_logic import Phase
from agents.neural_mcts import NeuralMCTSBot
from scripts.train_lstm_advanced import CoupActionMaskLSTM

def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ckpt_v2_path = os.path.join(base_dir, 'checkpoints_lstm_advanced_v2', 'checkpoint_50000')
    policy_dir = os.path.join(ckpt_v2_path, "policies", "main_policy")
    loaded_policy = Policy.from_checkpoint(policy_dir)

    print("--- ANTIGRAVITY BOT VS NEURAL MCTS BOTS ---")
    
    num_games = 10
    results = []

    for game_idx in range(num_games):
        # Vary player counts: 3, 4, 5, 6
        num_players = random.choice([3, 4, 5, 6])
        env = CoupEnv()
        env.reset(options={"num_players": num_players})
        
        # Me: AntigravityBot (Player 0) - Huge compute advantage + heuristics
        # Them: Standard bots - 50 sims, 0.5s limit
        bots = {}
        for agent in env.agents:
            agent_idx = int(agent.split("_")[1])
            if agent_idx == 0:
                # Antigravity uses 200 sims and 1.0 seconds
                bots[agent] = NeuralMCTSBot(agent_id=0, loaded_policy=loaded_policy, num_simulations=200, max_time=1.0)
            else:
                bots[agent] = NeuralMCTSBot(agent_id=agent_idx, loaded_policy=loaded_policy, num_simulations=20, max_time=0.2)

        print(f"\nStarting Game {game_idx + 1} with {num_players} players...")
        
        turn_count = 0
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

                bot = bots[agent]
                try:
                    action = bot.compute_action(env)
                except Exception as e:
                    print(f"Bot {agent} crashed! {e}")
                    obs = env.observe(agent)
                    legal_moves = [i for i, m in enumerate(obs["action_mask"]) if m == 1]
                    action = random.choice(legal_moves)

                env.step(action)
                turn_count += 1
                
                # Failsafe for infinite loops
                if turn_count > 500:
                    break
                    
            if turn_count > 500:
                print("Game aborted due to infinite loop.")
                break

        # Determine winner
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
        results.append(winner)
        print(f"Game {game_idx + 1} Winner: {winner} (Turns: {turn_count})")

    print("\n--- RESULTS ---")
    ag_wins = results.count("player_0")
    print(f"AntigravityBot (Me) Wins: {ag_wins} / {num_games}")
    for i in range(1, 6):
        bot_wins = results.count(f"player_{i}")
        if bot_wins > 0:
            print(f"Bot {i} Wins: {bot_wins}")

if __name__ == "__main__":
    main()
