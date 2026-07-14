import os
import sys
import copy
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from envs.coup.game_logic import Phase
from agents.neural_mcts import NeuralMCTSBot

import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from scripts.train_lstm_advanced import CoupActionMaskLSTM

def get_action_name(idx, agent_idx):
    if idx == 0: return "Income"
    if idx == 1: return "Foreign Aid"
    if idx == 2: return "Tax"
    if idx == 3: return "Exchange"
    
    MAX_PLAYERS = 6
    if 4 <= idx <= 9: return f"Steal from player_{(agent_idx + (idx - 4 + 1)) % MAX_PLAYERS}"
    if 10 <= idx <= 15: return f"Assassinate player_{(agent_idx + (idx - 10 + 1)) % MAX_PLAYERS}"
    if 16 <= idx <= 21: return f"Coup player_{(agent_idx + (idx - 16 + 1)) % MAX_PLAYERS}"
    if idx == 22: return "Challenge"
    if idx == 23: return "Allow/Pass"
    
    roles = {0: "Duke", 1: "Assassin", 2: "Captain", 3: "Ambassador", 4: "Contessa"}
    if idx == 24: return "Block with Duke"
    if idx == 25: return "Block with Captain"
    if idx == 27: return "Block with Contessa"
    if idx == 28: return "Block with Ambassador"
    if 29 <= idx <= 33: return f"Reveal {roles[idx - 29]}"
    if 34 <= idx <= 38: return f"Exchange return pool index {idx - 34}"
    
    return f"Unknown Action {idx}"

def print_game_state(env):
    state = env.state
    print("\n" + "="*50)
    print(f"PHASE: {state.turn.phase.name} | ACTIVE PLAYER: player_{state.turn.active_player}")
    print("="*50)
    for i in range(state.num_players):
        p = state.players[i]
        cards = []
        for inf in p.influence:
            if inf.revealed:
                cards.append(f"[{inf.role.name} (DEAD)]")
            else:
                cards.append(f"[{inf.role.name}]")
        print(f"player_{i: <2} | Coins: {p.cash: <2} | Cards: {' '.join(cards)}")
    print("="*50)

def main():
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ckpt_v2_path = os.path.join(base_dir, 'checkpoints_lstm_advanced_v2', 'checkpoint_50000')
    policy_dir = os.path.join(ckpt_v2_path, "policies", "main_policy")
    
    print("Loading base policy...")
    loaded_policy = Policy.from_checkpoint(policy_dir)
    
    env = CoupEnv()
    env.reset(options={"num_players": 4})
    
    bots = {}
    import torch
    global_lstm_states = {}
    
    for agent in env.agents:
        agent_idx = int(agent.split("_")[1])
        bots[agent] = NeuralMCTSBot(agent_id=agent_idx, loaded_policy=loaded_policy, num_simulations=400, max_time=0.4)
        global_lstm_states[agent] = [torch.tensor([s], dtype=torch.float32) for s in loaded_policy.get_initial_state()]
        
    print_game_state(env)
    
    move_count = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
            
        bot = bots[agent]
        agent_idx = int(agent.split('_')[1])
        
        # We only print the state when it is the start of a turn to avoid spamming the log
        if env.state.turn.phase == Phase.START_OF_TURN and env.state.turn.active_player == agent_idx:
            print_game_state(env)
            
        action = bot.compute_action(env, current_lstm_states=global_lstm_states)
        
        action_name = get_action_name(action, agent_idx)
        print(f">>> {agent} chose: {action_name}")
        
        env.step(action)
        move_count += 1
        
        if move_count > 200:
            print("Game reached 200 moves, stopping to prevent infinite loops.")
            break
            
    print("\nGAME OVER!")
    for i in range(env.MAX_PLAYERS):
        if env.state.players[i].influence_count > 0:
            print(f"🏆 WINNER: player_{i} 🏆")
            break
            
    ray.shutdown()

if __name__ == "__main__":
    main()
