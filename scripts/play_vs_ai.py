import os
import sys

# Ensure Python can find our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

from scripts.train_lstm import CoupActionMaskLSTM
from envs.coup.coup_env import CoupEnv

def get_action_name(idx):
    if idx == 0: return "Income"
    if idx == 1: return "Foreign Aid"
    if idx == 2: return "Tax"
    if idx == 3: return "Exchange"
    if 4 <= idx <= 9: return f"Steal from Player {idx - 4}"
    if 10 <= idx <= 15: return f"Assassinate Player {idx - 10}"
    if 16 <= idx <= 21: return f"Coup Player {idx - 16}"
    if idx == 22: return "Challenge"
    if idx == 23: return "Allow"
    
    roles = {0: "Duke", 1: "Assassin", 2: "Captain", 3: "Ambassador", 4: "Contessa"}
    if idx == 24: return "Block with Duke"
    if idx == 25: return "Block with Captain"
    if idx == 27: return "Block with Contessa"
    if idx == 28: return "Block with Ambassador"
    if 29 <= idx <= 33: return f"Reveal {roles[idx - 29]}"
    if 34 <= idx <= 38: return f"Exchange return pool index {idx - 34}"
    
    return f"Unknown Action {idx}"

def print_board(env):
    state = env.state
    print("\n" + "="*45)
    print(f"   --- COUP TERMINAL INTERFACE ---")
    print(f"Phase: {state.turn.phase.name} | Active Player: {state.turn.active_player}")
    print("="*45)
    for i in range(state.num_players):
        p = state.players[i]
        cards = []
        for inf in p.influence:
            if inf.revealed:
                cards.append(f"[{inf.role.name} (DEAD)]")
            elif i == 0:
                cards.append(f"[{inf.role.name}]")
            else:
                cards.append("[HIDDEN]")
        prefix = "YOU" if i == 0 else f"AI {i}"
        print(f"{prefix:6} | {p.cash} Coins | Cards: {' '.join(cards)}")
        
    if state.turn.phase.name == "EXCHANGE" and state.turn.active_player == 0:
        pool_cards = [f"[{i}: {r.name}]" for i, r in enumerate(state.turn.exchange_pool) if r.value != -1]
        print("-" * 45)
        print(f"EXCHANGE POOL: {' '.join(pool_cards)}")
        
    print("="*45 + "\n")

def main():
    ray.init()
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)

    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_lstm'))
    if not os.path.exists(checkpoint_dir):
        print(f"No valid checkpoints found at {checkpoint_dir}. Please wait for train_lstm.py to save one!")
        sys.exit(1)

    # Find the most recent checkpoint 
    load_dir = None
    highest_idx = -1
    for cp in os.listdir(checkpoint_dir):
        if cp.startswith("checkpoint_"):
            try:
                cp_idx = int(cp.split("_")[1])
                if cp_idx > highest_idx:
                    highest_idx = cp_idx
                    load_dir = os.path.join(checkpoint_dir, cp)
            except ValueError:
                pass
                            
    if not load_dir:
        print("No valid checkpoints found.")
        sys.exit(1)
        
    print(f"Loading AI Brain from: {load_dir}")
    
    algo = Algorithm.from_checkpoint(load_dir)
    
    env = CoupEnv()
    env.reset()
    
    state_map = {agent: algo.get_policy("main_policy").get_initial_state() for agent in env.agents if agent != "player_0"}
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
        
        action_mask = observation["action_mask"]
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        
        if agent == "player_0":
            print_board(env)
            print(f"--- IT IS YOUR TURN ---")
            
            for v in valid_actions:
                print(f"[{v:2d}] {get_action_name(v)}")
            
            action = None
            while action not in valid_actions:
                try:
                    choice = input("Enter Action Number: ")
                    action = int(choice)
                    if action not in valid_actions:
                        print("Invalid choice. Please select a valid number from the menu.")
                except ValueError:
                    print("Please enter a valid number.")
            
            env.step(action)
        else:
            # Pass observation to AI
            action, state_out, info = algo.compute_single_action(
                observation=observation,
                state=state_map[agent],
                policy_id="main_policy" # AI uses the master policy
            )
            state_map[agent] = state_out
            print(f">>> Player {agent} chose: {get_action_name(action)}")
            env.step(action)
            
    print("\nGAME OVER!")
    for i in range(env.MAX_PLAYERS):
        if env.state.players[i].influence_count > 0:
            winner = "YOU" if i == 0 else f"AI {i}"
            print(f"🏆 WINNER: {winner} 🏆")
            break

if __name__ == "__main__":
    main()
