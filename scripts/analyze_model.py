import os
import sys
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from envs.coup.game_logic import Phase, Role
from scripts.train_lstm_advanced import setup_rllib_config, env_creator

def action_to_str(action, target):
    if action == 0: return "Income"
    if action == 1: return "Foreign Aid"
    if action == 2: return "Tax"
    if action == 3: return "Exchange"
    if action in range(4, 9): return f"Steal from player_{target}"
    if action in range(10, 15): return f"Assassinate player_{target}"
    if action in range(16, 21): return f"Coup player_{target}"
    if action == 22: return "Challenge"
    if action == 23: return "Allow/Pass"
    if action == 24: return "Block with Duke"
    if action == 25: return "Block with Captain"
    if action == 27: return "Block with Contessa"
    if action == 28: return "Block with Ambassador"
    if action == 29: return "Reveal Duke"
    if action == 30: return "Reveal Assassin"
    if action == 31: return "Reveal Captain"
    if action == 32: return "Reveal Ambassador"
    if action == 33: return "Reveal Contessa"
    if action >= 34: return f"Return card {action-34} in exchange pool"
    return f"Unknown Action {action}"

def run_evaluation():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("iteration", type=int, help="Checkpoint iteration number to evaluate")
    args = parser.parse_args()
    iteration = args.iteration

    ray.init(ignore_reinit_error=True)
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_lstm_advanced_v2'))
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    print(f"Loading checkpoint {ckpt_path}...")
    config = setup_rllib_config(start_iter=0)
    algo = config.build_algo()
    algo.restore(ckpt_path)
    
    env = env_creator({})
    
    num_games = 200
    log_file = open(f"{iteration//1000}k_eval_log.txt", "w")
    
    policy_mapping = {
        "player_0": "main_policy",
        "player_1": "main_policy",
        "player_2": "rational_policy",
        "player_3": "liar_policy",
        "player_4": "honest_policy",
        "player_5": "random_policy"
    }

    metrics = {
        "main_wins": 0,
        "main_bluffs_duke": 0,
        "main_steals": 0,
        "main_assassinations": 0,
        "main_exchanges": 0,
        "main_challenges": 0,
        "main_blocks": 0
    }
    
    for game in range(num_games):
        obs_dict, info_dict = env.reset(options={"num_players": 6})
        
        lstm_states = {}
        log_file.write(f"\n{'='*40}\nGAME {game + 1}\n{'='*40}\n")
        
        # Get real game state
        real_state = env.par_env.unwrapped.state if hasattr(env, 'par_env') else env.env.unwrapped.state
        for i in range(6):
            p = real_state.players[i]
            roles = [inf.role.name for inf in p.influence]
            log_file.write(f"player_{i} ({policy_mapping[f'player_{i}']}) starts with: {roles}\n")
        
        while True:
            action_dict = {}
            for agent, agent_obs in obs_dict.items():
                policy_id = policy_mapping[agent]
                
                if policy_id == "main_policy":
                    if agent not in lstm_states:
                        lstm_states[agent] = algo.get_policy("main_policy").get_initial_state()
                        
                    action, state_out, _ = algo.compute_single_action(
                        agent_obs, 
                        state=lstm_states[agent],
                        policy_id="main_policy", 
                        explore=False
                    )
                    lstm_states[agent] = state_out
                else:
                    action_out = algo.compute_single_action(
                        agent_obs,
                        policy_id=policy_id,
                        explore=True
                    )
                    # Heuristic policies might just return the action directly instead of a tuple
                    if isinstance(action_out, tuple):
                        action = action_out[0]
                    else:
                        action = action_out
                
                action_mask = agent_obs["action_mask"]
                if action_mask[action] == 0:
                    valid_actions = np.where(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                
                action_dict[agent] = action
                
                agent_idx = int(agent.split("_")[1])
                phase = real_state.turn.phase
                
                target = -1
                if action in range(4, 9): target = (agent_idx + action - 4 + 1) % 6
                if action in range(10, 15): target = (agent_idx + action - 10 + 1) % 6
                if action in range(16, 21): target = (agent_idx + action - 16 + 1) % 6
                
                action_str = action_to_str(action, target)
                
                if action != 23: # Don't log every pass/allow
                    log_file.write(f"[{phase.name}] {agent} ({policy_id}) chose: {action_str}\n")
                
                if policy_id == "main_policy":
                    if action == 2:
                        # Check if it was a bluff
                        p = real_state.players[agent_idx]
                        has_duke = any(inf.role == Role.DUKE and not inf.revealed for inf in p.influence)
                        if not has_duke:
                            metrics["main_bluffs_duke"] += 1
                    if action in range(4, 9): metrics["main_steals"] += 1
                    if action in range(10, 15): metrics["main_assassinations"] += 1
                    if action == 3: metrics["main_exchanges"] += 1
                    if action == 22: metrics["main_challenges"] += 1
                    if action in [24, 25, 27, 28]: metrics["main_blocks"] += 1
            
            obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                winner = None
                for i in range(6):
                    if real_state.players[i].influence_count > 0:
                        winner = f"player_{i}"
                if winner and policy_mapping[winner] == "main_policy":
                    metrics["main_wins"] += 1
                log_file.write(f"GAME OVER. Winner: {winner}\n")
                break
                
    log_file.close()
    
    print("--- EVALUATION COMPLETE ---")
    print(f"Total Games: {num_games}")
    print(f"Main Policy Wins: {metrics['main_wins']} ({(metrics['main_wins']/num_games)*100:.1f}%)")
    print("Main Policy Behavior Stats:")
    for k, v in metrics.items():
        if k != "main_wins":
            print(f"  {k}: {v}")

if __name__ == "__main__":
    run_evaluation()
