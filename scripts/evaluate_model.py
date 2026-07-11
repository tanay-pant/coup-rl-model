import os
import sys
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.coup.coup_env import CoupEnv
from envs.coup.game_logic import Phase, Role

def get_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir): return None
    highest_idx = -1
    ckpt_path = None
    for cp in os.listdir(ckpt_dir):
        if cp.startswith("checkpoint_"):
            try:
                idx = int(cp.split("_")[1])
                if idx > highest_idx:
                    highest_idx = idx
                    ckpt_path = os.path.join(ckpt_dir, cp)
            except ValueError:
                pass
    return ckpt_path

def evaluate():
    ray.init(ignore_reinit_error=True)
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_lstm_advanced_v2'))
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    
    if not latest_ckpt:
        print("No checkpoint found in checkpoints_lstm_advanced_v2.")
        return
        
    print(f"Loading checkpoint {latest_ckpt}...")
    from scripts.train_lstm_advanced import setup_rllib_config, env_creator
    config = setup_rllib_config(start_iter=0)
    algo = config.build_algo()
    algo.restore(latest_ckpt)
    
    env = env_creator({})
    
    num_games = 500
    
    metrics = {
        "blind_foreign_aid": 0,
        "submissive_stealing": 0,
        "impossible_bluffs": 0,
        "endgame_blunders": 0,
        "passive_high_wealth": 0,
        "suboptimal_targeting": 0,
        "stubborn_assassination": 0,
        "total_actions": 0,
        "total_games": 0
    }
    
    for game in range(num_games):
        obs_dict, info_dict = env.reset()
        
        last_blocked_action = {}
        lstm_states = {}
        known_contessas = {}
            
        while True:
            action_dict = {}
            for agent, agent_obs in obs_dict.items():
                if agent not in lstm_states:
                    lstm_states[agent] = algo.get_policy("main_policy").get_initial_state()
                if agent not in last_blocked_action:
                    last_blocked_action[agent] = None
                    
                action, state_out, _ = algo.compute_single_action(
                    agent_obs, 
                    state=lstm_states[agent],
                    policy_id="main_policy", 
                    explore=False
                )
                lstm_states[agent] = state_out
                
                action_mask = agent_obs["action_mask"]
                if action_mask[action] == 0:
                    valid_actions = np.where(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                
                action_dict[agent] = action
                metrics["total_actions"] += 1
                
                agent_idx = int(agent.split("_")[1])
                phase = env.par_env.unwrapped.state.turn.phase if hasattr(env, 'par_env') else env.env.unwrapped.state.turn.phase
                turn_action = env.par_env.unwrapped.state.turn.action if hasattr(env, 'par_env') else env.env.unwrapped.state.turn.action
                
                if phase == Phase.START_OF_TURN:
                    if action == 1:
                        if last_blocked_action[agent] == 'fa':
                            metrics["blind_foreign_aid"] += 1
                        last_blocked_action[agent] = 'fa_attempt'
                    elif action in range(4, 9):
                        if last_blocked_action[agent] == 'steal':
                            metrics["submissive_stealing"] += 1
                        last_blocked_action[agent] = 'steal_attempt'
                    else:
                        last_blocked_action[agent] = None
                        
                elif phase == Phase.ACTION_BLOCK:
                    if action == 23: # Pass
                        active_p = env.par_env.unwrapped.state.turn.active_player if hasattr(env, 'par_env') else env.env.unwrapped.state.turn.active_player
                        if agent_idx == active_p:
                            if turn_action == 1:
                                last_blocked_action[agent] = 'fa'
                            elif turn_action in range(4, 9):
                                last_blocked_action[agent] = 'steal'
                    elif action == 22: # Challenge
                        last_blocked_action[agent] = None
                    elif action == 27: # Contessa block
                        known_contessas[agent_idx] = True
                        
                elif phase == Phase.BLOCK_RESPONSE:
                    if action == 22: # Challenge
                        target_p = env.par_env.unwrapped.state.turn.target if hasattr(env, 'par_env') else env.env.unwrapped.state.turn.target
                        known_contessas[target_p] = False
                        
                state = env.par_env.unwrapped.state if hasattr(env, 'par_env') else env.env.unwrapped.state
                
                if phase == Phase.START_OF_TURN:
                    p_state = state.players[agent_idx]
                    if p_state.cash >= 7 and action == 0:
                        metrics["passive_high_wealth"] += 1
                        
                    if action in range(10, 15) or action in range(16, 21):
                        target_offset = (action - 10 + 1) if action in range(10, 15) else (action - 16 + 1)
                        target_idx = (agent_idx + target_offset) % state.num_players
                        target_state = state.players.get(target_idx)
                        if target_state and target_state.influence_count == 1 and target_state.cash < 3:
                            for i, other_state in state.players.items():
                                if i != agent_idx and i != target_idx and other_state.influence_count == 2 and other_state.cash >= 7:
                                    metrics["suboptimal_targeting"] += 1
                                    break
                                    
                    if action in range(10, 15):
                        target_offset = action - 10 + 1
                        target_idx = (agent_idx + target_offset) % state.num_players
                        if known_contessas.get(target_idx, False):
                            metrics["stubborn_assassination"] += 1
                            
                    if action == 3: # Exchange
                        known_contessas[agent_idx] = False
                        
                global_dead = agent_obs["observation"][-5:]
                claimed_role = None
                if action == 2 or action == 24: claimed_role = 0
                elif action == 3 or action == 28: claimed_role = 3
                elif action in range(4, 9) or action == 25: claimed_role = 2
                elif action in range(10, 15): claimed_role = 1
                elif action == 27: claimed_role = 4
                
                if claimed_role is not None:
                    if global_dead[claimed_role] == 3:
                        metrics["impossible_bluffs"] += 1
                        
                alive_count = sum([1 for p in state.players.values() if p.influence_count > 0])
                if alive_count == 2:
                    if phase == Phase.ACTION_BLOCK and turn_action in range(10, 15) and action == 23:
                        if state.turn.target == agent_idx:
                            if state.players[agent_idx].influence_count == 1:
                                metrics["endgame_blunders"] += 1
            
            obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                break
            
        metrics["total_games"] += 1
        if (game + 1) % 100 == 0:
            print(f"Evaluated {game + 1} games...")

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    print("\nReport Analysis:")
    print("Blind Foreign Aid:", "High" if metrics["blind_foreign_aid"] > 50 else "Low")
    print("Submissive Stealing:", "High" if metrics["submissive_stealing"] > 50 else "Low")
    print("Impossible Bluffs:", "High" if metrics["impossible_bluffs"] > 10 else "Low")
    print("Endgame Blunders:", "High" if metrics["endgame_blunders"] > 20 else "Low")
    print("Passive High Wealth:", "High" if metrics["passive_high_wealth"] > 20 else "Low")
    print("Suboptimal Targeting:", "High" if metrics["suboptimal_targeting"] > 20 else "Low")
    print("Stubborn Assassination:", "High" if metrics["stubborn_assassination"] > 20 else "Low")

if __name__ == "__main__":
    evaluate()
