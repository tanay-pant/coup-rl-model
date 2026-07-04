import math

def calculate_stall_ev(num_players, current_orbit=0, max_moves=200):
    gamma = 0.99
    # EV of challenging
    challenge_ev = -0.5
    
    # Calculate finite sum
    # Assume we are at move 0 of the current state, and the game truncates at max_moves.
    moves_left = max_moves
    orbits_left = moves_left // num_players
    
    discounted_penalty_sum = 0
    for k in range(1, orbits_left + 1):
        actual_orbit = current_orbit + k
        penalty = 0.005 + 0.001 * actual_orbit
        # Assuming the active player acts at step 1 of their orbit (so it's discounted by 1)
        # But wait, if they just finished their turn, the next penalty applies at step `num_players`.
        step_applied = k * num_players
        discount = gamma ** step_applied
        discounted_penalty_sum -= penalty * discount
        
    truncation_discount = gamma ** max_moves
    total_ev = discounted_penalty_sum - (1.0 * truncation_discount)
    
    return discounted_penalty_sum, truncation_discount, total_ev

print("6-player game, starting from orbit 0:")
p_sum, t_disc, total = calculate_stall_ev(6)
print(f"Penalties: {p_sum:.3f}, Truncation: -{t_disc:.3f}, Total EV: {total:.3f}")

print("\n2-player game, starting from orbit 0:")
p_sum, t_disc, total = calculate_stall_ev(2)
print(f"Penalties: {p_sum:.3f}, Truncation: -{t_disc:.3f}, Total EV: {total:.3f}")

print("\n3-player game, starting from orbit 0:")
p_sum, t_disc, total = calculate_stall_ev(3)
print(f"Penalties: {p_sum:.3f}, Truncation: -{t_disc:.3f}, Total EV: {total:.3f}")

print("\n6-player game, if game was infinite length:")
print("Let's calculate the infinite sum for 6 players:")
inf_p_sum = 0
for k in range(1, 1000):
    penalty = 0.005 + 0.001 * k
    inf_p_sum -= penalty * (0.99 ** (k * 6))
print(f"Infinite Penalties: {inf_p_sum:.3f}")

print("\n2-player game, if game was infinite length:")
inf_p_sum_2 = 0
for k in range(1, 1000):
    penalty = 0.005 + 0.001 * k
    inf_p_sum_2 -= penalty * (0.99 ** (k * 2))
print(f"Infinite Penalties: {inf_p_sum_2:.3f}")
