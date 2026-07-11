import re

def parse_hand(hand_str):
    roles = []
    matches = re.findall(r'\[([A-Z]+)', hand_str)
    for m in matches:
        roles.append(m)
    return roles

def analyze_truth():
    with open('watch_bots_output.txt', 'r') as f:
        lines = f.readlines()
        
    truths = 0
    lies = 0
    
    current_hands = {}
    
    # Map actions to required roles
    action_to_role = {
        "Tax": "DUKE",
        "Steal": "CAPTAIN",
        "Assassinate": "ASSASSIN",
        "Exchange": "AMBASSADOR",
        "Block with Duke": "DUKE",
        "Block with Captain": "CAPTAIN",
        "Block with Ambassador": "AMBASSADOR",
        "Block with Contessa": "CONTESSA"
    }
    
    for i, line in enumerate(lines):
        # Update hands
        match = re.search(r'AI\s+(\d+) \| \d+ Coins \| Cards: (.*)', line)
        if match:
            agent_id = int(match.group(1))
            hand_str = match.group(2)
            current_hands[agent_id] = parse_hand(hand_str)
            
        # Check claims
        match_action = re.search(r'>>> AI (\d+) chose: (.*)', line)
        if match_action:
            agent_id = int(match_action.group(1))
            action = match_action.group(2)
            
            claimed_role = None
            for key, role in action_to_role.items():
                if action.startswith(key):
                    claimed_role = role
                    break
                    
            if claimed_role:
                hand = current_hands.get(agent_id, [])
                if claimed_role in hand:
                    truths += 1
                else:
                    lies += 1
                    
    print(f"Total Claims: {truths + lies}")
    print(f"Truths: {truths} ({(truths/(truths+lies))*100:.1f}%)")
    print(f"Lies: {lies} ({(lies/(truths+lies))*100:.1f}%)")

if __name__ == '__main__':
    analyze_truth()
