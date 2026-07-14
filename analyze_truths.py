import sys

def analyze():
    failed_challenges = 0
    total_challenges = 0
    with open("20k_eval_log.txt", "r") as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        line = lines[i].strip()
        if "main_policy) chose: Challenge" in line:
            total_challenges += 1
            # Check the next line to see if someone revealed an influence
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if "REVEAL_INFLUENCE" in next_line and "main_policy" not in next_line:
                    # An opponent was forced to reveal. Did they have the card?
                    # The env forces a reveal. If they had it, they reveal it. 
                    # If they didn't have it, they reveal their card (and lose it).
                    # Actually, if the opponent revealed the card they claimed, the challenger (main_policy) dies!
                    # But the log just says "Reveal [Role]". We can't immediately tell if they lied or not just from the reveal string unless we look at the action they were challenged on.
                    # But let's check if main_policy dies right after!
                    pass
            # Better check: how often does main_policy challenge and then LOSE an influence?
            # If main_policy loses an influence, it means it challenged wrong, OR it got challenged.
            
    print(f"Total challenges by main: {total_challenges}")

analyze()
