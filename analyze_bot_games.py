import re

def analyze_log():
    with open('watch_bots_output.txt', 'r') as f:
        content = f.read()
        
    games = content.split('STARTING GAME')
    
    challenge_count = 0
    allow_count = 0
    bluffs_caught = 0
    
    for game in games[1:]:
        lines = game.split('\n')
        for i, line in enumerate(lines):
            if "chose: Challenge" in line:
                challenge_count += 1
                print(f"CHALLENGE AT LINE: {line}")
                # Print context
                for j in range(max(0, i-10), min(len(lines), i+10)):
                    print(lines[j])
                print("-" * 50)
            elif "chose: Allow" in line:
                allow_count += 1
                
    print(f"Total Challenges: {challenge_count}")
    print(f"Total Allows: {allow_count}")

if __name__ == '__main__':
    analyze_log()
