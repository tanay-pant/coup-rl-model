import os
import shutil

checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_pbt', 'coup_pbt_run'))

for trial_dir in os.listdir(checkpoint_dir):
    if trial_dir.startswith("PPO_"):
        trial_path = os.path.join(checkpoint_dir, trial_dir)
        if os.path.isdir(trial_path):
            checkpoints = []
            for cp in os.listdir(trial_path):
                if cp.startswith("checkpoint_"):
                    checkpoints.append(cp)
            
            if not checkpoints:
                continue
                
            # Find the highest number
            highest_idx = -1
            latest_cp = None
            for cp in checkpoints:
                cp_idx = int(cp.split("_")[-1])
                if cp_idx > highest_idx:
                    highest_idx = cp_idx
                    latest_cp = cp
                    
            # Delete everything else
            deleted_count = 0
            for cp in checkpoints:
                if cp != latest_cp:
                    shutil.rmtree(os.path.join(trial_path, cp))
                    deleted_count += 1
                    
            print(f"[{trial_dir}] Deleted {deleted_count} old checkpoints. Kept {latest_cp}")
