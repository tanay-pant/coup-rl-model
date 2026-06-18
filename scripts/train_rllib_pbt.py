import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog
from scripts.train_rllib import setup_rllib_config, CoupActionMaskModel

from ray.tune import CLIReporter

def train_pbt():
    ray.init()
    
    # Must register the custom PyTorch model for Tune's workers
    ModelCatalog.register_custom_model("coup_mask_model", CoupActionMaskModel)
    
    # 2 rollout workers per trial. With 2 trials (populations), this is 4 workers total.
    # This keeps the memory usage low enough for a 16GB Mac.
    config = setup_rllib_config(num_workers=2, use_pbt=True)
    
    # The PBT Scheduler manages hyperparameter evolution dynamically
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="env_runners/episode_reward_mean",
        mode="max",
        perturbation_interval=25, # Evaluate and mutate every 25 iterations
        hyperparam_mutations={
            "lr": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            "entropy_coeff": [0.001, 0.01, 0.05, 0.1, 0.2],
            "vf_loss_coeff": [0.1, 0.5, 1.0],
        }
    )
    
    storage_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_pbt'))
    os.makedirs(storage_path, exist_ok=True)
    
    # Custom reporter to prevent terminal spam
    reporter = CLIReporter(
        metric_columns=["training_iteration", "env_runners/episode_reward_mean"],
        max_progress_rows=2,
        max_report_frequency=30 # Only print updates to the terminal every 30 seconds
    )
    
    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=2, # Keep at 2 to avoid crashing the Mac
        ),
        run_config=tune.RunConfig(
            name="coup_pbt_run",
            storage_path=storage_path,
            stop={"training_iteration": 10000}, # Stop after 10000 iterations
            progress_reporter=reporter,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=100,
                checkpoint_at_end=True
            ),
        ),
        param_space=config.to_dict()
    )
    
    print("Starting Population Based Training (2 Populations)...")
    results = tuner.fit()
    print("PBT Finished!")
    
if __name__ == "__main__":
    train_pbt()
