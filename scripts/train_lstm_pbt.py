import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog
from scripts.train_lstm import setup_rllib_config, CoupActionMaskLSTM

from ray.tune import CLIReporter

def train_pbt():
    ray.init()
    
    ModelCatalog.register_custom_model("coup_mask_lstm", CoupActionMaskLSTM)
    
    config = setup_rllib_config(num_workers=6, use_pbt=True)
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="env_runners/episode_reward_mean",
        mode="max",
        perturbation_interval=25, 
        hyperparam_mutations={
            "lr": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            "entropy_coeff": [0.001, 0.01, 0.05, 0.1, 0.2],
            "vf_loss_coeff": [0.1, 0.5, 1.0],
        }
    )
    
    storage_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_lstm_pbt'))
    os.makedirs(storage_path, exist_ok=True)
    
    reporter = CLIReporter(
        metric_columns=["training_iteration", "env_runners/episode_reward_mean"],
        max_progress_rows=2,
        max_report_frequency=30 
    )
    
    experiment_path = os.path.join(storage_path, "coup_lstm_pbt_run")
    if tune.Tuner.can_restore(experiment_path):
        print(f"Resuming LSTM PBT from {experiment_path}...")
        tuner = tune.Tuner.restore(experiment_path, trainable="PPO", resume_unfinished=True, resume_errored=True)
    else:
        tuner = tune.Tuner(
            "PPO",
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=2, 
            ),
            run_config=tune.RunConfig(
                name="coup_lstm_pbt_run",
                storage_path=storage_path,
                stop={"training_iteration": 4000}, 
                progress_reporter=reporter,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1000,
                    checkpoint_at_end=True,
                    num_to_keep=5
                ),
            ),
            param_space=config.to_dict()
        )
    
    print("Starting LSTM Population Based Training (2 Populations)...")
    results = tuner.fit()
    print("PBT Finished!")
    
if __name__ == "__main__":
    train_pbt()
