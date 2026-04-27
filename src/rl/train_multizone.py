import argparse
import yaml
import os
from src.config.schema import parse_config
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.multi_zone_sampler import MultiZoneScenarioSampler

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

def run_multizone_training(config_path: str, override_seed: int = None, load_path: str = None, run_id: str = None):
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
        
    if override_seed is not None:
        raw_config.setdefault("run_config", {})["training"] = raw_config.get("run_config", {}).get("training", {})
        raw_config["run_config"]["training"]["seed"] = override_seed
        raw_config["run_config"]["seed"] = override_seed
        
    config = parse_config(raw_config)
    
    # Extract Multi-Zone specific config
    # Expecting a list in raw_config['run_config']['training']['multi_zone_list']
    mz_list = raw_config.get("run_config", {}).get("training", {}).get("multi_zone_list", [])
    if not mz_list:
        raise ValueError("multi_zone_list is empty in config!")
        
    # Run ID organization
    log_dir = config.training.log_dir
    checkpoint_dir = config.training.checkpoint_dir
    if run_id:
        log_dir = os.path.join(log_dir, run_id)
        checkpoint_dir = os.path.join(checkpoint_dir, run_id)
        
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    sampler = MultiZoneScenarioSampler(raw_config, mz_list)
    
    # Fixed at 25 for this phase
    max_cands = 25
        
    print(f"Initializing Multi-Zone environment with max_candidates={max_cands}")
    env = WarehousePlacementEnv(sampler, max_candidates=max_cands)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="rl_model"
    )
    
    if load_path and os.path.exists(load_path):
        print(f"Resuming training from {load_path}")
        model = MaskablePPO.load(load_path, env=env, device=config.training.device)
        reset_num_timesteps = False
    else:
        print("Starting training from scratch")
        model = MaskablePPO(
            config.training.policy_name,
            env,
            learning_rate=config.training.learning_rate,
            n_steps=config.training.n_steps,
            batch_size=config.training.batch_size,
            tensorboard_log=log_dir,
            seed=config.training.seed,
            device=config.training.device,
            verbose=1
        )
        reset_num_timesteps = True
    
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=config.training.tensorboard_log_name,
        reset_num_timesteps=reset_num_timesteps
    )
    
    model.save(os.path.join(checkpoint_dir, "final_model"))
    print(f"Training complete. Final model saved to {checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--load_path", type=str, default=None, help="Path to existing model to resume")
    parser.add_argument("--run_id", type=str, default=None, help="Unique ID for this run")
    args = parser.parse_args()
    
    run_multizone_training(args.config, override_seed=args.seed, load_path=args.load_path, run_id=args.run_id)
