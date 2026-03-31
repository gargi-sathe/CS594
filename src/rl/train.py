import argparse
import yaml
import os
from src.config.schema import parse_config
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.scenario_sampler import ScenarioSampler

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

def run_training(config_path: str, override_seed: int = None):
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
        
    if override_seed is not None:
        raw_config.setdefault("run_config", {})["training"] = raw_config.get("run_config", {}).get("training", {})
        raw_config["run_config"]["training"]["seed"] = override_seed
        raw_config["run_config"]["seed"] = override_seed
        
    config = parse_config(raw_config)
    
    log_dir = config.training.log_dir
    os.makedirs(log_dir, exist_ok=True)
    snapshot_path = os.path.join(log_dir, "config_snapshot.yaml")
    with open(snapshot_path, 'w') as f:
        yaml.dump(raw_config, f)
        
    sampler = ScenarioSampler(raw_config)
    if config.map.type == "grid":
        grid_w, grid_h = config.map.grid_size
        max_cands = grid_w * grid_h
    else:
        max_cands = 100
        
    env = WarehousePlacementEnv(sampler, max_candidates=max_cands)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.checkpoint_freq,
        save_path=config.training.checkpoint_dir,
        name_prefix="rl_model"
    )
    
    model = MaskablePPO(
        config.training.policy_name,
        env,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        tensorboard_log=config.training.log_dir,
        seed=config.training.seed,
        device=config.training.device,
        verbose=1
    )
    
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=config.training.tensorboard_log_name
    )
    
    model.save(os.path.join(config.training.checkpoint_dir, "final_model"))
    
    summary = f"""# Training Summary
Total Timesteps: {config.training.total_timesteps}
Checkpoints Dir: {config.training.checkpoint_dir}
TensorBoard Dir: {log_dir}
Seed: {config.training.seed}
Learning Rate: {config.training.learning_rate}
N-Steps: {config.training.n_steps}
"""
    with open(os.path.join(log_dir, "training_summary.md"), "w") as f:
        f.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()
    
    run_training(args.config, override_seed=args.seed)
