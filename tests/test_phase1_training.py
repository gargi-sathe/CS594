import pytest
import os
import shutil
import numpy as np
from src.rl.train import run_training
from stable_baselines3.common.env_checker import check_env
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.scenario_sampler import ScenarioSampler

@pytest.fixture
def smoke_config():
    log_dir = "logs/test_tb/"
    ckpt_dir = "logs/test_checkpoints/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        
    cfg = {
        "run_config": {
            "seed": 42,
            "mode": "synthetic",
            "run_horizon_mins": 30,
            "map": {
                "type": "grid",
                "grid_size": [3, 3]
            },
            "entities": {
                "num_warehouses": 2,
                "num_drivers": 3
            },
            "parameters": {
                "delivery_target_mins": 10.0,
                "pick_pack_time_mins": 2.0,
                "order_lambda": 15.0
            },
            "policies": {
                "staging": "baseline",
                "dispatch": "baseline"
            },
            "costs": {
                "warehouse_base": 100.0,
                "driver_hourly": 20.0,
                "order_op_cost": 2.5
            },
            "reward_weights": {
                "weight_on_time": 2.0,
                "weight_delivered_success": 0.5,
                "weight_missed_rate": 1.0,
                "weight_tail_penalty": 0.5,
                "weight_cost": 0.1,
                "cost_norm_ref": 100.0
            },
            "training": {
                "total_timesteps": 64,  
                "checkpoint_freq": 64,
                "log_dir": log_dir,
                "checkpoint_dir": ckpt_dir,
                "learning_rate": 0.0003,
                "n_steps": 16,
                "batch_size": 4,
                "policy_name": "MultiInputPolicy",
                "device": "cpu",
                "tensorboard_log_name": "TestSmoke",
                "seed": 42
            },
            "outputs": {
                "output_dir": "results/test"
            }
        }
    }
    import yaml
    os.makedirs("configs", exist_ok=True)
    with open("configs/test_smoke.yaml", "w") as f:
        yaml.dump(cfg, f)
    return cfg

def test_env_sb3_compatibility(smoke_config):
    sampler = ScenarioSampler(smoke_config)
    env = WarehousePlacementEnv(sampler, max_candidates=9)
    try:
        check_env(env, warn=True, skip_render_check=True)
    except ValueError as e:
        if "Action mask is 0" not in str(e):
            raise e

def test_smoke_training_loop_and_artifacts(smoke_config):
    run_training("configs/test_smoke.yaml", override_seed=42)
    
    tb_dir = smoke_config["run_config"]["training"]["log_dir"]
    ckpt_dir = smoke_config["run_config"]["training"]["checkpoint_dir"]
    
    assert os.path.exists(os.path.join(tb_dir, "config_snapshot.yaml"))
    assert os.path.exists(os.path.join(tb_dir, "training_summary.md"))
    assert os.path.exists(os.path.join(ckpt_dir, "final_model.zip"))
    
    subfolders = [f for f in os.listdir(tb_dir) if f.startswith("TestSmoke")]
    assert len(subfolders) > 0

def test_action_prediction_reproducibility(smoke_config):
    from sb3_contrib import MaskablePPO
    
    sampler1 = ScenarioSampler(smoke_config)
    env1 = WarehousePlacementEnv(sampler1, max_candidates=9)
    env1.reset(seed=42)
    
    model1 = MaskablePPO("MultiInputPolicy", env1, seed=42)
    action1, _ = model1.predict(env1.obs, action_masks=env1.action_masks(), deterministic=True)
    
    sampler2 = ScenarioSampler(smoke_config)
    env2 = WarehousePlacementEnv(sampler2, max_candidates=9)
    env2.reset(seed=42)
    
    model2 = MaskablePPO("MultiInputPolicy", env2, seed=42)
    action2, _ = model2.predict(env2.obs, action_masks=env2.action_masks(), deterministic=True)
    
    assert action1 == action2
