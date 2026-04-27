import numpy as np
import yaml
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.scenario_sampler import ScenarioSampler
from sb3_contrib import MaskablePPO

def run_preflight():
    print(">>> Starting Chicago Real-Map Preflight Check...")
    
    config_dict = {
        "run_config": {
            "seed": 42,
            "mode": "synthetic",
            "run_horizon_mins": 120.0,
            "map": {
                "type": "osmnx",
                "osmnx_place": "Loop, Chicago, Illinois, USA",
                "candidate_generation_mode": "random_subsample",
                "candidate_subsample_size": 25
            },
            "entities": {
                "num_warehouses": 3,
                "num_drivers": 5
            },
            "parameters": {
                "delivery_target_mins": 15.0,
                "pick_pack_time_mins": 2.0,
                "order_lambda": 30.0
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
            "outputs": {
                "output_dir": "results/preflight"
            }
        }
    }
    
    sampler = ScenarioSampler(config_dict)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    
    # 1. Shape Check
    obs, info = env.reset(seed=42)
    print(f"Observation global_features shape: {obs['global_features'].shape} (Expected: (13,))")
    print(f"Observation candidate_features shape: {obs['candidate_features'].shape} (Expected: (25, 8))")
    print(f"Action mask length: {len(env.action_masks())} (Expected: 25)")
    
    assert obs['global_features'].shape == (13,)
    assert obs['candidate_features'].shape == (25, 8)
    assert len(env.action_masks()) == 25
    
    # 2. Reset Determinism Check
    candidates_1 = info["candidate_node_ids"]
    obs_2, info_2 = env.reset(seed=42)
    candidates_2 = info_2["candidate_node_ids"]
    
    if candidates_1 == candidates_2:
        print("Candidate determinism: PASSED")
    else:
        print("Candidate determinism: FAILED")
        print(f"C1: {candidates_1[:3]}...")
        print(f"C2: {candidates_2[:3]}...")
        
    # 3. Model Compatibility Check
    model_path = "logs/checkpoints/pilot_C_full/pilot_C2_full_run/final_model.zip"
    try:
        model = MaskablePPO.load(model_path, env=env)
        action, _ = model.predict(obs, action_masks=env.action_masks())
        print(f"Model predict dry-run: PASSED (Action selected: {action})")
    except Exception as e:
        print(f"Model compatibility check: FAILED - {e}")

    print("\n>>> Preflight Complete!")

if __name__ == "__main__":
    run_preflight()
