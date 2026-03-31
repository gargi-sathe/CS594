import pprint
from src.rl.scenario_sampler import ScenarioSampler
from src.rl.warehouse_env import WarehousePlacementEnv

base_config_raw = {
    "run_config": {
        "seed": 42,
        "mode": "synthetic",
        "run_horizon_mins": 120,
        "map": {
            "type": "grid",
            "grid_size": [5, 5]
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
            "output_dir": "results/test"
        }
    }
}

sampler = ScenarioSampler(base_config_raw)
env = WarehousePlacementEnv(sampler, max_candidates=25)
obs, info = env.reset(seed=42)
print("--- OBSERVATION DUMP ---")
print("GLOBAL FEATURES [13]:")
for i, v in enumerate(obs["global_features"]):
    print(f"  [{i}]: {v:.4f}")
    
print("\nCANDIDATE 0 FEATURES [8]:")
for i, v in enumerate(obs["candidate_features"][0]):
    print(f"  [{i}]: {v:.4f}")
    
print("\nSELECTED MASK HEAD (5):")
pprint.pprint(obs["selected_mask"][:5].tolist())
print("\nACTION MASK HEAD (5):")
pprint.pprint(obs["action_mask"][:5].tolist())
