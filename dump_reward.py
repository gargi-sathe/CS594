import pprint
from src.rl.scenario_sampler import ScenarioSampler
from src.rl.warehouse_env import WarehousePlacementEnv

base_scenario = {
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
        "outputs": {
            "output_dir": "results/test"
        }
    }
}

sampler = ScenarioSampler(base_scenario)
env = WarehousePlacementEnv(sampler, max_candidates=25)
env.reset(seed=42)

# Step 1 (Intermediate)
obs, reward1, term1, trunc1, info1 = env.step(0)
print("--- STEP 1 (INTERMEDIATE) ---")
print(f"Reward: {reward1}, Terminated: {term1}")

# Step 2 (Terminal)
obs, reward2, term2, trunc2, info2 = env.step(1)
print("\n--- STEP 2 (TERMINAL) ---")
print(f"Reward: {reward2:.4f}, Terminated: {term2}")
print("\nTERMINAL INFO KEYS:")
for k, v in info2.items():
    if k == "scenario_config":
        print(f"  {k}: [dict omitted]")
    else:
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
