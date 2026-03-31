import pprint
from src.rl.scenario_sampler import ScenarioSampler
from src.rl.warehouse_env import WarehousePlacementEnv
from src.baselines.warehouse_random import random_baseline
from src.baselines.warehouse_demand_clustering import demand_clustering_baseline
from src.baselines.warehouse_coverage_greedy import coverage_greedy_baseline
from src.baselines.evaluation_harness import run_baseline_evaluation

base_scenario = {
    "run_config": {
        "seed": 42,
        "mode": "synthetic",
        "run_horizon_mins": 30, # short smoke
        "map": {
            "type": "grid",
            "grid_size": [3, 3] # small 9 nodes
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
        "outputs": {
            "output_dir": "results/test"
        }
    }
}

print("=== M2 TINY SCENARIO BASELINE COMPARISON ===")
def print_metrics(name, m):
    print(f"\n[{name}]")
    print(f"Selected Nodes : {m['selected_warehouse_node_ids']}")
    print(f"On-Time Dlv %  : {m['on_time_delivery_rate']:.3f}")
    print(f"Dlv Success %  : {m['delivered_success_rate']:.3f}")
    print(f"P90 Time (m)   : {m['p90_delivery_time']:.1f}")
    print(f"P95 Time (m)   : {m['p95_delivery_time']:.1f}")
    print(f"Missed Orders  : {m['orders_missed_or_unserved']}")
    print(f"Est. Total Cost: ${m['simple_cost_estimate']:.2f}")
    print(f"Cost/Order     : ${m['cost_per_order']:.2f}")
    print(f"Driver Util %  : {m['driver_utilization']:.3f}")

# 1. Random
m_rand = run_baseline_evaluation(base_scenario, random_baseline)
print_metrics("Random Baseline", m_rand)

# 2. Demand Clustering
m_clust = run_baseline_evaluation(base_scenario, demand_clustering_baseline)
print_metrics("Demand Clustering Baseline", m_clust)

# 3. Coverage Greedy
m_greedy = run_baseline_evaluation(base_scenario, coverage_greedy_baseline)
print_metrics("Coverage Greedy Baseline", m_greedy)
