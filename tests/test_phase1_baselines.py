import pytest
from src.baselines.warehouse_random import random_baseline
from src.baselines.warehouse_demand_clustering import demand_clustering_baseline
from src.baselines.warehouse_coverage_greedy import coverage_greedy_baseline
from src.baselines.evaluation_harness import run_baseline_evaluation
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.scenario_sampler import ScenarioSampler

@pytest.fixture
def base_scenario():
    return {
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
            "outputs": {
                "output_dir": "results/test"
            }
        }
    }

def test_k_size_and_candidate_validity_random(base_scenario):
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    obs, info = env.reset(seed=42)
    selected = random_baseline(env.candidates, env.K, env.G, env.apsp, env.current_config_parsed)
    assert len(selected) == env.K
    assert len(set(selected)) == env.K
    assert all(s in env.candidates for s in selected)
    
    # Determinism
    selected2 = random_baseline(env.candidates, env.K, env.G, env.apsp, env.current_config_parsed)
    assert selected == selected2

def test_k_size_and_candidate_validity_clustering(base_scenario):
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    obs, info = env.reset(seed=42)
    selected = demand_clustering_baseline(env.candidates, env.K, env.G, env.apsp, env.current_config_parsed)
    assert len(selected) == env.K
    assert len(set(selected)) == env.K
    assert all(s in env.candidates for s in selected)

def test_k_size_and_candidate_validity_greedy(base_scenario):
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    obs, info = env.reset(seed=42)
    selected = coverage_greedy_baseline(env.candidates, env.K, env.G, env.apsp, env.current_config_parsed)
    assert len(selected) == env.K
    assert len(set(selected)) == env.K
    assert all(s in env.candidates for s in selected)

def test_evaluation_harness_output_schema(base_scenario):
    schema = run_baseline_evaluation(base_scenario, random_baseline)
    expected_keys = {
        "selected_warehouse_node_ids",
        "on_time_delivery_rate",
        "delivered_success_rate",
        "p90_delivery_time",
        "p95_delivery_time",
        "orders_missed_or_unserved",
        "simple_cost_estimate",
        "cost_per_order",
        "driver_utilization"
    }
    assert set(schema.keys()) == expected_keys
