import pytest
import numpy as np
from src.rl.scenario_sampler import ScenarioSampler
from src.rl.warehouse_env import WarehousePlacementEnv

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

def test_intermediate_reward_zero(base_scenario):
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert terminated is False
    assert "total_orders_generated" not in info

def test_terminal_info_keys_schema(base_scenario):
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    env.reset(seed=42)
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(1)
    
    assert terminated is True
    expected_keys = {
        "selected_warehouse_node_ids",
        "total_orders_generated",
        "delivered_count",
        "delivered_success_rate",
        "on_time_delivery_rate",
        "p50_delivery_time",
        "p90_delivery_time",
        "p95_delivery_time",
        "average_delivery_time",
        "average_queue_time",
        "average_pick_pack_time",
        "average_travel_to_customer_time",
        "driver_utilization",
        "orders_missed_or_unserved",
        "simple_cost_estimate",
        "cost_per_order",
        "missed_rate",
        "scenario_type",
        "scenario_config"
    }
    assert set(info.keys()) == expected_keys

def test_terminal_reward_formula_parity(base_scenario):
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    env.reset(seed=42)
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(1)
    
    raw_total = info["total_orders_generated"]
    denom = max(raw_total, 1)
    
    missed_rate = info["orders_missed_or_unserved"] / denom
    cost_per_order = info["simple_cost_estimate"] / denom
    normalized_cost = min(cost_per_order / 100.0, 5.0)
    
    T = max(10.0, 1e-6)
    p95 = info["p95_delivery_time"]
    tail_pen = max(0.0, (p95 - T) / T)
    
    expected_reward = (
        (2.0 * info["on_time_delivery_rate"]) +
        (0.5 * info["delivered_success_rate"]) -
        (1.0 * missed_rate) -
        (0.5 * tail_pen) -
        (0.1 * normalized_cost)
    )
    
    assert np.isclose(reward, expected_reward)

def test_zero_order_bounds_protection(base_scenario):
    # Set lambda to 0 to force 0 orders
    base_scenario["run_config"]["parameters"]["order_lambda"] = 0.0
    sampler = ScenarioSampler(base_scenario)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    env.reset(seed=42)
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(1)
    
    assert info["total_orders_generated"] == 0
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)
