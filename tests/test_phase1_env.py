import pytest
import numpy as np
from src.rl.scenario_sampler import ScenarioSampler
from src.rl.warehouse_env import WarehousePlacementEnv

@pytest.fixture
def base_config_raw():
    return {
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

def test_observation_schema_shape_and_bounds(base_config_raw):
    # Tests requirements: schema shape test & normalization bounds test
    sampler = ScenarioSampler(base_config_raw)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    obs, info = env.reset(seed=42)
    
    assert obs["global_features"].shape == (13,)
    assert obs["candidate_features"].shape == (25, 8)
    assert obs["selected_mask"].shape == (25,)
    assert obs["action_mask"].shape == (25,)
    assert obs["step_index"].shape == (1,)
    
    # bounds
    assert np.all((obs["global_features"] >= 0.0) & (obs["global_features"] <= 1.0))
    assert np.all((obs["candidate_features"] >= 0.0) & (obs["candidate_features"] <= 1.0))

def test_deterministic_reset_and_sampler(base_config_raw):
    # Tests requirements: deterministic reset test under fixed seed & synthetic scenario sampler determinism
    sampler = ScenarioSampler(base_config_raw)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    
    np.testing.assert_allclose(obs1["global_features"], obs2["global_features"])
    np.testing.assert_allclose(obs1["candidate_features"], obs2["candidate_features"])

def test_selected_action_mask_correctness(base_config_raw):
    sampler = ScenarioSampler(base_config_raw)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    obs, info = env.reset(seed=42)
    
    # Init state
    assert obs["selected_mask"].sum() == 0
    assert obs["action_mask"].sum() == 25
    assert np.all(obs["candidate_features"][:, 7] == 0) # No selected flags yet
    
    env.step(0)
    assert env.obs["selected_mask"][0] == 1
    assert env.obs["action_mask"][0] == 0
    assert env.obs["candidate_features"][0, 7] == 1.0
    
    # Check that mask overlaps are completely mutually exclusive for valid nodes
    for i in range(25):
        if env.obs["selected_mask"][i] == 1:
            assert env.obs["action_mask"][i] == 0

def test_candidate_set_generation(base_config_raw):
    sampler = ScenarioSampler(base_config_raw)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    obs, info = env.reset(seed=42)
    
    # 5x5 grid means 25 valid nodes generated exactly
    assert len(info["candidate_node_ids"]) == 25
    
    # Change config to 3x3 to test varying grid candidate sizes
    cfg = base_config_raw.copy()
    cfg["run_config"]["map"]["grid_size"] = [3, 3]
    sampler2 = ScenarioSampler(cfg)
    env2 = WarehousePlacementEnv(sampler2, max_candidates=25)
    obs2, info2 = env2.reset(seed=42)
    
    assert len(info2["candidate_node_ids"]) == 9
    assert obs2["action_mask"].sum() == 9 # Only 9 active actions mask allowed

def test_explicit_feature_schema_keys(base_config_raw):
    sampler = ScenarioSampler(base_config_raw)
    env = WarehousePlacementEnv(sampler, max_candidates=25)
    expected_keys = [
        "global_features",
        "candidate_features",
        "selected_mask",
        "action_mask",
        "step_index"
    ]
    assert set(env.observation_space.keys()) == set(expected_keys)
    obs, _ = env.reset(seed=42)
    assert set(obs.keys()) == set(expected_keys)
