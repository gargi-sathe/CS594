import pytest
import os
import shutil
import pandas as pd
from src.eval.run_evaluation import generate_frozen_scenarios, evaluate_method_on_frozen, run_evaluation
from src.config.schema import parse_config

@pytest.fixture
def eval_config():
    out_dir = "results/test_eval"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
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
            "training": {},
            "eval": {
                "scenarios_per_bucket": 1,
                "buckets": {"Low": 15.0},
                "output_dir": out_dir,
                "model_path": "DOES_NOT_EXIST.zip"
            },
            "outputs": {
                "output_dir": "results/test"
            }
        }
    }
    
    import yaml
    os.makedirs("configs", exist_ok=True)
    with open("configs/test_eval.yaml", "w") as f:
        yaml.dump(cfg, f)
    return cfg

def test_eval_frozen_scenario_equality(eval_config):
    # Verify scenarios trace identical bounds natively and don't leak randomness
    scenarios1 = generate_frozen_scenarios(eval_config, {"Low": 15.0}, 1)
    scenarios2 = generate_frozen_scenarios(eval_config, {"Low": 15.0}, 1)
    
    s1 = scenarios1[0]
    s2 = scenarios2[0]
    
    assert s1.scenario_id == s2.scenario_id
    assert s1.seed == s2.seed
    assert len(s1.candidate_node_ids) == len(s2.candidate_node_ids)
    assert s1.candidate_node_ids == s2.candidate_node_ids
    assert s1.demand_weights == s2.demand_weights
    assert s1.K == s2.K
    assert s1.D == s2.D
    assert s1.T == s2.T
    
def test_eval_output_match_schema(eval_config):
    scenarios = generate_frozen_scenarios(eval_config, {"Low": 15.0}, 1)
    from src.baselines.warehouse_random import random_baseline
    
    res = evaluate_method_on_frozen(scenarios[0], "Random", method_callable=random_baseline)
    
    expected_keys = {
        "scenario_id", "bucket", "method", "selected_warehouse_node_ids",
        "on_time_delivery_rate", "delivered_success_rate", "p90_delivery_time",
        "p95_delivery_time", "orders_missed_or_unserved", "simple_cost_estimate",
        "cost_per_order", "driver_utilization"
    }
    assert set(res.keys()) == expected_keys

def test_tiny_harness_end_to_end(eval_config):
    run_evaluation("configs/test_eval.yaml")
    out_dir = eval_config["run_config"]["eval"]["output_dir"]
    raw_csv = os.path.join(out_dir, "eval_raw_results.csv")
    sum_csv = os.path.join(out_dir, "eval_summary_table.csv")
    
    assert os.path.exists(raw_csv)
    assert os.path.exists(sum_csv)
    
    df_raw = pd.read_csv(raw_csv)
    # 3 baselines * 1 scenario = 3 rows (because model isn't loaded)
    assert len(df_raw) == 3
    assert "method" in df_raw.columns

