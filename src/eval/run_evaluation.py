import os
import pandas as pd
import yaml
import copy
from dataclasses import dataclass
from typing import Dict, Any, List

from sb3_contrib import MaskablePPO

from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.scenario_sampler import ScenarioSampler
from src.config.schema import parse_config, RunConfig
from src.core.engine import run_simulation
from src.metrics.aggregator import calculate_metrics

from src.baselines.warehouse_random import random_baseline
from src.baselines.warehouse_demand_clustering import demand_clustering_baseline
from src.baselines.warehouse_coverage_greedy import coverage_greedy_baseline

@dataclass
class FrozenScenario:
    scenario_id: str
    bucket: str
    seed: int
    candidate_node_ids: List[Any]
    demand_weights: Dict[Any, float]
    K: int
    D: int
    T: float
    map_reference: Any
    config: RunConfig
    G: Any
    apsp: Any

def evaluate_method_on_frozen(scenario: FrozenScenario, method_name: str, method_callable=None, model=None) -> Dict[str, Any]:
    # We create a pristine rollout config
    rollout_config = copy.deepcopy(scenario.config)
    
    if method_name in ["Random", "Clustering", "Greedy"]:
        selected = method_callable(
            scenario.candidate_node_ids, 
            scenario.K, 
            scenario.G, 
            scenario.apsp, 
            rollout_config
        )
    elif method_name == "MaskablePPO":
        # Need to reconstruct environment exactly to trace the step logic
        from dataclasses import asdict
        cfg_dict = asdict(scenario.config) if hasattr(scenario.config, "__dict__") else {"seed": scenario.seed}
        sampler = ScenarioSampler({"run_config": cfg_dict})
        env = WarehousePlacementEnv(sampler, max_candidates=len(scenario.candidate_node_ids))
        
        # Override env internals with frozen ones just in case
        env.G = scenario.G
        env.apsp = scenario.apsp
        env.candidates = scenario.candidate_node_ids
        env.current_config_parsed = copy.deepcopy(scenario.config)
        env.K = scenario.K
        
        obs, _ = env.reset(seed=scenario.seed)
        terminated = False
        selected = []
        
        while not terminated:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
        selected = info.get("selected_warehouse_node_ids", [])
        return {
            "scenario_id": scenario.scenario_id,
            "bucket": scenario.bucket,
            "method": method_name,
            "selected_warehouse_node_ids": selected,
            "on_time_delivery_rate": info.get("on_time_delivery_rate", 0.0),
            "delivered_success_rate": info.get("delivered_success_rate", 0.0),
            "p90_delivery_time": info.get("p90_delivery_time", 0.0),
            "p95_delivery_time": info.get("p95_delivery_time", 0.0),
            "orders_missed_or_unserved": info.get("orders_missed_or_unserved", 0),
            "simple_cost_estimate": info.get("simple_cost_estimate", 0.0),
            "cost_per_order": info.get("cost_per_order", 0.0),
            "driver_utilization": info.get("driver_utilization", 0.0)
        }
        
    # Baseline manual calculation
    if rollout_config.entities.warehouse_locations is None:
        rollout_config.entities.warehouse_locations = []
    rollout_config.entities.warehouse_locations = selected
    
    final_state = run_simulation(rollout_config)
    metrics = calculate_metrics(final_state, rollout_config)
    
    raw_total_orders = metrics.get("total_orders_generated", 0)
    denom = max(raw_total_orders, 1)
    
    cost_per_order = metrics.get("simple_cost_estimate", 0.0) / denom
    missed = metrics.get("orders_missed_or_unserved", 0)
    
    return {
        "scenario_id": scenario.scenario_id,
        "bucket": scenario.bucket,
        "method": method_name,
        "selected_warehouse_node_ids": selected,
        "on_time_delivery_rate": metrics.get("on_time_delivery_rate", 0.0),
        "delivered_success_rate": metrics.get("delivered_success_rate", 0.0),
        "p90_delivery_time": metrics.get("p90_delivery_time", 0.0),
        "p95_delivery_time": metrics.get("p95_delivery_time", 0.0),
        "orders_missed_or_unserved": missed,
        "simple_cost_estimate": metrics.get("simple_cost_estimate", 0.0),
        "cost_per_order": cost_per_order,
        "driver_utilization": metrics.get("driver_utilization", 0.0)
    }

def generate_frozen_scenarios(base_config_raw: Dict[str, Any], buckets: Dict[str, float], n_per: int) -> List[FrozenScenario]:
    frozen = []
    for b_name, lambd in buckets.items():
        for i in range(n_per):
            cfg_raw = copy.deepcopy(base_config_raw)
            # Offset the seed locally for unique scenarios
            cfg_seed = cfg_raw.get("run_config", {}).get("seed", 42) + i * 10
            cfg_raw["run_config"]["seed"] = cfg_seed
            cfg_raw["run_config"]["parameters"]["order_lambda"] = lambd
            
            cfg_parsed = parse_config(cfg_raw)
            sampler = ScenarioSampler(cfg_raw)
            env = WarehousePlacementEnv(sampler)
            env.reset(seed=cfg_seed)
            
            demand_field = {n: env.G.nodes[n].get("demand_weight", 1.0) for n in env.G.nodes()}
            
            f = FrozenScenario(
                scenario_id=f"{b_name}_{cfg_seed}",
                bucket=b_name,
                seed=cfg_seed,
                candidate_node_ids=env.candidates,
                demand_weights=demand_field,
                K=env.K,
                D=cfg_parsed.entities.num_drivers,
                T=cfg_parsed.parameters.delivery_target_mins,
                map_reference=cfg_parsed.map,
                config=cfg_parsed,
                G=env.G,
                apsp=env.apsp
            )
            frozen.append(f)
    return frozen

def run_evaluation(config_path: str):
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        
    cfg = parse_config(raw_config)
    os.makedirs(cfg.eval.output_dir, exist_ok=True)
    
    model = None
    if os.path.exists(cfg.eval.model_path):
        model = MaskablePPO.load(cfg.eval.model_path)
    else:
        print(f"Warning: RL Model not found at {cfg.eval.model_path}. RL evaluation will be skipped.")
        
    scenarios = generate_frozen_scenarios(raw_config, cfg.eval.buckets, cfg.eval.scenarios_per_bucket)
    
    all_results = []
    
    for s in scenarios:
        methods = {
            "Random": random_baseline,
            "Clustering": demand_clustering_baseline,
            "Greedy": coverage_greedy_baseline
        }
        
        for name, func in methods.items():
            res = evaluate_method_on_frozen(s, name, method_callable=func)
            all_results.append(res)
            
        if model is not None:
            res_rl = evaluate_method_on_frozen(s, "MaskablePPO", model=model)
            all_results.append(res_rl)
            
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(cfg.eval.output_dir, "eval_raw_results.csv"), index=False)
    
    summary = df.groupby(["bucket", "method"])[
        ["on_time_delivery_rate", "delivered_success_rate", "p90_delivery_time", 
         "p95_delivery_time", "orders_missed_or_unserved", "cost_per_order", "driver_utilization"]
    ].mean().reset_index()
    
    summary.to_csv(os.path.join(cfg.eval.output_dir, "eval_summary_table.csv"), index=False)
    
    with open(os.path.join(cfg.eval.output_dir, "eval_summary_table.md"), "w") as f:
        f.write("# Evaluation Summary Table\n")
        f.write(summary.to_markdown(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_evaluation(args.config)
