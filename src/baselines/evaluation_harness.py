import copy
from typing import Dict, Any, Callable
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.scenario_sampler import ScenarioSampler
from src.core.engine import run_simulation
from src.metrics.aggregator import calculate_metrics

def run_baseline_evaluation(
    scenario_config: Dict[str, Any], 
    baseline_callable: Callable
) -> Dict[str, Any]:
    """
    Executes a baseline uniquely invoking the true Phase 0 Simulator Core cleanly.
    Locks scenario map limits internally strictly preventing random config derivations.
    """
    sampler = ScenarioSampler(scenario_config)
    env = WarehousePlacementEnv(sampler)
    
    obs, info = env.reset()
    
    candidates = env.candidates
    K = env.K
    G = env.G
    apsp = env.apsp
    config_frozen = env.current_config_parsed
    
    selected_nodes = baseline_callable(candidates, K, G, apsp, config_frozen)
    
    config_rollout = copy.deepcopy(config_frozen)
    config_rollout.entities.warehouse_locations = selected_nodes
    
    final_state = run_simulation(config_rollout)
    metrics = calculate_metrics(final_state, config_rollout)
    
    cost_per_order = metrics.get("simple_cost_estimate", 0.0) / max(metrics.get("total_orders_generated", 1), 1)
    metrics["cost_per_order"] = cost_per_order
    metrics["selected_warehouse_node_ids"] = selected_nodes
    
    return {
        "selected_warehouse_node_ids": metrics["selected_warehouse_node_ids"],
        "on_time_delivery_rate": metrics.get("on_time_delivery_rate", 0.0),
        "delivered_success_rate": metrics.get("delivered_success_rate", 0.0),
        "p90_delivery_time": metrics.get("p90_delivery_time", 0.0),
        "p95_delivery_time": metrics.get("p95_delivery_time", 0.0),
        "orders_missed_or_unserved": metrics.get("orders_missed_or_unserved", 0),
        "simple_cost_estimate": metrics.get("simple_cost_estimate", 0.0),
        "cost_per_order": metrics["cost_per_order"],
        "driver_utilization": metrics.get("driver_utilization", 0.0)
    }
