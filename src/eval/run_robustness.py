import os
import yaml
import random
import pandas as pd
import json
import copy
from sb3_contrib import MaskablePPO

from src.config.schema import parse_config
from src.core.engine import bootstrap_simulation, dispatch_loop
from src.core.state import SimulatorState
from src.entities.models import Order, OrderState, WarehouseState
from src.metrics.aggregator import calculate_metrics
from src.rl.warehouse_env import WarehousePlacementEnv

# Reuse evaluation logic where possible
from src.eval.run_evaluation import evaluate_method_on_frozen, FrozenScenario
from src.baselines.warehouse_random import random_baseline
from src.baselines.warehouse_demand_clustering import demand_clustering_baseline
from src.baselines.warehouse_coverage_greedy import coverage_greedy_baseline

# --- Stress Generators ---

def robust_order_generator(state: SimulatorState, config: dict, stress_type: str, settings: dict):
    """
    Injected order generator that handles Spikes and Time-Shifts.
    """
    nodes = list(state.graph.nodes)
    grid_size = config['run_config']['map'].get('grid_size')
    horizon = config['run_config']['run_horizon_mins']
    
    order_id_counter = 0
    
    while state.env.now < horizon:
        # 1. Determine current lambda based on stress_type
        current_lambda = config['run_config']['parameters']['order_lambda']
        
        if stress_type == "time_shift":
            # Profile: list of [start, end, lambda]
            for start, end, lmbda in settings['lambda_profile']:
                if start <= state.env.now < end:
                    current_lambda = lmbda
                    break
        
        lambda_per_min = current_lambda / 60.0
        if lambda_per_min <= 0:
            yield state.env.timeout(0.5)
            continue

        inter_arrival = random.expovariate(lambda_per_min)
        yield state.env.timeout(inter_arrival)
        
        if state.env.now >= horizon:
            break
            
    # 2. Handle Spatial Spike
        loc = random.choice(nodes)
        if stress_type == "demand_spike":
            multiplier = settings['multiplier']
            quad = settings['quadrant']
            
            in_quad = False
            if config['run_config']['map']['type'] == "grid":
                width = grid_size[0]
                x = loc // width
                y = loc % width
                mid_x, mid_y = grid_size[0] / 2, grid_size[1] / 2
            else:
                # Real map: use node attributes x/y
                node_data = state.graph.nodes[loc]
                x = node_data.get('x', 0)
                y = node_data.get('y', 0)
                # For Chicago Loop, we'll estimate midpoints based on graph extent
                xs = [d.get('x', 0) for n, d in state.graph.nodes(data=True)]
                ys = [d.get('y', 0) for n, d in state.graph.nodes(data=True)]
                mid_x, mid_y = (min(xs) + max(xs))/2, (min(ys) + max(ys))/2

            if quad == 0 and x < mid_x and y < mid_y: in_quad = True
            elif quad == 1 and x >= mid_x and y < mid_y: in_quad = True
            elif quad == 2 and x < mid_x and y >= mid_y: in_quad = True
            elif quad == 3 and x >= mid_x and y >= mid_y: in_quad = True
            
            if in_quad:
                # To simulate a spike of multiplier X, we can either:
                # 1. Generate X orders instead of 1
                # 2. Reject non-quadrant orders (but that's not a spike, that's a shift)
                # Let's generate additional orders
                for _ in range(int(multiplier)):
                    order_id_counter += 1
                    oid = f"O-{order_id_counter}"
                    o = Order(id=oid, location=loc, arrival_time=state.env.now, state=OrderState.UNASSIGNED)
                    state.orders[oid] = o
                continue # Skip the default one
        
        order_id_counter += 1
        oid = f"O-{order_id_counter}"
        o = Order(id=oid, location=loc, arrival_time=state.env.now, state=OrderState.UNASSIGNED)
        state.orders[oid] = o

def closure_process(state: SimulatorState, settings: dict):
    """Fails a warehouse mid-run."""
    yield state.env.timeout(settings['close_at_min'])
    target = settings['target_warehouse_id']
    if target in state.warehouses:
        state.warehouses[target].state = WarehouseState.CLOSED
        state.log_event("Warehouse", target, "CLOSED_FOR_ROBUSTNESS", f"time={state.env.now}")

# --- Core Robustness Engine ---

def run_stress_test(config_dict, stress_type, settings, model=None, method_callable=None, seed=42):
    """
    Runs a single simulation with injected stress.
    """
    raw_config = copy.deepcopy(config_dict)
    
    # Apply Driver Shortage before bootstrap
    if stress_type == "driver_shortage":
        original = raw_config['run_config']['entities']['num_drivers']
        reduced = max(1, int(original * (1 - settings['reduction_fraction'])))
        raw_config['run_config']['entities']['num_drivers'] = reduced
    
    cfg = parse_config(raw_config)
    cfg.seed = seed
    random.seed(seed)
    
    # Simple mock sampler for the environment
    class MockSampler:
        def __init__(self, config_raw): self.config_raw = config_raw
        def sample(self): return self.config_raw
    
    mock_sampler = MockSampler(raw_config)
    
    # Bootstrap
    state = bootstrap_simulation(cfg)
    
    # If RL, apply placement
    if model:
        env_for_mask = WarehousePlacementEnv(mock_sampler, max_candidates=25)
        obs, _ = env_for_mask.reset(seed=seed)
        mask = env_for_mask.action_masks()
        action_indices = []
        # Multi-step placement loop
        for _ in range(cfg.entities.num_warehouses):
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, _, _, info = env_for_mask.step(action)
            mask = env_for_mask.action_masks()
        
        # Apply placement to state
        selected_nodes = info["selected_warehouse_node_ids"]
        for i, node_id in enumerate(selected_nodes):
            w_id = list(state.warehouses.keys())[i]
            state.warehouses[w_id].location = node_id
    elif method_callable:
        # Heuristic placement
        # Baselines expect (k, G, apsp, config)
        action = method_callable(
            k=cfg.entities.num_warehouses,
            candidates=list(state.graph.nodes),
            G=state.graph,
            apsp=state.apsp,
            config=cfg
        )
        for i, node_id in enumerate(action):
            if i < len(state.warehouses):
                w_id = list(state.warehouses.keys())[i]
                state.warehouses[w_id].location = node_id
            
    # Inject Custom Processes
    state.env.process(robust_order_generator(state, raw_config, stress_type, settings))
    state.env.process(dispatch_loop(state, cfg))
    
    if stress_type == "warehouse_closure":
        state.env.process(closure_process(state, settings))
        
    # Run
    state.env.run(until=cfg.run_horizon_mins)
    
    # Calculate Metrics
    metrics = calculate_metrics(state, cfg)
    
    wh_locations = [w.location for w in state.warehouses.values()]
    
    return {
        "stress_type": stress_type,
        "seed": seed,
        "wh_locations": str(wh_locations),
        "on_time_delivery_rate": metrics["on_time_delivery_rate"],
        "delivered_success_rate": metrics["delivered_success_rate"],
        "p95_delivery_time": metrics["p95_delivery_time"],
        "orders_missed_or_unserved": metrics["orders_missed_or_unserved"],
        "cost_per_order": metrics["simple_cost_estimate"] / max(1, metrics["delivered_count"]),
        "driver_utilization": metrics["driver_utilization"]
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_robustness.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--osmnx_place", type=str, default=None)
    parser.add_argument("--num_warehouses", type=int, default=None)
    parser.add_argument("--num_drivers", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        full_cfg = yaml.safe_load(f)
    
    if args.osmnx_place:
        full_cfg['run_config']['map']['osmnx_place'] = args.osmnx_place
    if args.num_warehouses:
        full_cfg['run_config']['entities']['num_warehouses'] = args.num_warehouses
    if args.num_drivers:
        full_cfg['run_config']['entities']['num_drivers'] = args.num_drivers
    
    rob_cfg = full_cfg['run_config']['robustness']
    
    # Overrides
    model_path = args.model_path if args.model_path else rob_cfg['model_path']
    output_dir = args.output_dir if args.output_dir else rob_cfg['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load RL Model
    print(f"Loading model from: {model_path}")
    model = MaskablePPO.load(model_path)
    
    methods = {
        "MaskablePPO": (model, None),
        "Clustering": (None, demand_clustering_baseline),
        "Greedy": (None, coverage_greedy_baseline),
        "Random": (None, random_baseline)
    }
    
    all_raw_results = []
    
    stress_settings = rob_cfg['stress_settings']
    n_seeds = rob_cfg['scenarios_per_type']
    
    for s_type, settings in stress_settings.items():
        print(f"\n>>> Running robustness test: {s_type}")
        for seed in range(42, 42 + n_seeds):
            for m_name, (m_obj, m_func) in methods.items():
                print(f"  Testing {m_name} (seed {seed})...")
                res = run_stress_test(full_cfg, s_type, settings, model=m_obj, method_callable=m_func, seed=seed)
                res["method"] = m_name
                all_raw_results.append(res)
                
    # Save Raw Results
    df = pd.DataFrame(all_raw_results)
    df.to_csv(os.path.join(output_dir, "robustness_raw_results.csv"), index=False)
    
    # Aggregated Summary
    summary = df.groupby(["stress_type", "method"]).mean(numeric_only=True).reset_index()
    summary.to_csv(os.path.join(output_dir, "robustness_summary_table.csv"), index=False)
    
    with open(os.path.join(output_dir, "robustness_summary_table.md"), "w") as f:
        f.write("# Phase 1 Robustness Summary Table\n\n")
        f.write(summary.to_markdown(index=False))
        
    print(f"\n>>> Robustness tests complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
