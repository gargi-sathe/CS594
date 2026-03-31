import os
import json
import csv
import yaml
from typing import Dict, Any, List
from dataclasses import asdict
from src.core.state import SimulatorState
from src.metrics.aggregator import calculate_metrics
from src.config.schema import RunConfig
from src.visualization.plotter import plot_synthetic_run

def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def write_outputs(state: SimulatorState, config: RunConfig, run_id: str = "run_001"):
    base_dir = config.outputs.output_dir
    run_dir = os.path.join(base_dir, run_id)
    _ensure_dir(run_dir)
    
    metrics = calculate_metrics(state, config)
    
    # 1. run_summary.json
    summary = {
        "run_id": run_id,
        "mode": config.mode,
        "seed": config.seed,
        "total_orders_generated": len(state.orders),
        "delivered_success_rate": metrics.get("delivered_success_rate", 0.0),
        "on_time_delivery_rate": metrics.get("on_time_delivery_rate", 0.0),
        "simple_cost_estimate": metrics.get("simple_cost_estimate", 0.0)
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, sort_keys=True)
        
    # 2. metrics.json
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)
        
    # 3. config_snapshot.yaml
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.dump({"run_config": asdict(config)}, f, sort_keys=True)
        
    # 4. event_log.csv
    with open(os.path.join(run_dir, "event_log.csv"), "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "entity_type", "entity_id", "event_type", "details"])
        for ts, e_type, e_id, e_event, details_str in state.event_log:
            try:
                # Basic parsing: 'k1=v1, k2=v2' -> {'k1':'v1', 'k2':'v2'}
                details_dict = {}
                for part in details_str.split(','):
                    if '=' in part:
                        k, v = part.strip().split('=', 1)
                        details_dict[k] = v
                    else:
                        details_dict["info"] = part.strip()
                final_details = json.dumps(details_dict, sort_keys=True)
            except Exception:
                final_details = json.dumps({"raw": details_str}, sort_keys=True)
                
            writer.writerow([ts, e_type, e_id, e_event, final_details])
            
    # 5. Serialization for map visual
    try:
        plot_synthetic_run(state, config, run_id, run_dir)
    except Exception as e:
        print(f"Warning: map_plot.png generation natively failed bounded dependencies: {e}")
