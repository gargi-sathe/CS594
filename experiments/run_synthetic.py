import argparse
import os
import yaml
from src.config.schema import parse_config
from src.core.engine import run_simulation
from src.metrics.writer import write_outputs

def main():
    parser = argparse.ArgumentParser(description="Run synthetic UrbanScale Phase 0 simulator")
    parser.add_argument("--config", type=str, required=True, help="Path to run config YAML")
    parser.add_argument("--run_id", type=str, default="demo_run", help="Unique explicit run identity string")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
        
    config = parse_config(raw_cfg)
    
    print(f"Starting simulation for run_id={args.run_id}, horizon={config.run_horizon_mins} mins...")
    state = run_simulation(config)
    print(f"Simulation physically completed. Processed {len(state.orders)} orders generated.")
    
    print("Calculating passive metric traces and writing outputs...")
    write_outputs(state, config, run_id=args.run_id)
    print(f"Success! Artifacts precisely validated into -> {os.path.join(config.outputs.output_dir, args.run_id)}")

if __name__ == "__main__":
    main()
