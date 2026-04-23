import os
import json
import pandas as pd
import argparse
from src.eval.run_evaluation import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Compare multiple Phase 1 checkpoints.")
    parser.add_argument("--config", type=str, default="configs/eval_selection.yaml")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True, 
                        help="Paths to checkpoint .zip files")
    parser.add_argument("--output_dir", type=str, default="results/phase1_selection/comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    all_summary_data = []

    for cp_path in args.checkpoints:
        cp_name = os.path.splitext(os.path.basename(cp_path))[0]
        print(f"\n>>> Evaluating checkpoint: {cp_name}...")
        
        # Run evaluation for this checkpoint in a specific subdir
        cp_output_dir = os.path.join(args.output_dir, "raw", cp_name)
        run_evaluation(
            config_path=args.config,
            model_path=cp_path,
            output_dir=cp_output_dir
        )
        
        # Load the generated summary
        summary_path = os.path.join(cp_output_dir, "results_summary.json")
        with open(summary_path, "r") as f:
            data = json.load(f)
            
        # Flatten the results for the table
        for res in data["results"]:
            row = {
                "checkpoint": cp_name,
                "bucket": res["bucket"],
                "method": res["method"],
                "on_time_rate": res["on_time_delivery_rate"],
                "success_rate": res["delivered_success_rate"],
                "p95_time": res["p95_delivery_time"],
                "missed_orders": res["orders_missed_or_unserved"],
                "cost_per_order": res["cost_per_order"],
                "driver_util": res["driver_utilization"]
            }
            all_summary_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(all_summary_data)
    
    # Save master comparison table
    df.to_csv(os.path.join(args.output_dir, "master_comparison_table.csv"), index=False)
    
    # Create a nice markdown table
    with open(os.path.join(args.output_dir, "master_comparison_table.md"), "w") as f:
        f.write("# Phase 1 Checkpoint Comparison Table\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"\n>>> Comparison complete! Master table saved to {args.output_dir}")

if __name__ == "__main__":
    main()
