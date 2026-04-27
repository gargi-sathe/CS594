import pandas as pd
import os

results_dirs = {
    "Synthetic Champion (Zero-Shot)": "results/chicago_eval_smoke/manual_test",
    "Warm-Start Chicago (1k)": "results/chicago_eval_smoke/smoke_A_warm",
    "Scratch Chicago (1k)": "results/chicago_eval_smoke/smoke_B_scratch"
}

summary_frames = []

for name, path in results_dirs.items():
    csv_path = os.path.join(path, "robustness_summary_table.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["model_source"] = name
        summary_frames.append(df)

if not summary_frames:
    print("No results found!")
else:
    all_summary = pd.concat(summary_frames)
    
    # Pivot for easier comparison
    # We'll focus on on_time_delivery_rate for now
    pivot_df = all_summary.pivot_table(
        index=["stress_type", "method"], 
        columns="model_source", 
        values="on_time_delivery_rate"
    )
    
    print("\n# Chicago Real-Map Smoke Comparison (On-Time Delivery Rate)")
    print(pivot_df.to_markdown())
    
    pivot_df.to_csv("results/chicago_eval_smoke/comparison_pivot.csv")
