import pandas as pd
import os

results_dirs = {
    "Synthetic Champion": "results/chicago_eval_smoke/manual_test",
    "Chicago 10k": "results/chicago_realmap/chicago_10k",
    "Chicago 18k": "results/chicago_realmap/chicago_18k",
    "Chicago 22k": "results/chicago_realmap/chicago_22k",
    "Chicago 24k": "results/chicago_realmap/chicago_24k"
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
    pivot_df = all_summary.pivot_table(
        index=["stress_type", "method"], 
        columns="model_source", 
        values="on_time_delivery_rate"
    )
    
    # Reorder columns logically
    cols = ["Synthetic Champion", "Chicago 10k", "Chicago 18k", "Chicago 22k", "Chicago 24k"]
    pivot_df = pivot_df[cols]
    
    print("\n# Chicago 24k Training Comparison (On-Time Delivery Rate)")
    print(pivot_df.to_markdown())
    
    pivot_df.to_csv("results/chicago_realmap/final_comparison_pivot.csv")
