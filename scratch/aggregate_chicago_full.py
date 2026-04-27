import pandas as pd
import os

results_dirs = {
    "Synthetic Champion": "results/chicago_eval_smoke/manual_test", # Reusing previous manual test for champ
    "Chicago Scratch 2k": "results/chicago_realmap/chicago_2k",
    "Chicago Scratch 6k": "results/chicago_realmap/chicago_6k",
    "Chicago Scratch 10k": "results/chicago_realmap/chicago_10k",
    "Chicago Scratch Final": "results/chicago_realmap/chicago_final"
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
    cols = ["Synthetic Champion", "Chicago Scratch 2k", "Chicago Scratch 6k", "Chicago Scratch 10k", "Chicago Scratch Final"]
    pivot_df = pivot_df[cols]
    
    print("\n# Chicago Real-Map Training Comparison (On-Time Delivery Rate)")
    print(pivot_df.to_markdown())
    
    pivot_df.to_csv("results/chicago_realmap/checkpoint_comparison.csv")
    
    # Average performance across models for baselines should be identical
    # We can just pick one row for Greedy/Clustering
