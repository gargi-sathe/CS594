import pandas as pd
import os

zones = {
    "Loop": "results/chicago_expansion/Loop",
    "Near North Side": "results/chicago_expansion/Near_North_Side",
    "Near West Side": "results/chicago_expansion/Near_West_Side",
    "Hyde Park": "results/chicago_expansion/Hyde_Park"
}

summary_frames = []

for name, path in zones.items():
    csv_path = os.path.join(path, "robustness_summary_table.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["zone"] = name
        summary_frames.append(df)

if not summary_frames:
    print("No results found!")
else:
    all_summary = pd.concat(summary_frames)
    
    # We want to compare RL vs Clustering vs Greedy across zones
    # Metric: on_time_delivery_rate
    
    pivot_df = all_summary.pivot_table(
        index=["stress_type", "method"], 
        columns="zone", 
        values="on_time_delivery_rate"
    )
    
    # Reorder zones for clarity
    cols = ["Loop", "Near North Side", "Near West Side", "Hyde Park"]
    pivot_df = pivot_df[cols]
    
    print("\n# Multi-Zone Expansion Benchmark (On-Time Delivery Rate)")
    print(pivot_df.to_markdown())
    
    pivot_df.to_csv("results/chicago_expansion/zone_comparison.csv")
    
    # Identify gaps
    print("\n# Generalization Gap (RL vs Clustering)")
    rl_rows = all_summary[all_summary["method"] == "MaskablePPO"].set_index(["stress_type", "zone"])["on_time_delivery_rate"]
    clust_rows = all_summary[all_summary["method"] == "Clustering"].set_index(["stress_type", "zone"])["on_time_delivery_rate"]
    gap = rl_rows - clust_rows
    print(gap.unstack().to_markdown())
