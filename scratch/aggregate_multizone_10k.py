import pandas as pd
import os

zones = ["Loop", "Near_North_Side", "Near_West_Side", "Hyde_Park"]
models = ["MZ_2k", "MZ_6k", "MZ_10k", "Loop_Champion"]

summary_frames = []

for zone in zones:
    for model in models:
        csv_path = f"results/chicago_multizone/10k_eval/{zone}/{model}/robustness_summary_table.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["zone"] = zone
            df["model_name"] = model
            summary_frames.append(df)

if not summary_frames:
    print("No results found!")
else:
    all_summary = pd.concat(summary_frames)
    
    rl_only = all_summary[all_summary["method"] == "MaskablePPO"]
    
    pivot_df = rl_only.pivot_table(
        index=["stress_type", "zone"], 
        columns="model_name", 
        values="on_time_delivery_rate"
    )
    
    baselines = all_summary[all_summary["model_name"] == "Loop_Champion"]
    clust = baselines[baselines["method"] == "Clustering"].pivot_table(index=["stress_type", "zone"], values="on_time_delivery_rate").rename(columns={"on_time_delivery_rate": "Clustering"})
    greedy = baselines[baselines["method"] == "Greedy"].pivot_table(index=["stress_type", "zone"], values="on_time_delivery_rate").rename(columns={"on_time_delivery_rate": "Greedy"})
    
    # Reorder columns to show training progress
    cols = ["MZ_2k", "MZ_6k", "MZ_10k", "Loop_Champion", "Clustering", "Greedy"]
    final_pivot = pd.concat([pivot_df, clust, greedy], axis=1)[cols]
    
    print("\n# Multi-Zone 10k Benchmark (On-Time Delivery Rate)")
    print(final_pivot.to_markdown())
    
    final_pivot.to_csv("results/chicago_multizone/10k_eval_summary.csv")
    
    # Full Metrics Aggregation for the Best MZ Model (10k)
    best_mz = all_summary[(all_summary["model_name"] == "MZ_10k") & (all_summary["method"] == "MaskablePPO")]
    best_mz.to_csv("results/chicago_multizone/10k_full_metrics.csv")
