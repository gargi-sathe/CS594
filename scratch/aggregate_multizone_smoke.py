import pandas as pd
import os

zones = ["Loop", "Near_North_Side", "Near_West_Side", "Hyde_Park"]
models = ["Loop_18k_Champion", "MZ_2k_Smoke"]

summary_frames = []

for zone in zones:
    for model in models:
        csv_path = f"results/chicago_multizone/smoke_eval/{zone}/{model}/robustness_summary_table.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["zone"] = zone
            df["model_name"] = model
            summary_frames.append(df)

if not summary_frames:
    print("No results found!")
else:
    all_summary = pd.concat(summary_frames)
    
    # We want to compare RL Models (Loop Champion vs MZ Smoke) across zones
    # For a fair comparison, we focus on the MaskablePPO method rows
    rl_only = all_summary[all_summary["method"] == "MaskablePPO"]
    
    pivot_df = rl_only.pivot_table(
        index=["stress_type", "zone"], 
        columns="model_name", 
        values="on_time_delivery_rate"
    )
    
    # Add Baselines (from Loop_18k_Champion run since they are the same)
    baselines = all_summary[all_summary["model_name"] == "Loop_18k_Champion"]
    clust = baselines[baselines["method"] == "Clustering"].pivot_table(index=["stress_type", "zone"], values="on_time_delivery_rate").rename(columns={"on_time_delivery_rate": "Clustering"})
    greedy = baselines[baselines["method"] == "Greedy"].pivot_table(index=["stress_type", "zone"], values="on_time_delivery_rate").rename(columns={"on_time_delivery_rate": "Greedy"})
    
    final_pivot = pd.concat([pivot_df, clust, greedy], axis=1)
    
    print("\n# Multi-Zone Smoke Benchmark (On-Time Delivery Rate)")
    print(final_pivot.to_markdown())
    
    final_pivot.to_csv("results/chicago_multizone/smoke_eval_summary.csv")
