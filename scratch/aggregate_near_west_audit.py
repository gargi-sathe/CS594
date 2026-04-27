import pandas as pd
import os

variants = {
    "Current (K=3, D=5)": "results/chicago_expansion/near_west_audit_current",
    "Medium (K=4, D=7)": "results/chicago_expansion/near_west_audit_medium",
    "Large (K=5, D=8)": "results/chicago_expansion/near_west_audit_large"
}

summary_frames = []

for name, path in variants.items():
    csv_path = os.path.join(path, "robustness_summary_table.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["variant"] = name
        summary_frames.append(df)

if not summary_frames:
    print("No results found!")
else:
    all_summary = pd.concat(summary_frames)
    
    pivot_df = all_summary.pivot_table(
        index=["stress_type", "method"], 
        columns="variant", 
        values="on_time_delivery_rate"
    )
    
    # Reorder columns
    cols = ["Current (K=3, D=5)", "Medium (K=4, D=7)", "Large (K=5, D=8)"]
    pivot_df = pivot_df[cols]
    
    print("\n# Near West Side Capacity Audit (On-Time Delivery Rate)")
    print(pivot_df.to_markdown())
    
    pivot_df.to_csv("results/chicago_expansion/near_west_capacity_audit_raw.csv")
