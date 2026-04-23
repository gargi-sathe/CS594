import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs='+', required=True, help="List of result directories to aggregate")
    parser.add_argument("--baseline_ref", type=str, default="results/pilot_eval/C2", help="Directory containing baseline results")
    args = parser.parse_args()

    all_dfs = []
    for d in args.dirs:
        csv_path = os.path.join(d, "robustness_raw_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['source'] = os.path.basename(d)
            all_dfs.append(df)

    if not all_dfs:
        print("No results found.")
        return

    big_df = pd.concat(all_dfs)
    
    # MaskablePPO Only Progression
    ppo = big_df[big_df['method'] == 'MaskablePPO']
    summary = ppo.pivot_table(index='stress_type', columns='source', values='on_time_delivery_rate', aggfunc='mean')
    
    print("\n### On-Time Delivery Rate Progression (RL Only)")
    print(summary.to_markdown())

    # Final Comparison vs Baselines
    # Pull baselines from the last directory or the explicit reference
    ref_path = os.path.join(args.baseline_ref, "robustness_raw_results.csv")
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
        baselines = ref_df[ref_df['method'] != 'MaskablePPO']
        base_summary = baselines.pivot_table(index='stress_type', columns='method', values='on_time_delivery_rate', aggfunc='mean')
        
        # Add the latest RL from summary
        latest_col = summary.columns[-1]
        final_comp = base_summary.copy()
        final_comp[f"RL_{latest_col}"] = summary[latest_col]
        
        print("\n### Latest RL vs Baselines (On-Time Rate)")
        print(final_comp.to_markdown())

if __name__ == "__main__":
    main()
