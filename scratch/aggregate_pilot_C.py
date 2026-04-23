import pandas as pd
import os

paths = {
    "DR_only_7k": "results/robust_full_eval/rl_model_7000_steps/robustness_raw_results.csv",
    "C1_Moderate": "results/pilot_eval/C1/robustness_raw_results.csv",
    "C2_Strong": "results/pilot_eval/C2/robustness_raw_results.csv"
}

all_dfs = []

for name, path in paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['source'] = name
        all_dfs.append(df)

if not all_dfs:
    print("No results found.")
    exit()

big_df = pd.concat(all_dfs)

# Compare RL methods across variants
ppo_results = big_df[big_df['method'] == 'MaskablePPO']
summary = ppo_results.groupby(['stress_type', 'source'])['on_time_delivery_rate'].mean().unstack()

print("\n### On-Time Delivery Rate: Reward Shaping Comparison (RL Only)")
print(summary.to_markdown())

# Compare best C variant against Greedy
best_source = "C1_Moderate" # Placeholder, checking C2 too
greedy_ref = big_df[(big_df['method'] == 'Greedy') & (big_df['source'] == 'C1_Moderate')]
greedy_mean = greedy_ref.groupby('stress_type')['on_time_delivery_rate'].mean()

print("\n### Best C-Variant vs Greedy (On-Time Rate)")
best_ppo = ppo_results.groupby(['stress_type', 'source'])['on_time_delivery_rate'].mean().loc[:, ["C1_Moderate", "C2_Strong"]]
comp = pd.DataFrame({
    "Greedy": greedy_mean,
    "C1_Moderate": best_ppo["C1_Moderate"],
    "C2_Strong": best_ppo["C2_Strong"]
})
print(comp.to_markdown())
