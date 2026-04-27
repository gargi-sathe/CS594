import subprocess
import os
import sys

zones = [
    ("Loop, Chicago, Illinois, USA", 3, 5),
    ("Near North Side, Chicago, Illinois, USA", 3, 5),
    ("Near West Side, Chicago, Illinois, USA", 5, 8),
    ("Hyde Park, Chicago, Illinois, USA", 3, 5)
]

# We will evaluate 2k, 6k, 10k, and final to save time while showing trend
checkpoint_dir = "logs/checkpoints/chicago_multizone/chicago_multizone_10k"
models = {
    "MZ_2k": os.path.join(checkpoint_dir, "rl_model_2000_steps.zip"),
    "MZ_6k": os.path.join(checkpoint_dir, "rl_model_6000_steps.zip"),
    "MZ_10k": os.path.join(checkpoint_dir, "rl_model_10000_steps.zip"),
    "Loop_Champion": "logs/checkpoints/chicago_realmap/chicago_scratch_24k/rl_model_18240_steps.zip"
}

config_path = "configs/chicago_expansion_eval.yaml"

for zone_place, k, d in zones:
    zone_name = zone_place.split(",")[0].replace(" ", "_")
    for m_name, m_path in models.items():
        if not os.path.exists(m_path):
            print(f"Skipping {m_name}, path not found: {m_path}")
            continue
            
        print(f"\n>>> Evaluating {m_name} on {zone_name} (K={k}, D={d})...")
        out_dir = f"results/chicago_multizone/10k_eval/{zone_name}/{m_name}"
        
        cmd = [
            sys.executable, "-m", "src.eval.run_robustness",
            "--model_path", m_path,
            "--config", config_path,
            "--output_dir", out_dir,
            "--osmnx_place", zone_place,
            "--num_warehouses", str(k),
            "--num_drivers", str(d)
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        subprocess.run(cmd, env=env)

print("\n>>> Multi-zone 10k evaluation complete!")
