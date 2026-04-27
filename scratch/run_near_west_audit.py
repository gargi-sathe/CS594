import subprocess
import os
import sys

# Capacity Variants: (K, D)
variants = [
    (3, 5, "current"),
    (4, 7, "medium"),
    (5, 8, "large")
]

zone = "Near West Side, Chicago, Illinois, USA"
model_path = "logs/checkpoints/chicago_realmap/chicago_scratch_24k/rl_model_18240_steps.zip"
config_path = "configs/chicago_expansion_eval.yaml"

for k, d, name in variants:
    print(f"\n>>> Auditing Capacity: K={k}, D={d} ({name})...")
    out_dir = f"results/chicago_expansion/near_west_audit_{name}"
    
    cmd = [
        sys.executable, "-m", "src.eval.run_robustness",
        "--model_path", model_path,
        "--config", config_path,
        "--output_dir", out_dir,
        "--osmnx_place", zone,
        "--num_warehouses", str(k),
        "--num_drivers", str(d)
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, env=env)

print("\n>>> Near West Side capacity audit complete!")
