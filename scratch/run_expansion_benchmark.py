import subprocess
import os
import sys

zones = [
    "Loop, Chicago, Illinois, USA",
    "Near North Side, Chicago, Illinois, USA",
    "Near West Side, Chicago, Illinois, USA",
    "Hyde Park, Chicago, Illinois, USA"
]

model_path = "logs/checkpoints/chicago_realmap/chicago_scratch_24k/rl_model_18240_steps.zip"
config_path = "configs/chicago_expansion_eval.yaml"

for zone in zones:
    name = zone.split(",")[0].replace(" ", "_")
    print(f"\n>>> Benchmarking Zone: {zone}...")
    out_dir = f"results/chicago_expansion/{name}"
    
    cmd = [
        sys.executable, "-m", "src.eval.run_robustness",
        "--model_path", model_path,
        "--config", config_path,
        "--output_dir", out_dir,
        "--osmnx_place", zone
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, env=env)

print("\n>>> Multi-zone expansion benchmark complete!")
