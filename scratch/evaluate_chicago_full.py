import subprocess
import os
import sys

# Note: SB3 MaskablePPO.load(path) appends .zip if not present, 
# but sometimes it appends even if it is present depending on the path object.
# To be safe, we provide paths without .zip for checkpointed ones.
models = {
    # "synthetic_champion": "logs/checkpoints/pilot_C_full/pilot_C2_full_run/final_model",
    # "chicago_10k": "logs/checkpoints/chicago_realmap/chicago_scratch_10k/final_model",
    "chicago_18k": "logs/checkpoints/chicago_realmap/chicago_scratch_24k/rl_model_18240_steps",
    "chicago_22k": "logs/checkpoints/chicago_realmap/chicago_scratch_24k/rl_model_22240_steps",
    "chicago_24k": "logs/checkpoints/chicago_realmap/chicago_scratch_24k/final_model"
}

config_path = "configs/chicago_eval_smoke.yaml"

for name, path in models.items():
    print(f"\n>>> Evaluating {name}...")
    out_dir = f"results/chicago_realmap/{name}"
    cmd = [
        sys.executable, "-m", "src.eval.run_robustness",
        "--model_path", path,
        "--config", config_path,
        "--output_dir", out_dir
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, env=env)

print("\n>>> All Chicago 24k evaluations complete!")
