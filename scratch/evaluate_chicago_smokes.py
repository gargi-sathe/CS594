import subprocess
import os
import sys

models = {
    # "synthetic_champion": "logs/checkpoints/pilot_C_full/pilot_C2_full_run/final_model.zip",
    "smoke_A_warm": "logs/checkpoints/chicago_smoke/smoke_A_warm_v2/final_model.zip",
    "smoke_B_scratch": "logs/checkpoints/chicago_smoke/smoke_B_scratch_v2/final_model.zip"
}

config_path = "configs/chicago_eval_smoke.yaml"

for name, path in models.items():
    print(f"\n>>> Evaluating {name}...")
    out_dir = f"results/chicago_eval_smoke/{name}"
    cmd = [
        sys.executable, "-m", "src.eval.run_robustness",
        "--model_path", path,
        "--config", config_path,
        "--output_dir", out_dir
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, env=env)

print("\n>>> All Chicago evaluations complete!")
