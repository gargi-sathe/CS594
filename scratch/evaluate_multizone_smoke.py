import subprocess
import os
import sys

zones = [
    ("Loop, Chicago, Illinois, USA", 3, 5),
    ("Near North Side, Chicago, Illinois, USA", 3, 5),
    ("Near West Side, Chicago, Illinois, USA", 5, 8),
    ("Hyde Park, Chicago, Illinois, USA", 3, 5)
]

models = {
    "Loop_18k_Champion": "logs/checkpoints/chicago_realmap/chicago_scratch_24k/rl_model_18240_steps.zip",
    "MZ_2k_Smoke": "logs/checkpoints/chicago_multizone/final_model.zip"
}

config_path = "configs/chicago_expansion_eval.yaml"

for zone_place, k, d in zones:
    zone_name = zone_place.split(",")[0].replace(" ", "_")
    for m_name, m_path in models.items():
        print(f"\n>>> Evaluating {m_name} on {zone_name} (K={k}, D={d})...")
        out_dir = f"results/chicago_multizone/smoke_eval/{zone_name}/{m_name}"
        
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

print("\n>>> Multi-zone smoke evaluation complete!")
