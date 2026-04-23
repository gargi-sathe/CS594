import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    args = parser.parse_args()

    checkpoints = [os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir) if f.endswith(".zip")]
    checkpoints.sort()

    for cp in checkpoints:
        name = os.path.basename(cp).replace(".zip", "")
        out_dir = os.path.join(args.output_base, name)
        if os.path.exists(os.path.join(out_dir, "robustness_raw_results.csv")):
            print(f"Skipping {cp}, already evaluated.")
            continue
            
        print(f"\n>>> Evaluating {cp} ...")
        cmd = [
            "./venv/bin/python3", "-m", "src.eval.run_robustness",
            "--model_path", cp,
            "--output_dir", out_dir
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
