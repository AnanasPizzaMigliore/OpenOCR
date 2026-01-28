import os
import subprocess
import itertools

# Define your config path
config_path = "/scratch/penghao/OpenOCR/configs/det/dbnet/timm_real_repvit.yml"

# Define the sweep ranges
thresh_values = [0.3, 0.4, 0.5]
box_thresh_values = [0.6, 0.7, 0.8]

# Create all (t, b) pairs
combinations = list(itertools.product(thresh_values, box_thresh_values))

# Output log file
log_file = "postprocess_sweep_results.txt"

with open(log_file, "w") as log:
    for t, b in combinations:
        cmd = [
            "python", "tools/eval_det.py",
            "-c", config_path,
            "-o",
            f"PostProcess.thresh={t}",
            f"PostProcess.box_thresh={b}"
        ]
        print(f"\n>>> Running thresh={t}, box_thresh={b}")
        log.write(f"\n>>> Running thresh={t}, box_thresh={b}\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log.write(result.stdout)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error at thresh={t}, box_thresh={b}")
            log.write(f"Error: {e}\n{e.stdout}\n{e.stderr}\n")

print(f"\n✅ Sweep complete! Results saved to {log_file}")
