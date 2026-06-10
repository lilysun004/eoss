import os, subprocess

SCRIPT_DIR = "/n/home06/mwalden/eoss/marc_files/early_bimodality_scan"
LOG_DIR = "/n/home06/mwalden/eoss/marc_files/logs"
os.makedirs(SCRIPT_DIR, exist_ok=True)

TEMPLATE = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o {log_dir}/{job_name}_%j.out
#SBATCH -e {log_dir}/{job_name}_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/early_bimodality_scan/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \\
    --model {model} \\
    --dataset cifar10 \\
    --num_data {num_data} \\
    --optimizer_name SGD \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --steps {steps} \\
    --stop_loss None \\
    --track_from {track_from} \\
    --track_until {track_until} \\
    --track_stride 2 \\
    --more_freq_measure True \\
    --fixed_u True \\
    --results_subfolder early_scan_{model}
"""

# Baselines that produced strong late-window bimodality signatures (now collapsed):
#   CNN: SGD lr=0.02 b=32, num_data=16384
#   MLP: SGD lr=0.02 b=8 (and lr=0.005 b=128 for the long sweep)
# Avrajit: bimodality shows up EARLY in training, and larger step size -> more likely.
# So scan a grid of LRs at/above the baseline, tracking an early window (steps ~300-2300)
# instead of the tail. steps=3000 total is enough to cover that window plus warmup.

cells = []
for lr in [0.02, 0.04, 0.08, 0.16]:
    cells.append(dict(model="cnn", num_data=16384, batch_size=32, lr=lr,
                      steps=3000, track_from=300, track_until=2300))
for lr in [0.02, 0.05, 0.1, 0.2]:
    cells.append(dict(model="mlp", num_data=8192, batch_size=32, lr=lr,
                      steps=3000, track_from=300, track_until=2300))

submitted = []
for i, c in enumerate(cells):
    job_name = f"eb{i:02d}_{c['model']}_lr{c['lr']}"[:30]
    script = TEMPLATE.format(job_name=job_name, log_dir=LOG_DIR, **c)
    fname = os.path.join(SCRIPT_DIR, f"run_{i:02d}_{job_name}.sh")
    with open(fname, "w") as f:
        f.write(script)
    os.chmod(fname, 0o755)
    out = subprocess.run(["sbatch", fname], capture_output=True, text=True)
    print(f"[{i:02d}] model={c['model']} lr={c['lr']} -> {fname}")
    print(f"     -> {out.stdout.strip()} {out.stderr.strip()}")
    submitted.append((c, fname, out.stdout.strip()))

print(f"\nSubmitted {len(submitted)} jobs")
