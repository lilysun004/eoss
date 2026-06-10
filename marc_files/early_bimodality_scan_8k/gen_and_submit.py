import os, subprocess

SCRIPT_DIR = "/n/home06/mwalden/eoss/marc_files/early_bimodality_scan_8k"
LOG_DIR = "/n/home06/mwalden/eoss/marc_files/logs"
os.makedirs(SCRIPT_DIR, exist_ok=True)

TEMPLATE = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t {time}
#SBATCH -o {log_dir}/{job_name}_%j.out
#SBATCH -e {log_dir}/{job_name}_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/early_bimodality_scan_8k/results
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
    --track_stride {track_stride} \\
    --more_freq_measure True \\
    --fixed_u True \\
    --results_subfolder early_scan_{model}_8k
"""

# Track the ENTIRE 8000-step run instead of a late or short-early window,
# to see whether genuine (non-artifact) bimodality emerges anywhere across
# the full trajectory at large step sizes (per Avrajit's hypothesis).
# CNN measurements are ~8.8s each -> stride=8 keeps cost ~similar to the
# previous 2000-step/stride=2 attempt (~1000 measurements, ~2.5h) while
# covering the full 8000-step run. MLP is cheap -> stride=2 is fine.
cells = []
for lr in [0.02, 0.04, 0.08, 0.16]:
    cells.append(dict(model="cnn", num_data=16384, batch_size=32, lr=lr,
                      steps=8000, track_from=0, track_until=8000,
                      track_stride=8, time="04:00:00"))
for lr in [0.02, 0.05, 0.1, 0.2]:
    cells.append(dict(model="mlp", num_data=8192, batch_size=32, lr=lr,
                      steps=8000, track_from=0, track_until=8000,
                      track_stride=2, time="02:00:00"))

submitted = []
for i, c in enumerate(cells):
    job_name = f"e8{i:02d}_{c['model']}_lr{c['lr']}"[:30]
    script = TEMPLATE.format(job_name=job_name, log_dir=LOG_DIR, **c)
    fname = os.path.join(SCRIPT_DIR, f"run_{i:02d}_{job_name}.sh")
    with open(fname, "w") as f:
        f.write(script)
    os.chmod(fname, 0o755)
    out = subprocess.run(["sbatch", "--partition=kempner_requeue", fname], capture_output=True, text=True)
    print(f"[{i:02d}] model={c['model']} lr={c['lr']} -> {fname}")
    print(f"     -> {out.stdout.strip()} {out.stderr.strip()}")
    submitted.append((c, fname, out.stdout.strip()))

print(f"\nSubmitted {len(submitted)} jobs")
