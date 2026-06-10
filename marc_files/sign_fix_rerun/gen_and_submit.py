import json, os, re, subprocess

CFG_PATH = "/tmp/ambiguous_configs.json"
SCRIPT_DIR = "/n/home06/mwalden/eoss/marc_files/sign_fix_rerun"
LOG_DIR = "/n/home06/mwalden/eoss/marc_files/logs"

with open(CFG_PATH) as f:
    configs = json.load(f)

TEMPLATE = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p kempner_h100
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
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \\
    --model {model} \\
    --dataset {dataset} \\
    --num_data {num_data} \\
    --optimizer_name {optimizer_name} \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --optimizer_params "{optimizer_params}" \\
    --loss_type {loss_type} \\
    --steps {steps} \\
    --track_from {track_from} \\
    --track_until {track_until} \\
    --more_freq_measure True \\
    --fixed_u True \\
    --results_subfolder {results_subfolder}
"""

# rough wall-clock budget by step count (steps -> hours)
def time_budget(steps):
    if steps <= 50000:
        return "06:00:00"
    elif steps <= 100000:
        return "10:00:00"
    else:
        return "16:00:00"

subfolder_map = {
    "MLP_sweep": "MLP_sweep_signfix",
    "marc_vit_sweep": "marc_vit_sweep_signfix",
    "SST_opt_batch_sweep": "SST_opt_batch_sweep_signfix",
}

submitted = []
for i, c in enumerate(configs):
    sweep = c["run"].split("/")[0]
    results_subfolder = subfolder_map[sweep]

    opt_safe = re.sub(r'[^A-Za-z0-9]+', '', c["optimizer_name"])
    job_name = f"sf{i:02d}_{c['model']}_{opt_safe}_b{c['batch_size']}"
    job_name = job_name[:30]

    script = TEMPLATE.format(
        job_name=job_name,
        time=time_budget(c["steps"]),
        log_dir=LOG_DIR,
        model=c["model"],
        dataset=c["dataset"],
        num_data=c["num_data"],
        optimizer_name=c["optimizer_name"],
        lr=c["lr"],
        batch_size=c["batch_size"],
        optimizer_params=c["optimizer_params"],
        loss_type=c["loss"],
        steps=c["steps"],
        track_from=c["track_from"],
        track_until=c["track_until"],
        results_subfolder=results_subfolder,
    )

    fname = os.path.join(SCRIPT_DIR, f"run_{i:02d}_{job_name}.sh")
    with open(fname, "w") as f:
        f.write(script)
    os.chmod(fname, 0o755)

    out = subprocess.run(["sbatch", fname], capture_output=True, text=True)
    print(f"[{i:02d}] {c['run']}")
    print(f"     -> {fname}")
    print(f"     -> {out.stdout.strip()} {out.stderr.strip()}")
    submitted.append((c["run"], fname, out.stdout.strip()))

print("\nSubmitted", len(submitted), "jobs")
