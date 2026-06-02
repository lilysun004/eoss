#!/bin/bash
#SBATCH -J curv_smoke
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_smoke_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_smoke_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

set -e

echo "[$(date)] Starting curvature smoke test"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn --dataset cifar10 --num_data 8192 \
    --optimizer_name SGD --lr 0.02 --batch_size 32 \
    --steps 200 --track_from 100 --track_until 200 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 20 \
    --results_subfolder smoke_curvature

echo "[$(date)] Training done — running sanity checks"

# Find the run folder (most recent under smoke_curvature) and validate npz
/n/home06/mwalden/.conda/envs/eoss/bin/python - <<'PY'
import os, sys, glob
import numpy as np

root = os.path.join(os.environ['RESULTS'], 'smoke_curvature')
runs = sorted(glob.glob(os.path.join(root, '2*')))
assert runs, f"no run folders under {root}"
run = runs[-1]
print(f"Inspecting {run}")

npz_path = os.path.join(run, 'curvature_segment.npz')
assert os.path.exists(npz_path), f"missing {npz_path}"
d = np.load(npz_path)

print("Keys:", list(d.keys()))
print("steps =", d['steps'])
print("alphas =", d['alphas'])
print("curv_segment.shape =", d['curv_segment'].shape)
print("curv_along_u.shape =", d['curv_along_u'].shape)
print("betas.shape       =", d['betas'].shape)
print("lambda_mid =", d['lambda_mid'])
print("lambda_w_t =", d['lambda_w_t'])
print("step_proj_u_t =", d['step_proj_u_t'])

# (1) shape sanity
n_meas = len(d['steps'])
assert d['curv_segment'].shape == (n_meas, 13), d['curv_segment'].shape
assert d['curv_along_u'].shape == (n_meas, 9),  d['curv_along_u'].shape
assert d['betas'].shape       == (n_meas, 9),  d['betas'].shape

# (2) self-consistency: scan value at α=0.5 ≈ lambda_mid
alpha_idx_half = int(np.argmin(np.abs(d['alphas'] - 0.5)))
print(f"alpha_idx for α=0.5 = {alpha_idx_half}, α = {d['alphas'][alpha_idx_half]}")
S_half = d['curv_segment'][:, alpha_idx_half]
rel = np.abs(S_half - d['lambda_mid']) / (np.abs(d['lambda_mid']) + 1e-12)
print(f"|S(0.5) - λ_mid|/|λ_mid| max = {rel.max():.4g}, mean = {rel.mean():.4g}")
assert rel.max() < 0.10, f"S(0.5) deviates too much from λ_mid: {rel}"

# (3) self-consistency: scan value at β=0 ≈ lambda_w_t
beta_idx_zero = int(np.argmin(np.abs(d['betas'][0])))
print(f"beta_idx for β=0 = {beta_idx_zero}, β[0,{beta_idx_zero}] = {d['betas'][0, beta_idx_zero]}")
S_zero = d['curv_along_u'][:, beta_idx_zero]
rel_b = np.abs(S_zero - d['lambda_w_t']) / (np.abs(d['lambda_w_t']) + 1e-12)
print(f"|S(β=0) - λ_w_t|/|λ_w_t| max = {rel_b.max():.4g}, mean = {rel_b.mean():.4g}")
assert rel_b.max() < 0.10, f"S(β=0) deviates too much from λ_w_t: {rel_b}"

print("\nALL SMOKE-TEST CHECKS PASSED")
PY

echo "[$(date)] Sanity checks passed"
