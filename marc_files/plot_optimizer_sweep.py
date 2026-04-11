"""
Plot training curves and per-run histograms for all runs in marc_optimizer_sweep.
Usage: python marc_files/plot_optimizer_sweep.py
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_ROOT = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/marc_optimizer_sweep")
OUT_DIR      = Path("/n/home06/mwalden/eoss/marc_files")
PLOT_SCRIPT  = Path("/n/home06/mwalden/eoss/plot_histograms.py")
PYTHON       = "/n/home06/mwalden/.conda/envs/eoss/bin/python"

OPTIMIZERS = ['SGD-Momentum', 'SGD-Nesterov', 'Muon', 'RMSProp', 'Adam']
BATCH_SIZES = [8, 32, 128, 1024, 8192]
OPT_COLORS  = {
    'SGD-Momentum': '#1f77b4',
    'SGD-Nesterov': '#ff7f0e',
    'Muon':         '#2ca02c',
    'RMSProp':      '#d62728',
    'Adam':         '#9467bd',
}
OPT_LRS = {
    'SGD-Momentum': 0.002,
    'SGD-Nesterov': 0.002,
    'Muon':         0.001,
    'RMSProp':      0.00003,
    'Adam':         0.00003,
}


def parse_opt(folder_name):
    m = re.search(r'_(SGD(?:-Momentum|-Nesterov)?|Adam|RMSProp|Muon)_', folder_name)
    return m.group(1) if m else None


def parse_batch(folder_name):
    m = re.search(r'_b(\d+)', folder_name)
    return int(m.group(1)) if m else -1


def load(folder):
    return pd.read_csv(folder / 'results.txt', comment='#')


# ── Collect runs ──────────────────────────────────────────────────────────────
runs = sorted(
    [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and (d / 'results.txt').exists()],
    key=lambda d: (parse_opt(d.name) or '', parse_batch(d.name))
)
if not runs:
    print(f"No runs found in {RESULTS_ROOT}")
    raise SystemExit(1)
print(f"Found {len(runs)} runs.")

# ── Training curves: one subplot per optimizer, batch size as linestyle shade ─
n_opt = len(OPTIMIZERS)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Optimizer sweep: CIFAR-10 MLP — lmax, batch sharpness, loss", fontsize=11)

bs_norm = plt.Normalize(np.log2(min(BATCH_SIZES)), np.log2(max(BATCH_SIZES)))
bs_cmap = cm.Blues

for folder in runs:
    opt = parse_opt(folder.name)
    bs  = parse_batch(folder.name)
    if opt not in OPT_COLORS or bs not in BATCH_SIZES:
        continue
    try:
        df = load(folder)
    except Exception as e:
        print(f"  Skipping {folder.name}: {e}")
        continue

    base_color = OPT_COLORS[opt]
    alpha = 0.4 + 0.5 * (np.log2(bs) - np.log2(min(BATCH_SIZES))) / (np.log2(max(BATCH_SIZES)) - np.log2(min(BATCH_SIZES)))
    lw = 0.8

    lm = df[['step', 'lmax']].dropna()
    if not lm.empty:
        axes[2].plot(lm['step'], lm['lmax'], color=base_color, linewidth=lw, alpha=alpha)

    bsh = df[['step', 'batch_sharpness']].dropna()
    if not bsh.empty:
        axes[1].plot(bsh['step'], bsh['batch_sharpness'], color=base_color, linewidth=lw, alpha=alpha)

    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        axes[0].semilogy(loss['step'], loss['full_loss'], color=base_color, linewidth=lw, alpha=alpha)

# Legend: one entry per optimizer
for opt, color in OPT_COLORS.items():
    lr = OPT_LRS[opt]
    for ax in axes:
        ax.plot([], [], color=color, linewidth=2, label=f'{opt} lr={lr}')

for ax in axes:
    ax.legend(fontsize=7, framealpha=0.7)

axes[0].set_title("Full-batch loss"); axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
axes[1].set_title("Batch sharpness"); axes[1].set_xlabel("step")
axes[2].set_title(r"$\lambda_{max}$"); axes[2].set_xlabel("step")

plt.tight_layout()
out = OUT_DIR / "optimizer_sweep_curves.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved training curves → {out}")
plt.close()

# ── Per-run histograms ────────────────────────────────────────────────────────
for folder in runs:
    npz = folder / 'projections.npz'
    if npz.exists():
        print(f"Plotting histograms for {folder.name} ...")
        subprocess.run([PYTHON, str(PLOT_SCRIPT), str(folder)], check=True)
    else:
        print(f"  No projections.npz in {folder.name}, skipping.")
