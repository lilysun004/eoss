"""
Plot training curves (loss, batch sharpness, lmax) for all runs in marc_batch_sweep,
and generate per-run histogram plots.

Usage: python marc_files/plot_batch_sweep.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_ROOT = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/marc_batch_sweep")
OUT_DIR = Path("/n/home06/mwalden/eoss/marc_files")
PLOT_SCRIPT = Path("/n/home06/mwalden/eoss/plot_histograms.py")


def parse_batch_size(folder_name: str) -> int:
    m = re.search(r'_b(\d+)', folder_name)
    return int(m.group(1)) if m else -1


def load(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "results.txt", comment="#")


# ── Collect runs ──────────────────────────────────────────────────────────────
runs = sorted(
    [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and (d / "results.txt").exists()],
    key=lambda d: parse_batch_size(d.name)
)

if not runs:
    print(f"No runs found in {RESULTS_ROOT}")
    raise SystemExit(1)

batch_sizes = [parse_batch_size(r.name) for r in runs]
print(f"Found {len(runs)} runs: batch sizes {batch_sizes}")

# ── Colormap: log-spaced batch size → color ───────────────────────────────────
log_bs = np.log2(np.array(batch_sizes, dtype=float))
norm = plt.Normalize(log_bs.min(), log_bs.max())
cmap = cm.viridis

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Batch sweep: SGD lr=0.02, CIFAR-10 MLP, 40k steps", fontsize=12)

for folder, bs in zip(runs, batch_sizes):
    try:
        df = load(folder)
    except Exception as e:
        print(f"  Skipping {folder.name}: {e}")
        continue

    color = cmap(norm(np.log2(bs)))

    loss = df[["step", "full_loss"]].dropna()
    if not loss.empty:
        axes[0].semilogy(loss["step"], loss["full_loss"], color=color, linewidth=0.8, alpha=0.85)

    bs_col = df[["step", "batch_sharpness"]].dropna()
    if not bs_col.empty:
        axes[1].plot(bs_col["step"], bs_col["batch_sharpness"], color=color, linewidth=0.8, alpha=0.85)

    lm = df[["step", "lmax"]].dropna()
    if not lm.empty:
        axes[2].plot(lm["step"], lm["lmax"], color=color, linewidth=0.8, alpha=0.85)

# 2/lr reference line
for ax in axes[1:]:
    ax.axhline(2 / 0.02, color='black', linestyle='--', linewidth=1, label='2/lr = 100')
    ax.legend(fontsize=8)

axes[0].set_title("Full-batch loss"); axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
axes[1].set_title("Batch sharpness"); axes[1].set_xlabel("step")
axes[2].set_title(r"$\lambda_{max}$"); axes[2].set_xlabel("step")

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
cbar.set_label("log₂(batch size)", fontsize=9)
tick_bs = [b for b in batch_sizes if b in {2, 8, 32, 128, 512, 2048, 8192}]
cbar.set_ticks([np.log2(b) for b in tick_bs])
cbar.set_ticklabels([str(b) for b in tick_bs])

out = OUT_DIR / "batch_sweep_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved training curves → {out}")
plt.close()

# ── Per-run histograms ────────────────────────────────────────────────────────
import subprocess, sys
python = "/n/home06/mwalden/.conda/envs/eoss/bin/python"

for folder in runs:
    npz = folder / "projections.npz"
    if npz.exists():
        print(f"Plotting histograms for {folder.name} ...")
        subprocess.run([python, str(PLOT_SCRIPT), str(folder)], check=True)
    else:
        print(f"  No projections.npz in {folder.name}, skipping histogram.")
