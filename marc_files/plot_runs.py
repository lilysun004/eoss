"""
Plot training loss, batch_sharpness, and lmax over time for the two test runs.
Usage: python marc_files/plot_runs.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = [
    Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/marc_projection_test/20260410_2020_09_SGD_lr0.02_b64"),
    Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/marc_projection_test/20260410_2043_32_SGD-Momentum_lr0.002_b64_beta-0.9"),
]
LABELS = ["SGD  lr=0.02", "SGD-Momentum  lr=0.002  β=0.9"]
LRS    = [0.02, 0.002]


def load(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "results.txt", comment="#")


def rolling(s: pd.Series, frac: float = 0.01) -> pd.Series:
    w = max(1, int(len(s) * frac))
    return s.rolling(w, min_periods=1, center=True).mean()


fig, axes = plt.subplots(1, 3, figsize=(16, 4))
colors = ["#1f77b4", "#d62728"]

for df, label, lr, color in zip([load(r) for r in RUNS], LABELS, LRS, colors):
    # --- Loss ---
    loss = df[["step", "full_loss"]].dropna()
    axes[0].semilogy(loss["step"], loss["full_loss"], label=label, color=color)

    # --- Batch sharpness ---
    bs = df[["step", "batch_sharpness"]].dropna()
    axes[1].plot(bs["step"], bs["batch_sharpness"], label=label, color=color)

    # --- lmax ---
    lm = df[["step", "lmax"]].dropna()
    axes[2].plot(lm["step"], lm["lmax"], label=label, color=color)

# 2/lr threshold lines
for lr, color in zip(LRS, colors):
    for ax in axes[1:]:
        ax.axhline(2 / lr, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

axes[0].set_title("Full-batch loss"); axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
axes[1].set_title("Batch sharpness"); axes[1].set_xlabel("step"); axes[1].set_ylabel(r"$E[g^THg/\|g\|^2]$")
axes[2].set_title(r"$\lambda_{max}$"); axes[2].set_xlabel("step"); axes[2].set_ylabel(r"$\lambda_{max}$")

for ax in axes:
    ax.legend(fontsize=8)

plt.tight_layout()
out = Path(__file__).parent / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
