"""
Render 2-row paper figures for CNN, ViT, and SSTTransformer:
  Row 0: Training dynamics (lmax, batch sharpness, loss)
  Row 1: <theta_t, w_full_k(t)> histograms (proj_w_top5, full-H per-step eigenvecs)

Output: marc_files/curvature_results/paper_bimodality/
  CNN_SGD_lr0.02_b32.png
  ViT_Muon_lr0.003_b128.png
  SST_Adam_lr3e-05_b8192.png

Run from repo root:
  python marc_files/curvature_results/make_paper_bimodality_figures.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

# ── import helpers from plot_histograms.py ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_histograms import (
    plot_training_curve, plot_histogram, _all_nan, K_COLORS, TOP_K
)

# ── source run folders ────────────────────────────────────────────────────────
RUNS = [
    dict(
        label="CNN · SGD · lr=0.02 · b=32",
        out_name="CNN_SGD_lr0.02_b32.png",
        folder=Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results"
                    "/marc_cnn_sweep_fixed_u/20260425_0136_10_SGD_lr0.02_b32"),
    ),
    dict(
        label="ViT · Muon · lr=0.003 · b=128",
        out_name="ViT_Muon_lr0.003_b128.png",
        folder=Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results"
                    "/marc_vit_sweep/20260424_0142_39_Muon_lr0.003_b128_momentum-0.9"),
    ),
    dict(
        label="Text Classification Transformer · Adam · lr=3e-5 · b=8192",
        out_name="SST_Adam_lr3e-05_b8192.png",
        folder=Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results"
                    "/SST_opt_batch_sweep"
                    "/20260504_0657_16_Adam_lr3e-05_b8192_beta1-0.9_beta2-0.99"),
    ),
]

OUT_DIR = Path(__file__).resolve().parent / "paper_bimodality"
OUT_DIR.mkdir(exist_ok=True)

BINS = 60

def make_figure(run: dict):
    folder = run["folder"]
    d = np.load(folder / "projections.npz")
    track_from  = int(d["track_from"])
    track_until = int(d["track_until"])
    proj_w      = np.asarray(d["proj_w_top5"])   # (n_steps, 5)

    fig = plt.figure(figsize=(30, 14))
    outer = GridSpec(2, 1, figure=fig,
                     height_ratios=[1.3, 1.0],
                     hspace=0.38,
                     top=0.91, bottom=0.06, left=0.045, right=0.995)

    # Row 0: 3-column layout — narrow padding | wide training curve | narrow padding
    # Ratios [1, 5, 1] → training curve occupies ~71% of the row width, centered.
    gs_top = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], wspace=0,
        width_ratios=[1, 5, 1],
    )
    gs_bot = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.18)

    # Replace "lr" with η in label; this becomes the training dynamics title
    dyn_title = "Training Dynamics — " + run["label"].replace("lr=", r"$\eta$=")

    # Row 0: training curve in center column only
    ax_curve = fig.add_subplot(gs_top[0, 1])
    plot_training_curve(ax_curve, folder, track_from, track_until)
    fig.add_subplot(gs_top[0, 0]).set_visible(False)
    fig.add_subplot(gs_top[0, 2]).set_visible(False)

    # Enlarge fonts and fold run info into the title (paper-ready).
    ax_curve.set_title(dyn_title, fontsize=30, pad=10)
    ax_curve.set_xlabel("step", fontsize=24)
    ax_curve.set_ylabel(r"Batch Sharpness / $\lambda_{\max}$", fontsize=24)
    ax_curve.tick_params(labelsize=20)
    # Also resize the twin y-axis (loss) ticks + label
    for ax_twin in ax_curve.figure.axes:
        if ax_twin is not ax_curve and ax_twin.get_shared_x_axes().joined(ax_curve, ax_twin):
            ax_twin.tick_params(labelsize=20)
            ax_twin.set_ylabel(ax_twin.get_ylabel(), fontsize=24)
    # Re-draw legend with larger font using the handles already set by plot_training_curve
    leg = ax_curve.get_legend()
    if leg is not None:
        handles = leg.legend_handles
        labels  = [t.get_text() for t in leg.get_texts()]
        ax_curve.legend(handles, labels, fontsize=26, framealpha=0.7, loc="upper right")

    # Row 1: proj_w_top5 histograms
    label_fmt = r"$\langle\theta_t,\, w^{\mathrm{full}}_{%d}(t)\rangle$"
    bot_axes = []
    for k in range(TOP_K):
        ax = fig.add_subplot(gs_bot[0, k])
        col = proj_w[:, k]
        if _all_nan(col):
            ax.set_visible(False)
            continue
        plot_histogram(ax, col, label_fmt % (k + 1), BINS,
                       show_ylabel=(k == 0), color=K_COLORS[k])
        if k == 0:
            ax.set_ylabel(r"$\langle\theta_t,\, w^{\mathrm{full}}_k(t)\rangle$ Count",
                          fontsize=18)
        bot_axes.append(ax)

    # Enlarge μ/σ annotation text and x-axis ticks in each histogram
    for ax in bot_axes:
        for txt in ax.texts:
            txt.set_fontsize(18)
        ax.tick_params(axis='x', labelsize=20)

    stem = Path(run["out_name"]).stem
    for ext, kw in [(".png", {"dpi": 150}), (".pdf", {})]:
        out_path = OUT_DIR / (stem + ext)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.3, **kw)
        print(f"Saved: {out_path}")
    plt.close(fig)

for run in RUNS:
    make_figure(run)

print("Done.")
