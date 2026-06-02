"""
plot_histograms.py — Combined training curve + projection histograms + cos-sim.

Produces up to two PNGs per run:

  histograms_<run>.png          (always)
      Row 0: training dynamics (lmax, batch sharpness, loss)
      Row 1: 3 basic histograms (proj_g_full, proj_g, proj_h)
      Row 2: 5 histograms of ⟨θ_t, w_k^full(t)⟩ for k=1..5 (full-H per-step)
      Row 3: 5 histograms of ⟨θ_t, w_k^batch(t)⟩ for k=1..5 (batch-H per-step)
      Row 4: 5 histograms of ⟨θ_t, w_k^fixed⟩ for k=1..5 (fixed at track_from)
      Row 5: cosine-similarity curves |⟨w_k(t-1), w_k(t)⟩| for k=1..5

  histograms_<run>_precond.png  (Adam / RMSProp only)
      Row 0: training dynamics (same)
      Row 1: 5 histograms of ⟨θ_t, w̃_k^full(t)⟩
      Row 2: 5 histograms of ⟨θ_t, w̃_k^batch(t)⟩
      Row 3: 5 histograms of ⟨θ_t, w̃_k^fixed⟩
      Row 4: cosine-similarity curves (precond)

Usage:
    python plot_histograms.py /path/to/run_folder
    python plot_histograms.py /path/to/run_folder --bins 80
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


TOP_K = 5
COS_SIM_BINS = 60  # fixed-edge histogram on [0, 1] for cosine similarity

K_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def parse_run_title(folder_name: str) -> str:
    opt_match  = re.search(r'_(SGD(?:-Momentum|-Nesterov)?|Adam|RMSProp|Muon)_', folder_name)
    lr_match   = re.search(r'_lr([\d.eE+-]+)', folder_name)
    b_match    = re.search(r'_b(\d+)', folder_name)
    beta_match = re.search(r'_beta-([\d.eE+-]+)', folder_name)

    opt  = opt_match.group(1) if opt_match else '?'
    lr   = lr_match.group(1)  if lr_match  else '?'
    b    = b_match.group(1)   if b_match   else '?'
    beta = beta_match.group(1) if beta_match else None

    title = f'{opt}  lr={lr}  b={b}'
    if beta:
        title += f'  β={beta}'
    return title


def _all_nan(arr):
    return np.all(np.isnan(arr))


def plot_training_curve(ax, run_folder: Path, track_from: int, track_until: int,
                        lambda_top1_precond=None, steps_precond=None):
    """lmax (blue), batch_sharpness (green), loss (grey right axis).
    Optionally overlay λ_max(D^{-1/2}HD^{-1/2}) in the tracking window (k=1 column)."""
    results_path = run_folder / 'data' / 'results.txt'
    if not results_path.exists():
        results_path = run_folder / 'results.txt'
    if not results_path.exists():
        ax.set_title('No results.txt', fontsize=15)
        return

    df = pd.read_csv(results_path, comment='#')

    ax_loss = ax.twinx()
    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        ax_loss.semilogy(loss['step'], loss['full_loss'],
                         color='#aaaaaa', linewidth=1.4, label='loss', zorder=1)
        ax_loss.set_ylabel('Loss (log)', fontsize=14, color='#555555')
        ax_loss.tick_params(labelsize=14, colors='#555555')
        ax_loss.yaxis.label.set_color('#555555')

    lm = df[['step', 'lmax']].dropna()
    if not lm.empty:
        ax.plot(lm['step'], lm['lmax'], color='#1f77b4', linewidth=1.6,
                label=r'$\lambda_{max}$', zorder=3)

    bs = df[['step', 'batch_sharpness']].dropna()
    if not bs.empty:
        ax.plot(bs['step'], bs['batch_sharpness'], color='#2ca02c', linewidth=1.6,
                label='batch sharp.', zorder=3)

    if lambda_top1_precond is not None and steps_precond is not None:
        valid = ~np.isnan(lambda_top1_precond)
        if valid.any():
            ax.scatter(steps_precond[valid], lambda_top1_precond[valid],
                       color='#9467bd', s=8, label=r'$\lambda_{max}(\tilde{H})$', zorder=4)

    ax.axvspan(track_from, track_until, alpha=0.10, color='orange', label='tracked')

    ax.set_xlabel('step', fontsize=18)
    ax.set_ylabel(r'Batch Sharpness / $\lambda_{max}$', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_title('Training Dynamics', fontsize=26)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=17, framealpha=0.7,
              loc='upper right')


def plot_histogram(ax, values, title, bins, show_ylabel=False, color='steelblue',
                   show_mean=True):
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        ax.set_visible(False)
        return
    try:
        ax.hist(clean, bins=bins, color=color, edgecolor='white', linewidth=0.5)
    except ValueError:
        # Full-batch / deterministic dynamics can collapse a projection to a constant
        # (or to a range too narrow for `bins` finite-sized bins). Render a delta-spike
        # at the mean and a sensible xlim so the spike is visible.
        center = float(clean.mean())
        ax.axvline(center, color=color, linewidth=3.0)
        margin = max(float(clean.max() - clean.min()) * 5, abs(center) * 1e-3, 1e-6)
        ax.set_xlim(center - margin, center + margin)
    ax.set_title(title, fontsize=26)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=18)
    ax.tick_params(labelsize=16)

    if show_mean:
        mean = float(np.nanmean(values))
        std  = float(np.nanstd(values))
        ax.axvline(mean, color='firebrick', linewidth=1.6, linestyle='--')
        ax.text(0.97, 0.95, f'μ={mean:.5g}\nσ={std:.3g}', transform=ax.transAxes,
                ha='right', va='top', fontsize=13, color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.75))


def plot_cos_sim(ax, steps, cos_arr, title):
    """cos_arr: [n_steps, 5]. Row 0 expected NaN."""
    if cos_arr is None or cos_arr.size == 0 or _all_nan(cos_arr):
        ax.set_visible(False)
        return
    for k in range(TOP_K):
        y = cos_arr[:, k]
        valid = ~np.isnan(y)
        if not valid.any():
            continue
        ax.plot(steps[valid], y[valid], color=K_COLORS[k], linewidth=1.5,
                label=f'k={k+1}')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('step', fontsize=18)
    ax.set_ylabel(r'$|\langle w_k(t{-}1),\, w_k(t)\rangle|$', fontsize=18)
    ax.set_title(title, fontsize=26)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=17, framealpha=0.7, loc='lower right', ncol=5)
    ax.grid(True, alpha=0.3)


def _hist_row(fig, gs, row, steps_2d, labels, bins, row_label, hide_when_all_nan=True):
    """Place 5 histograms in one GridSpec row. steps_2d has shape [n_steps, 5]."""
    if steps_2d is None or (hide_when_all_nan and _all_nan(steps_2d)):
        return False
    for k in range(TOP_K):
        ax = fig.add_subplot(gs[row, k])
        plot_histogram(ax, steps_2d[:, k], labels[k], bins,
                       show_ylabel=(k == 0))
    return True


def _add_row_label(fig, gs, row, text):
    """Add a left-margin label spanning the row."""
    ax = fig.add_subplot(gs[row, :])
    ax.axis('off')
    ax.text(-0.015, 0.5, text, transform=ax.transAxes,
            ha='right', va='center', fontsize=14, fontweight='bold',
            rotation=0)


def make_main_png(run_folder, data, out_path, bins, has_precond, run_title):
    steps       = data['steps']
    track_from  = int(data['track_from'])
    track_until = int(data['track_until'])
    n_steps     = len(steps)

    lambda_precond_top1 = None  # intentionally not overlaid on the main training-dynamics plot

    # 5 rows × 5 cols. Row 0 merges training curve (cols 0–1) + 3 basic hists (cols 2–4).
    # Use nested GridSpecs so the top row can have a wider wspace (room for twin-y loss label)
    # while the other rows stay tightly packed.
    nrows, ncols = 5, 5
    fig = plt.figure(figsize=(30, 26))
    outer_gs = GridSpec(nrows, 1, figure=fig,
                        height_ratios=[1.3, 1.0, 1.0, 1.0, 1.0],
                        hspace=0.34,
                        top=0.935, bottom=0.025, left=0.045, right=0.995)
    gs_top = GridSpecFromSubplotSpec(
        1, ncols, subplot_spec=outer_gs[0], wspace=0.34,
        width_ratios=[0.92, 0.92, 1.05, 1.05, 1.05],
    )
    gs_rows = [GridSpecFromSubplotSpec(1, ncols, subplot_spec=outer_gs[r], wspace=0.18)
               for r in range(1, nrows)]

    fig.suptitle(run_title, fontsize=38, fontweight='bold', y=0.985)

    # Row 0a: training curve (cols 0–1)
    ax_curve = fig.add_subplot(gs_top[0, 0:2])
    plot_training_curve(
        ax_curve, run_folder, track_from, track_until,
        lambda_top1_precond=lambda_precond_top1,
        steps_precond=steps if has_precond else None,
    )

    # Row 0b: 3 basic histograms in cols 2, 3, 4
    basic = [
        ('proj_g_full', r'$\langle\theta_t,\, \mathbb{E}[g_t]\rangle$'),
        ('proj_g',      r'$\langle\theta_t,\, g_t\rangle$'),
        ('proj_h',      r'$\langle\theta_t,\, h_t\rangle$'),
    ]
    for i, (key, label) in enumerate(basic):
        ax = fig.add_subplot(gs_top[0, 2 + i])
        if key in data:
            plot_histogram(ax, data[key], label, bins, show_ylabel=(i == 0))
        else:
            ax.set_visible(False)

    # Rows 1–3: top-5 histograms per family
    families = [
        (1, 'proj_w_top5',       r'$\langle\theta_t,\, w^{\mathrm{full}}_{%d}(t)\rangle$',
            'Full H top-5 Eigenvec Count'),
        (2, 'proj_wb_top5',      r'$\langle\theta_t,\, w^{\mathrm{batch}}_{%d}(t)\rangle$',
            'Batch H top-5 Eigenvec Count'),
        (3, 'proj_w_fixed_top5', r'$\langle\theta_t,\, w^{\mathrm{fixed}}_{%d}\rangle$',
            'Full H top-5 Fixed Eigenvec Count'),
    ]
    for row, key, label_fmt, row_name in families:
        arr = data[key] if key in data else None
        if arr is None:
            continue
        for k in range(TOP_K):
            ax = fig.add_subplot(gs_rows[row - 1][0, k])
            if _all_nan(arr[:, k]):
                ax.set_visible(False)
                continue
            plot_histogram(ax, arr[:, k], label_fmt % (k + 1), bins,
                           show_ylabel=(k == 0), color=K_COLORS[k])
            if key == 'proj_w_fixed_top5':
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            if k == 0:
                ax.set_ylabel(row_name, fontsize=18)

    # Row 4: cosine similarity histograms (5 cells, one per k)
    cos = data['cos_sim_full_top5'] if 'cos_sim_full_top5' in data else None
    if cos is not None and not _all_nan(cos):
        for k in range(TOP_K):
            ax = fig.add_subplot(gs_rows[3][0, k])
            if _all_nan(cos[:, k]):
                ax.set_visible(False)
                continue
            label = r'$|\langle w^{\mathrm{full}}_{%d}(t{-}1),\, w^{\mathrm{full}}_{%d}(t)\rangle|$' % (k + 1, k + 1)
            col_clean = cos[:, k][~np.isnan(cos[:, k])]
            edges = np.linspace(0.0, 1.0, COS_SIM_BINS + 1)
            clipped = np.clip(col_clean, 0.0, 1.0)
            ax.hist(clipped, bins=edges, color=K_COLORS[k],
                    edgecolor='white', linewidth=0.5)
            ax.set_xlim(0.0, 1.0)
            ax.set_title(label, fontsize=26)
            ax.tick_params(labelsize=16)
            if col_clean.size:
                mean_k = float(col_clean.mean())
                std_k = float(col_clean.std())
                ax.text(0.05, 0.95,
                        f'μ={mean_k:.6f}\nσ={std_k:.2e}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=12, color='#333',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='none', alpha=0.85))
            if k == 0:
                ax.set_ylabel('Cosine Similarity Counts', fontsize=18)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    print(f"Saved main figure to {out_path}")


def make_precond_png(run_folder, data, out_path, bins, run_title):
    steps       = data['steps']
    track_from  = int(data['track_from'])
    track_until = int(data['track_until'])
    n_steps     = len(steps)

    lambda_precond_top1 = None
    if 'lambda_precond_top5' in data:
        lp = data['lambda_precond_top5']
        if lp.size > 0:
            lambda_precond_top1 = lp[:, 0]

    nrows, ncols = 5, 5
    fig = plt.figure(figsize=(30, 22))
    gs = GridSpec(nrows, ncols, figure=fig,
                  height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0],
                  hspace=0.55, wspace=0.35)

    fig.suptitle(
        f'{run_title}  |  PRECONDITIONED  |  steps {track_from}–{track_until}  ({n_steps} recorded)',
        fontsize=18,
    )

    # Row 0: training curve
    ax_curve = fig.add_subplot(gs[0, :])
    plot_training_curve(
        ax_curve, run_folder, track_from, track_until,
        lambda_top1_precond=lambda_precond_top1,
        steps_precond=steps,
    )

    families = [
        (1, 'proj_w_precond_top5',       r'$\langle\theta_t,\, \tilde{w}^{\mathrm{full}}_{%d}(t)\rangle$',
            'Precond. Full-H top-5 (per-step)'),
        (2, 'proj_wb_precond_top5',      r'$\langle\theta_t,\, \tilde{w}^{\mathrm{batch}}_{%d}(t)\rangle$',
            'Precond. Batch-H top-5 (per-step)'),
        (3, 'proj_w_precond_fixed_top5', r'$\langle\theta_t,\, \tilde{w}^{\mathrm{fixed}}_{%d}\rangle$',
            'Precond. Full-H top-5 (fixed at track_from)'),
    ]
    for row, key, label_fmt, row_name in families:
        arr = data[key] if key in data else None
        if arr is None:
            continue
        for k in range(TOP_K):
            ax = fig.add_subplot(gs[row, k])
            if _all_nan(arr[:, k]):
                ax.set_visible(False)
                continue
            plot_histogram(ax, arr[:, k], label_fmt % (k + 1), bins,
                           show_ylabel=(k == 0), color=K_COLORS[k])
            if k == 0:
                ax.set_ylabel(f'{row_name}\nCount', fontsize=18)

    # Row 4: cosine similarity histograms (precond) — 5 cells, one per k
    cos = data['cos_sim_precond_top5'] if 'cos_sim_precond_top5' in data else None
    if cos is not None and not _all_nan(cos):
        for k in range(TOP_K):
            ax = fig.add_subplot(gs[4, k])
            if _all_nan(cos[:, k]):
                ax.set_visible(False)
                continue
            label = r'$|\langle \tilde{w}^{\mathrm{full}}_{%d}(t{-}1),\, \tilde{w}^{\mathrm{full}}_{%d}(t)\rangle|$' % (k + 1, k + 1)
            plot_histogram(ax, cos[:, k], label, COS_SIM_BINS,
                           show_ylabel=(k == 0), color=K_COLORS[k],
                           show_mean=False)
            ax.ticklabel_format(useOffset=False, axis='x')
            col_clean = cos[:, k][~np.isnan(cos[:, k])]
            if col_clean.size:
                mean_k = float(col_clean.mean())
                std_k = float(col_clean.std())
                ax.text(0.05, 0.95,
                        f'μ={mean_k:.6f}\nσ={std_k:.2e}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=12, color='#333',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='none', alpha=0.85))
            if k == 0:
                ax.set_ylabel('Cosine Similarity Counts', fontsize=18)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    print(f"Saved precond figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curve + projection histograms.')
    parser.add_argument('run_folder', type=str)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--out', type=str, default=None,
                        help='Main PNG path; precond PNG is derived by appending "_precond".')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    npz_path = run_folder / 'data' / 'projections.npz'
    if not npz_path.exists():
        npz_path = run_folder / 'projections.npz'
    if not npz_path.exists():
        print(f"Error: projections.npz not found in {run_folder} or {run_folder/'data'}.", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)

    has_precond = (
        'proj_w_precond_top5' in data and not _all_nan(data['proj_w_precond_top5'])
    )

    run_title = parse_run_title(run_folder.name)

    if args.out:
        main_out = Path(args.out)
        precond_out = main_out.with_name(main_out.stem + '_precond' + main_out.suffix)
    else:
        safe_title = run_title.replace(' ', '_').replace('=', '').replace('/', '-')
        main_out = run_folder / f'histograms_{safe_title}.png'
        precond_out = run_folder / f'histograms_{safe_title}_precond.png'

    make_main_png(run_folder, data, main_out, args.bins, has_precond, run_title)

    if has_precond:
        make_precond_png(run_folder, data, precond_out, args.bins, run_title)


if __name__ == '__main__':
    main()
