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
from matplotlib.gridspec import GridSpec


TOP_K = 5

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
    results_path = run_folder / 'results.txt'
    if not results_path.exists():
        ax.set_title('No results.txt', fontsize=10)
        return

    df = pd.read_csv(results_path, comment='#')

    ax_loss = ax.twinx()
    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        ax_loss.semilogy(loss['step'], loss['full_loss'],
                         color='#cccccc', linewidth=0.8, label='loss', zorder=1)
        ax_loss.set_ylabel('Loss (log)', fontsize=8, color='#999999')
        ax_loss.tick_params(labelsize=7, colors='#999999')
        ax_loss.yaxis.label.set_color('#999999')

    lm = df[['step', 'lmax']].dropna()
    if not lm.empty:
        ax.plot(lm['step'], lm['lmax'], color='#1f77b4', linewidth=0.9,
                label=r'$\lambda_{max}$', zorder=3)

    bs = df[['step', 'batch_sharpness']].dropna()
    if not bs.empty:
        ax.plot(bs['step'], bs['batch_sharpness'], color='#2ca02c', linewidth=0.9,
                label='batch sharp.', zorder=3)

    if lambda_top1_precond is not None and steps_precond is not None:
        valid = ~np.isnan(lambda_top1_precond)
        if valid.any():
            ax.scatter(steps_precond[valid], lambda_top1_precond[valid],
                       color='#9467bd', s=3, label=r'$\lambda_{max}(\tilde{H})$', zorder=4)

    ax.axvspan(track_from, track_until, alpha=0.08, color='orange', label='tracked')

    ax.set_xlabel('step', fontsize=8)
    ax.set_ylabel(r'Sharpness / $\lambda_{max}$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title('Training dynamics', fontsize=10)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, framealpha=0.6,
              loc='upper right')


def plot_histogram(ax, values, title, bins, show_ylabel=False):
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        ax.set_visible(False)
        return
    ax.hist(clean, bins=bins, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_title(title, fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=7)
    ax.tick_params(labelsize=6)

    mean = float(np.nanmean(values))
    std  = float(np.nanstd(values))
    ax.axvline(mean, color='firebrick', linewidth=1.0, linestyle='--')
    ax.text(0.97, 0.95, f'μ={mean:.3g}\nσ={std:.3g}', transform=ax.transAxes,
            ha='right', va='top', fontsize=6, color='#444',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.6))


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
        ax.plot(steps[valid], y[valid], color=K_COLORS[k], linewidth=0.9,
                label=f'k={k+1}')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('step', fontsize=8)
    ax.set_ylabel(r'$|\langle w_k(t{-}1),\, w_k(t)\rangle|$', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.6, loc='lower right', ncol=5)
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
            ha='right', va='center', fontsize=10, fontweight='bold',
            rotation=0)


def make_main_png(run_folder, data, out_path, bins, has_precond, run_title):
    steps       = data['steps']
    track_from  = int(data['track_from'])
    track_until = int(data['track_until'])
    n_steps     = len(steps)

    lambda_precond_top1 = None
    if has_precond and 'lambda_precond_top5' in data:
        lp = data['lambda_precond_top5']
        if lp.size > 0:
            lambda_precond_top1 = lp[:, 0]

    # 6 rows × 5 cols. Row 0 is training curve (spans all cols); row 1 has only 3 hists.
    nrows, ncols = 6, 5
    fig = plt.figure(figsize=(26, 22))
    gs = GridSpec(nrows, ncols, figure=fig,
                  height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0, 1.0],
                  hspace=0.55, wspace=0.35)

    fig.suptitle(
        f'{run_title}  |  steps {track_from}–{track_until}  ({n_steps} recorded)',
        fontsize=12,
    )

    # Row 0: training curve (spans all 5 cols)
    ax_curve = fig.add_subplot(gs[0, :])
    plot_training_curve(
        ax_curve, run_folder, track_from, track_until,
        lambda_top1_precond=lambda_precond_top1,
        steps_precond=steps if has_precond else None,
    )

    # Row 1: 3 basic histograms in cols 0, 1, 2 (cols 3, 4 empty)
    basic = [
        ('proj_g_full', r'$\langle\theta_t,\, \mathbb{E}[g_t]\rangle$'),
        ('proj_g',      r'$\langle\theta_t,\, g_t\rangle$'),
        ('proj_h',      r'$\langle\theta_t,\, h_t\rangle$'),
    ]
    for i, (key, label) in enumerate(basic):
        ax = fig.add_subplot(gs[1, i])
        if key in data:
            plot_histogram(ax, data[key], label, bins, show_ylabel=(i == 0))
        else:
            ax.set_visible(False)

    # Rows 2–4: top-5 histograms per family
    families = [
        (2, 'proj_w_top5',       r'$\langle\theta_t,\, w^{\mathrm{full}}_{%d}(t)\rangle$',
            'Full-H top-5 (per-step)'),
        (3, 'proj_wb_top5',      r'$\langle\theta_t,\, w^{\mathrm{batch}}_{%d}(t)\rangle$',
            'Batch-H top-5 (per-step)'),
        (4, 'proj_w_fixed_top5', r'$\langle\theta_t,\, w^{\mathrm{fixed}}_{%d}\rangle$',
            'Full-H top-5 (fixed at track_from)'),
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
            title = f'{row_name}  —  k={k+1}' if k == 0 else f'k={k+1}'
            plot_histogram(ax, arr[:, k], label_fmt % (k + 1), bins,
                           show_ylabel=(k == 0))
            if k == 0:
                ax.set_ylabel(f'{row_name}\nCount', fontsize=8)

    # Row 5: cosine similarity (span all cols)
    ax_cos = fig.add_subplot(gs[5, :])
    cos = data['cos_sim_full_top5'] if 'cos_sim_full_top5' in data else None
    plot_cos_sim(ax_cos, steps, cos,
                 'Cosine similarity of full-H top-5 eigenvectors across consecutive tracked steps')

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved main PNG to {out_path}")


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
    fig = plt.figure(figsize=(26, 19))
    gs = GridSpec(nrows, ncols, figure=fig,
                  height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0],
                  hspace=0.55, wspace=0.35)

    fig.suptitle(
        f'{run_title}  |  PRECONDITIONED  |  steps {track_from}–{track_until}  ({n_steps} recorded)',
        fontsize=12,
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
                           show_ylabel=(k == 0))
            if k == 0:
                ax.set_ylabel(f'{row_name}\nCount', fontsize=8)

    # Row 4: cosine similarity (precond)
    ax_cos = fig.add_subplot(gs[4, :])
    cos = data['cos_sim_precond_top5'] if 'cos_sim_precond_top5' in data else None
    plot_cos_sim(ax_cos, steps, cos,
                 'Cosine similarity of preconditioned full-H top-5 eigenvectors across consecutive tracked steps')

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved precond PNG to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curve + projection histograms.')
    parser.add_argument('run_folder', type=str)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--out', type=str, default=None,
                        help='Main PNG path; precond PNG is derived by appending "_precond".')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    npz_path = run_folder / 'projections.npz'

    if not npz_path.exists():
        print(f"Error: {npz_path} not found.", file=sys.stderr)
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
