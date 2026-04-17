"""
plot_histograms.py — Combined training curve + projection histograms.

Layout (multi-row), training curve always in cell 0:
  5 proj  (no fixed_u, no precond):  2×3
  6 proj  (fixed_u, no precond):     2×4  (1 unused)
  7 proj  (no fixed_u, precond):     2×4
  9 proj  (fixed_u, precond):        3×4  (training + 9 hist, last 2 unused)

Usage:
    python plot_histograms.py /path/to/run_folder
    python plot_histograms.py /path/to/run_folder --bins 80 --out my_histograms.png
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


PROJECTIONS_BASE = [
    ('proj_g_full',  r'$\langle\theta_t,\, \mathbb{E}[g_t]\rangle$', 'Full-batch gradient'),
    ('proj_g',       r'$\langle\theta_t,\, g_t\rangle$',              'Mini-batch gradient'),
    ('proj_h',       r'$\langle\theta_t,\, h_t\rangle$',              r'Step $\Delta\theta$'),
    ('proj_w_fixed', r'$\langle\theta_t,\, u_{\mathrm{from}}\rangle$', 'Fixed full Hessian eigvec'),
    ('proj_w',       r'$\langle\theta_t,\, w_t\rangle$',              'Full Hessian eigvec'),
    ('proj_wb',      r'$\langle\theta_t,\, w^b_t\rangle$',            'Batch Hessian eigvec'),
]
PROJECTIONS_PRECOND = [
    ('proj_w_precond_fixed', r'$\langle\theta_t,\, \tilde{u}_{\mathrm{from}}\rangle$', r'Fixed precond. full eigvec'),
    ('proj_w_precond',       r'$\langle\theta_t,\, \tilde{w}_t\rangle$',               r'Precond. full eigvec'),
    ('proj_wb_precond',      r'$\langle\theta_t,\, \tilde{w}^b_t\rangle$',             r'Precond. batch eigvec'),
]


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
                        lambda_precond=None, steps_precond=None):
    """Plot lmax (blue), batch_sharpness (green), loss (grey right axis).
    Optionally overlay λ_max(D^{-1/2}HD^{-1/2}) in purple in the tracking window.
    """
    results_path = run_folder / 'results.txt'
    if not results_path.exists():
        ax.set_title('No results.txt', fontsize=10)
        return

    df = pd.read_csv(results_path, comment='#')

    # Loss on right axis (log scale, light grey)
    ax_loss = ax.twinx()
    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        ax_loss.semilogy(loss['step'], loss['full_loss'],
                         color='#cccccc', linewidth=0.8, label='loss', zorder=1)
        ax_loss.set_ylabel('Loss (log)', fontsize=7, color='#999999')
        ax_loss.tick_params(labelsize=6, colors='#999999')
        ax_loss.yaxis.label.set_color('#999999')

    # lmax (blue) and batch sharpness (green) on left axis
    lm = df[['step', 'lmax']].dropna()
    if not lm.empty:
        ax.plot(lm['step'], lm['lmax'], color='#1f77b4', linewidth=0.9,
                label=r'$\lambda_{max}$', zorder=3)

    bs = df[['step', 'batch_sharpness']].dropna()
    if not bs.empty:
        ax.plot(bs['step'], bs['batch_sharpness'], color='#2ca02c', linewidth=0.9,
                label='batch sharp.', zorder=3)

    # Preconditioned λ_max in purple (only in tracking window)
    if lambda_precond is not None and steps_precond is not None:
        valid = ~np.isnan(lambda_precond)
        if valid.any():
            ax.scatter(steps_precond[valid], lambda_precond[valid],
                       color='#9467bd', s=2, label=r'$\lambda_{max}(\tilde{H})$', zorder=4)

    # Shade tracking window
    ax.axvspan(track_from, track_until, alpha=0.08, color='orange', label='tracked')

    ax.set_xlabel('step', fontsize=7)
    ax.set_ylabel(r'Sharpness / $\lambda_{max}$', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title('Training dynamics', fontsize=9)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, framealpha=0.6,
              loc='upper right')


def plot_histogram(ax, values, title, subtitle, bins, is_first_col):
    if _all_nan(values):
        ax.set_visible(False)
        return
    ax.hist(values[~np.isnan(values)], bins=bins,
            color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(subtitle, fontsize=7)
    if is_first_col:
        ax.set_ylabel('Count', fontsize=7)
    ax.tick_params(labelsize=6)

    mean = float(np.nanmean(values))
    std  = float(np.nanstd(values))
    ax.axvline(mean, color='firebrick', linewidth=1.0, linestyle='--',
               label=f'μ={mean:.3g}')
    ax.legend(fontsize=6, framealpha=0.6)
    ax.text(0.97, 0.95, f'σ={std:.3g}', transform=ax.transAxes,
            ha='right', va='top', fontsize=6, color='#444')


def main():
    parser = argparse.ArgumentParser(description='Plot training curve + projection histograms.')
    parser.add_argument('run_folder', type=str)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    npz_path = run_folder / 'projections.npz'

    if not npz_path.exists():
        print(f"Error: {npz_path} not found.", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)
    steps      = data['steps']
    track_from = int(data['track_from'])
    track_until = int(data['track_until'])
    n_steps    = len(steps)

    # Determine which projections to show (all-NaN slots are hidden by plot_histogram)
    has_precond = (
        'proj_w_precond' in data and not _all_nan(data['proj_w_precond'])
    )
    projections = PROJECTIONS_BASE + (PROJECTIONS_PRECOND if has_precond else [])

    # Layout: choose grid based on total cells needed (1 training + n_hist histograms)
    n_hist = len(projections)
    if n_hist <= 5:
        nrows, ncols = 2, 3    # 1 + 5 → 2×3 (last cell unused)
    elif n_hist <= 7:
        nrows, ncols = 2, 4    # 1 + 6 or 7 → 2×4
    else:
        nrows, ncols = 3, 4    # 1 + 8 or 9 → 3×4 (last 2 unused)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes_flat = axes.flatten()

    run_title = parse_run_title(run_folder.name)
    fig.suptitle(
        f'{run_title}  |  steps {track_from}–{track_until}  ({n_steps} recorded)',
        fontsize=10,
    )

    # First subplot: training curve
    lambda_precond = data['lambda_precond'] if 'lambda_precond' in data else None
    plot_training_curve(
        axes_flat[0], run_folder, track_from, track_until,
        lambda_precond=lambda_precond if has_precond else None,
        steps_precond=steps if has_precond else None,
    )

    # Histogram subplots
    for i, (key, title, subtitle) in enumerate(projections):
        ax = axes_flat[i + 1]
        values = data[key] if key in data else np.full(n_steps, np.nan)
        is_first_col = (i + 1) % ncols == 0 or i == 0
        plot_histogram(ax, values, title, subtitle, args.bins, is_first_col=(i == 0))

    # Hide any unused axes
    for j in range(n_hist + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        safe_title = run_title.replace(' ', '_').replace('=', '').replace('/', '-')
        out_path = run_folder / f'histograms_{safe_title}.png'

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
