"""
plot_histograms.py — Plot histograms of <theta_t, v> projections.

Usage:
    python plot_histograms.py /path/to/run_folder
    python plot_histograms.py /path/to/run_folder --bins 80 --out my_histograms.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECTIONS = [
    ('proj_g_full', r'$\langle\theta_t,\, \mathbb{E}[g_t]\rangle$', 'Full-batch gradient'),
    ('proj_g',      r'$\langle\theta_t,\, g_t\rangle$',             'Mini-batch gradient'),
    ('proj_h',      r'$\langle\theta_t,\, h_t\rangle$',             r'Step $\Delta\theta$'),
    ('proj_w',      r'$\langle\theta_t,\, w_t\rangle$',             'Full Hessian eigvec'),
    ('proj_wb',     r'$\langle\theta_t,\, w^b_t\rangle$',           'Batch Hessian eigvec'),
]


def main():
    parser = argparse.ArgumentParser(description='Plot projection histograms.')
    parser.add_argument('run_folder', type=str,
                        help='Path to the run folder containing projections.npz')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output filename (default: histograms.png inside run_folder)')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    npz_path = run_folder / 'projections.npz'

    if not npz_path.exists():
        print(f"Error: {npz_path} not found.", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)

    steps = data['steps']
    track_from = int(data['track_from'])
    track_until = int(data['track_until'])
    n_steps = len(steps)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        f'Projection histograms  |  steps {track_from}–{track_until}  ({n_steps} recorded)',
        fontsize=12,
    )

    for ax, (key, title, subtitle) in zip(axes, PROJECTIONS):
        values = data[key]
        ax.hist(values, bins=args.bins, color='steelblue', edgecolor='white', linewidth=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(subtitle, fontsize=9)
        ax.set_ylabel('Count' if ax is axes[0] else '')
        ax.tick_params(labelsize=8)

        # Annotate mean and std
        mean, std = float(np.mean(values)), float(np.std(values))
        ax.axvline(mean, color='firebrick', linewidth=1.2, linestyle='--', label=f'mean={mean:.3g}')
        ax.legend(fontsize=7, framealpha=0.6)
        ax.text(0.97, 0.95, f'std={std:.3g}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='#444')

    plt.tight_layout()

    out_path = Path(args.out) if args.out else run_folder / 'histograms.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram to {out_path}")


if __name__ == '__main__':
    main()
