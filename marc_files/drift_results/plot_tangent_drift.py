"""plot_tangent_drift.py — composite analysis for tangent drift + Cv per eigvec rank.

Reads `projections.npz` from one run folder and renders a single PNG + PDF showing,
for each of 7 chosen ranks (k ∈ {1, 5, 10, 25, 50, 75, 100}, 1-indexed):

  - histogram of ⟨θ_t, v_k(t)⟩  (live, warm-started basis)
  - histogram of ⟨θ_t, v_k^F⟩    (frozen at first tracked step)
  - rolling Cv(k) = Var of centered projection vs step
  - cumulative drift D_k(t) against the frozen basis (log-log of |D_k|)

Plus a Cat 3 row (histogram, Cv, drift of ⟨θ_t, v_star⟩ where v_star is the
random direction orthogonal to V_{top-K} at track_from — guaranteed Cat 3 by
construction).

Plus a training-curve row (loss + λ_max vs step) and a spectrum row (λ_k at
first tracked step vs k, log-log; with chosen ranks marked).

Usage:
    python plot_tangent_drift.py /path/to/run_folder
    python plot_tangent_drift.py /path/to/run_folder --bins 60
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 1-indexed ranks to highlight (the eigvec/eigvalue arrays in the npz are
# 0-indexed: column 0 = top-1 eigvec, column 99 = rank-100 eigvec).
CHOSEN_RANKS_1IDX = (1, 3, 5, 10, 15, 20, 30)
ROLLING_WINDOW = 20   # samples (= ROLLING_WINDOW * track_stride training steps)


def parse_run_title(folder_name: str) -> str:
    opt_match = re.search(r'_(SGD(?:-Momentum|-Nesterov)?|Adam|RMSProp|Muon)_', folder_name)
    lr_match  = re.search(r'_lr([\d.eE+-]+)', folder_name)
    b_match   = re.search(r'_b(\d+)', folder_name)
    opt = opt_match.group(1) if opt_match else '?'
    lr  = lr_match.group(1)  if lr_match  else '?'
    b   = b_match.group(1)   if b_match   else '?'
    return f'{opt}  lr={lr}  b={b}'


def _hist(ax, values, title, bins, color='steelblue'):
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        ax.set_visible(False)
        return
    try:
        ax.hist(clean, bins=bins, color=color, edgecolor='white', linewidth=0.4)
    except ValueError:
        center = float(clean.mean())
        ax.axvline(center, color=color, linewidth=2.0)
        m = max(float(clean.max() - clean.min()) * 5, abs(center) * 1e-3, 1e-6)
        ax.set_xlim(center - m, center + m)
    ax.set_title(title, fontsize=13)
    mean = float(np.nanmean(values))
    std  = float(np.nanstd(values))
    ax.axvline(mean, color='firebrick', linewidth=1.2, linestyle='--')
    ax.text(0.97, 0.95, f'μ={mean:.4g}\nσ={std:.3g}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#333',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.75))
    ax.tick_params(labelsize=10)


def _rolling_var(x, window):
    """Rolling variance over `window` samples, valid at the right edge."""
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=max(2, window // 2)).var().values


def _plot_cv(ax, steps, proj, k_label, color='steelblue'):
    """Rolling Cv(k) over time."""
    valid = ~np.isnan(proj)
    if not valid.any():
        ax.set_visible(False)
        return
    cv = _rolling_var(proj, ROLLING_WINDOW)
    ax.plot(steps, cv, color=color, linewidth=1.3)
    ax.set_yscale('log')
    ax.set_title(f'$C_v$ k={k_label}', fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)


def _plot_cv_band(ax, steps, proj_2d, k_label, color='#2ca02c'):
    """Median + IQR band of Cv across m sampled tangent directions.
    proj_2d shape: (n_steps, m)."""
    m = proj_2d.shape[1]
    cv_all = np.stack(
        [_rolling_var(proj_2d[:, j], ROLLING_WINDOW) for j in range(m)],
        axis=1,
    )  # (n_steps, m)
    if not np.isfinite(cv_all).any():
        ax.set_visible(False)
        return
    with np.errstate(all='ignore'):
        med = np.nanmedian(cv_all, axis=1)
        q25 = np.nanpercentile(cv_all, 25, axis=1)
        q75 = np.nanpercentile(cv_all, 75, axis=1)
        q10 = np.nanpercentile(cv_all, 10, axis=1)
        q90 = np.nanpercentile(cv_all, 90, axis=1)
    ax.fill_between(steps, q10, q90, color=color, alpha=0.15, label='10–90%')
    ax.fill_between(steps, q25, q75, color=color, alpha=0.30, label='25–75%')
    ax.plot(steps, med, color=color, linewidth=1.6, label='median')
    ax.set_yscale('log')
    ax.set_title(f'$C_v$ k={k_label}  (m={m} dirs)', fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')


def _plot_drift(ax, steps, proj_frozen, k_label, color='steelblue',
                smooth_window=ROLLING_WINDOW):
    """Tangent drift along frozen basis (paper-style):
        D_k(t) = ⟨bar_θ_t, v_k^F⟩ − ⟨bar_θ_{t_0}, v_k^F⟩,
    where bar_θ_t = E[θ_s : s ∈ W_t] is the slow-mean trajectory. Following the
    paper's decomposition θ_t = bar_θ_t + δ_t (slow drift + fast oscillation),
    the tangent drift is the motion of bar_θ_t, NOT of θ_t. We approximate
    bar_θ_t by a centered rolling mean of θ_t over `smooth_window` measurements,
    which averages out δ_t. By linearity:
        ⟨bar_θ_t, v⟩ = rolling-mean of ⟨θ_t, v⟩."""
    valid = ~np.isnan(proj_frozen)
    if valid.sum() < smooth_window:
        ax.set_visible(False)
        return
    smoothed = pd.Series(proj_frozen).rolling(
        window=smooth_window, min_periods=max(2, smooth_window // 2),
        center=True,
    ).mean().values
    valid_s = ~np.isnan(smoothed)
    if valid_s.sum() < 2:
        ax.set_visible(False)
        return
    first_idx = int(np.argmax(valid_s))
    drift = smoothed - smoothed[first_idx]
    # Drop the anchor sample: drift[first_idx] is identically 0 by construction
    # and would render as 1e-30 on the log axis, stretching the y-range.
    mask = valid_s.copy()
    mask[first_idx] = False
    if mask.sum() < 2:
        ax.set_visible(False)
        return
    x = steps[mask].astype(float)
    ax.semilogy(x, np.abs(drift[mask]) + 1e-30, color=color, linewidth=1.3)
    # Reference curves anchored at the first valid sample (t0):
    #   |D| = c * (step - t0)   (systematic, linear)
    #   |D| = c * sqrt(step - t0) (random walk)
    if x.size >= 2:
        t0 = float(x[0])
        dt = x - t0
        t1 = float(dt[-1])
        d_ref = float(np.nanmax(np.abs(drift))) + 1e-30
        if t1 > 0 and d_ref > 0:
            c_lin = d_ref / t1
            c_rw  = d_ref / np.sqrt(t1)
            # Skip dt=0 endpoint: c_lin*0 → 1e-30 collapses log-y range.
            ax.semilogy(x[1:], c_lin * dt[1:], color='#888888',
                        linewidth=0.8, linestyle='--', alpha=0.7)      # ∝ Δt
            ax.semilogy(x[1:], c_rw * np.sqrt(dt[1:]), color='#888888',
                        linewidth=0.8, linestyle=':',  alpha=0.7)      # ∝ √Δt
    ax.set_title(f'|$D_k$| k={k_label}  (slow-mean)', fontsize=13)
    ax.set_xlabel('step', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which='both')


def _plot_ratio(ax, steps, proj_live, proj_frozen, k_label,
                color='#ff7f0e', smooth_window=ROLLING_WINDOW):
    """|D_k(t)| / C_v(k,t) — tangent drift per unit oscillation power.
    Numerator: rolling-mean slow drift in FROZEN basis (clean tangent drift).
    Denominator: rolling variance in LIVE basis (true oscillation power along
    the time-varying sharp direction). Matches the (frozen/live) basis split
    used by rows 4 (Cv, live) and 5 (drift, frozen)."""
    cv = _rolling_var(proj_live, smooth_window)
    smoothed = pd.Series(proj_frozen).rolling(
        window=smooth_window, min_periods=max(2, smooth_window // 2),
        center=True,
    ).mean().values
    valid_s = ~np.isnan(smoothed)
    if valid_s.sum() < 2:
        ax.set_visible(False)
        return
    first_idx = int(np.argmax(valid_s))
    drift = smoothed - smoothed[first_idx]
    eps = 1e-30
    with np.errstate(all='ignore'):
        ratio = np.abs(drift) / (cv + eps)
    mask = valid_s & np.isfinite(ratio) & (cv > 0)
    # Drop the anchor: drift == 0 by construction makes the ratio collapse to 0.
    mask[first_idx] = False
    if mask.sum() < 2:
        ax.set_visible(False)
        return
    ax.semilogy(steps[mask], ratio[mask] + eps,
                color=color, linewidth=1.3)
    ax.set_title(f'|$D_k$|/$C_v$ k={k_label}', fontsize=13)
    ax.set_xlabel('step', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which='both')


def _plot_ratio_single(ax, steps, proj_single, k_label,
                       color='#ff7f0e', smooth_window=ROLLING_WINDOW):
    """Like _plot_ratio but for a single 1D projection (e.g. Cat 3 v_star,
    where live == frozen because v_star is fixed at track_from)."""
    _plot_ratio(ax, steps, proj_single, proj_single, k_label,
                color=color, smooth_window=smooth_window)


def _plot_ratio_band(ax, steps, proj_2d, k_label, color='#2ca02c',
                     smooth_window=ROLLING_WINDOW):
    """Median + IQR band of |D_k(t)|/C_v(k,t) across m sampled directions.
    proj_2d shape: (n_steps, m). For Cat 3 v_star_j (fixed at track_from),
    live and frozen are the same projection."""
    n_steps, m = proj_2d.shape
    eps = 1e-30
    ratio_all = np.full((n_steps, m), np.nan, dtype=np.float64)
    for j in range(m):
        cv = _rolling_var(proj_2d[:, j], smooth_window)
        smoothed = pd.Series(proj_2d[:, j]).rolling(
            window=smooth_window, min_periods=max(2, smooth_window // 2),
            center=True,
        ).mean().values
        valid_s = ~np.isnan(smoothed)
        if valid_s.sum() < 2:
            continue
        first_idx = int(np.argmax(valid_s))
        drift = smoothed - smoothed[first_idx]
        with np.errstate(all='ignore'):
            r = np.abs(drift) / (cv + eps)
        r[~(valid_s & np.isfinite(r) & (cv > 0))] = np.nan
        r[first_idx] = np.nan  # drift == 0 by construction at anchor
        ratio_all[:, j] = r
    if not np.isfinite(ratio_all).any():
        ax.set_visible(False)
        return
    with np.errstate(all='ignore'):
        med = np.nanmedian(ratio_all, axis=1)
        q25 = np.nanpercentile(ratio_all, 25, axis=1)
        q75 = np.nanpercentile(ratio_all, 75, axis=1)
        q10 = np.nanpercentile(ratio_all, 10, axis=1)
        q90 = np.nanpercentile(ratio_all, 90, axis=1)
    mask = np.isfinite(med)
    if mask.sum() < 2:
        ax.set_visible(False)
        return
    x = steps[mask].astype(float)
    ax.fill_between(x, q10[mask] + eps, q90[mask] + eps,
                    color=color, alpha=0.15, label='10–90%')
    ax.fill_between(x, q25[mask] + eps, q75[mask] + eps,
                    color=color, alpha=0.30, label='25–75%')
    ax.semilogy(x, med[mask] + eps, color=color, linewidth=1.6, label='median')
    ax.set_title(f'|$D_k$|/$C_v$ k={k_label}  (m={m} dirs)', fontsize=13)
    ax.set_xlabel('step', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='best')


def _plot_drift_band(ax, steps, proj_2d_frozen, k_label, color='#2ca02c',
                     smooth_window=ROLLING_WINDOW):
    """Median + IQR band of |D_k(t)| across m sampled tangent directions.
    proj_2d_frozen shape: (n_steps, m). Each column treated as one realization
    of the slow-mean tangent drift (see _plot_drift for the per-direction formula)."""
    n_steps, m = proj_2d_frozen.shape
    if n_steps < smooth_window:
        ax.set_visible(False)
        return
    # Per-column smoothed trajectory minus its first valid value.
    drift_all = np.full((n_steps, m), np.nan, dtype=np.float64)
    for j in range(m):
        smoothed = pd.Series(proj_2d_frozen[:, j]).rolling(
            window=smooth_window, min_periods=max(2, smooth_window // 2),
            center=True,
        ).mean().values
        valid_s = ~np.isnan(smoothed)
        if valid_s.sum() < 2:
            continue
        first_idx = int(np.argmax(valid_s))
        col_drift = smoothed - smoothed[first_idx]
        # Anchor sample is identically 0 — drop so log-y isn't dominated by eps.
        col_drift[first_idx] = np.nan
        drift_all[:, j] = col_drift
    abs_drift = np.abs(drift_all)
    if not np.isfinite(abs_drift).any():
        ax.set_visible(False)
        return
    with np.errstate(all='ignore'):
        med = np.nanmedian(abs_drift, axis=1)
        q25 = np.nanpercentile(abs_drift, 25, axis=1)
        q75 = np.nanpercentile(abs_drift, 75, axis=1)
        q10 = np.nanpercentile(abs_drift, 10, axis=1)
        q90 = np.nanpercentile(abs_drift, 90, axis=1)
    eps = 1e-30
    mask = np.isfinite(med)
    x = steps[mask].astype(float)
    if x.size < 2:
        ax.set_visible(False)
        return
    ax.fill_between(x, q10[mask] + eps, q90[mask] + eps,
                    color=color, alpha=0.15, label='10–90%')
    ax.fill_between(x, q25[mask] + eps, q75[mask] + eps,
                    color=color, alpha=0.30, label='25–75%')
    ax.semilogy(x, med[mask] + eps, color=color, linewidth=1.6, label='median')
    # Slope-1 (systematic) and slope-1/2 (random walk) references.
    t0 = float(x[0]); dt = x - t0; t1 = float(dt[-1])
    d_ref = float(np.nanmax(q90)) + eps
    if t1 > 0 and d_ref > 0:
        c_lin = d_ref / t1
        c_rw  = d_ref / np.sqrt(t1)
        # Skip dt=0 endpoint to avoid collapsing the log-y range.
        ax.semilogy(x[1:], c_lin * dt[1:], color='#888888',
                    linewidth=0.8, linestyle='--', alpha=0.7)
        ax.semilogy(x[1:], c_rw * np.sqrt(dt[1:]), color='#888888',
                    linewidth=0.8, linestyle=':',  alpha=0.7)
    ax.set_title(f'|$D_k$| k={k_label}  (m={m} dirs, slow-mean)', fontsize=13)
    ax.set_xlabel('step', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='best')


def plot_training_curve(ax, run_folder: Path, track_from: int, track_until: int):
    results_path = run_folder / 'data' / 'results.txt'
    if not results_path.exists():
        results_path = run_folder / 'results.txt'
    if not results_path.exists():
        ax.set_title('No results.txt', fontsize=13)
        return
    df = pd.read_csv(results_path, comment='#')
    ax_loss = ax.twinx()
    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        ax_loss.semilogy(loss['step'], loss['full_loss'],
                         color='#aaaaaa', linewidth=1.2, label='loss')
        ax_loss.set_ylabel('loss (log)', fontsize=11, color='#555')
    lm = df[['step', 'lmax']].dropna()
    if not lm.empty:
        ax.plot(lm['step'], lm['lmax'], color='#1f77b4', linewidth=1.4,
                label=r'$\lambda_{max}$')
    bs = df[['step', 'batch_sharpness']].dropna()
    if not bs.empty:
        ax.plot(bs['step'], bs['batch_sharpness'], color='#2ca02c',
                linewidth=1.4, label='batch sharp.')
    ax.axvspan(track_from, track_until, alpha=0.1, color='orange', label='tracked')
    ax.set_xlabel('step', fontsize=12)
    ax.set_ylabel(r'$\lambda_{max}$ / batch sharp.', fontsize=12)
    ax.set_title('Training Dynamics', fontsize=14)
    ax.tick_params(labelsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')


def plot_spectrum(ax, lambda_top5_arr, chosen_ranks_1idx, eta_2over=100.0):
    """λ_k at first tracked step vs k (log-log)."""
    if lambda_top5_arr is None or lambda_top5_arr.size == 0:
        ax.set_visible(False)
        return
    lam0 = lambda_top5_arr[0]
    valid = ~np.isnan(lam0)
    if not valid.any():
        ax.set_visible(False)
        return
    k = np.arange(1, len(lam0) + 1)
    ax.loglog(k[valid], np.abs(lam0[valid]) + 1e-12,
              marker='.', linewidth=1.0, color='#1f77b4')
    ax.axhline(eta_2over, color='r', linewidth=1.0, linestyle='--',
               label=f'2/η = {eta_2over:g}')
    for r in chosen_ranks_1idx:
        if r <= len(lam0) and valid[r - 1]:
            ax.scatter([r], [abs(lam0[r - 1]) + 1e-12],
                       color='orange', s=40, zorder=5, edgecolor='black',
                       linewidth=0.6)
    ax.set_xlabel('rank k (1-indexed)', fontsize=12)
    ax.set_ylabel(r'$\lambda_k$ at $t_0$', fontsize=12)
    ax.set_title('Hessian spectrum (top-K) at track_from', fontsize=14)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_folder', type=Path)
    parser.add_argument('--bins', type=int, default=60)
    parser.add_argument('--out', type=Path, default=None,
                        help='Output PNG path; PDF written alongside.')
    args = parser.parse_args()

    run_folder: Path = args.run_folder
    npz_path = run_folder / 'data' / 'projections.npz'
    if not npz_path.exists():
        npz_path = run_folder / 'projections.npz'
    if not npz_path.exists():
        raise SystemExit(f"No projections.npz in {run_folder} or {run_folder/'data'}")
    data = np.load(npz_path)

    steps        = data['steps']
    track_from   = int(data['track_from'])
    track_until  = int(data['track_until'])
    proj_w       = data['proj_w_top5']             # [n, K] live
    proj_w_fixed = data['proj_w_fixed_top5']       # [n, K] frozen
    lambda_topk  = data['lambda_top5']             # [n, K] eigenvalues
    proj_cat3    = data['proj_cat3'] if 'proj_cat3' in data.files else None
    top_k        = int(data['top_k']) if 'top_k' in data.files else proj_w.shape[1]

    chosen_idx_0 = [r - 1 for r in CHOSEN_RANKS_1IDX if r <= top_k]
    chosen_lbl   = [str(r) for r in CHOSEN_RANKS_1IDX if r <= top_k]
    n_picks      = len(chosen_idx_0)
    if n_picks == 0:
        raise SystemExit(f"None of the chosen ranks fit in top-K={top_k}")

    title = parse_run_title(run_folder.name)

    # Figure layout:
    # rows = 7 + (1 if proj_cat3 else 0)
    # row 0: training curve (full width) | row 1: spectrum (full width)
    # row 2: n_picks histograms (live)   | row 3: n_picks histograms (frozen)
    # row 4: n_picks Cv panels           | row 5: n_picks drift panels
    # row 6: n_picks |D_k|/Cv ratio panels
    # row 7 (if cat3): 4 panels (hist, Cv, drift, ratio of v_star) across full width
    has_cat3 = proj_cat3 is not None
    n_rows = 7 + (1 if has_cat3 else 0)

    fig = plt.figure(figsize=(3.6 * n_picks, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_picks, hspace=0.55, wspace=0.35)

    # Row 0: training curve spanning all cols
    ax_train = fig.add_subplot(gs[0, :])
    plot_training_curve(ax_train, run_folder, track_from, track_until)

    # Row 1: spectrum spanning all cols
    ax_spec = fig.add_subplot(gs[1, :])
    plot_spectrum(ax_spec, lambda_topk, CHOSEN_RANKS_1IDX)

    # Rows 2 & 3: live + frozen histograms
    for col, (idx0, lbl) in enumerate(zip(chosen_idx_0, chosen_lbl)):
        _hist(fig.add_subplot(gs[2, col]),
              proj_w[:, idx0], f'live k={lbl}', args.bins, color='steelblue')
        _hist(fig.add_subplot(gs[3, col]),
              proj_w_fixed[:, idx0], f'frozen k={lbl}', args.bins, color='#9467bd')

    # Row 4: Cv per rank (live basis)
    for col, (idx0, lbl) in enumerate(zip(chosen_idx_0, chosen_lbl)):
        _plot_cv(fig.add_subplot(gs[4, col]), steps, proj_w[:, idx0], lbl)

    # Row 5: cumulative drift against frozen basis
    for col, (idx0, lbl) in enumerate(zip(chosen_idx_0, chosen_lbl)):
        _plot_drift(fig.add_subplot(gs[5, col]), steps, proj_w_fixed[:, idx0], lbl,
                    color='#9467bd')

    # Row 6: ratio |D_k(t)| / Cv(k,t) per rank (frozen for D, live for Cv)
    for col, (idx0, lbl) in enumerate(zip(chosen_idx_0, chosen_lbl)):
        _plot_ratio(fig.add_subplot(gs[6, col]), steps,
                    proj_w[:, idx0], proj_w_fixed[:, idx0], lbl,
                    color='#ff7f0e')

    # Row 7 (optional): Cat 3 panel — split width into 4 sub-axes
    if has_cat3:
        # Normalize shape: legacy runs saved 1D (m=1); new runs save 2D (n, m).
        if proj_cat3.ndim == 1:
            cat3_2d = proj_cat3[:, None]
        else:
            cat3_2d = proj_cat3
        m_cat3 = cat3_2d.shape[1]
        sub = gs[7, :].subgridspec(1, 4, wspace=0.30)
        # Histogram: pool all m × n_steps values into one distribution.
        _hist(fig.add_subplot(sub[0]), cat3_2d.flatten(),
              rf'Cat 3 ($v_\star$): pooled hist  (m={m_cat3})',
              args.bins, color='#2ca02c')
        if m_cat3 > 1:
            _plot_cv_band(fig.add_subplot(sub[1]), steps, cat3_2d, r'$\star$',
                          color='#2ca02c')
            _plot_drift_band(fig.add_subplot(sub[2]), steps, cat3_2d, r'$\star$',
                             color='#2ca02c')
            _plot_ratio_band(fig.add_subplot(sub[3]), steps, cat3_2d, r'$\star$',
                             color='#2ca02c')
        else:
            _plot_cv(fig.add_subplot(sub[1]), steps, cat3_2d[:, 0], r'$\star$',
                     color='#2ca02c')
            _plot_drift(fig.add_subplot(sub[2]), steps, cat3_2d[:, 0], r'$\star$',
                        color='#2ca02c')
            _plot_ratio_single(fig.add_subplot(sub[3]), steps, cat3_2d[:, 0],
                               r'$\star$', color='#2ca02c')

    fig.suptitle(f'Tangent drift / Cv  —  {title}', fontsize=16, y=0.998)

    out_png = args.out if args.out else run_folder / f'tangent_drift_{run_folder.name}.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == '__main__':
    main()
