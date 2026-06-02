"""plot_curvature_failure.py — central-flow failure-mode diagnostic plots.

Reads `curvature_segment.npz` from one run folder (plus `results.txt` for the
training curve overlay) and renders a single PNG + PDF showing:

  (1) Training curve (loss + λ_max vs step) with the tracking window shaded.
  (2) Example segment scans S(α) = u_mid^T H(α w_t + (1-α) w_{t+1}) u_mid for
      12 representative measurement steps, with Taylor_1 and Taylor_2
      approximations from finite differences at α=0.5 ± 0.02 overlaid.
      Direct visual reproduction of Cohen et al. Fig 29.
  (3) Example along-u scans S(β) = u_t^T H(w_t + β u_t) u_t for the same 12 steps.
  (4) Aggregate signed deviation Δ(α) = S(α) − Taylor_1(α) across all steps
      (median + IQR shading) vs α.
  (5) Aggregate signed deviation Δ(β) = S(β) − λ_w_t across all steps
      (median + IQR shading) vs β/|δ_t|.
  (6) Time evolution: signed endpoint deviation Δ(α=0) and Δ(α=1) vs step.

Usage:
    python plot_curvature_failure.py /path/to/run_folder
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


FD_H = 0.02   # finite-difference offset used at training time for α=0.5 ± h


def parse_run_title(folder_name: str) -> str:
    opt_match = re.search(r'_(SGD(?:-Momentum|-Nesterov)?|Adam|RMSProp|Muon)_', folder_name)
    lr_match  = re.search(r'_lr([\d.eE+-]+)', folder_name)
    b_match   = re.search(r'_b(\d+)', folder_name)
    opt = opt_match.group(1) if opt_match else '?'
    lr  = lr_match.group(1)  if lr_match  else '?'
    b   = b_match.group(1)   if b_match   else '?'
    return f'{opt}  lr={lr}  b={b}'


def load_training_curve(run_folder: Path):
    """Load results.txt; return a pandas DataFrame (or None)."""
    csv_path = run_folder / 'data' / 'results.txt'
    if not csv_path.exists():
        csv_path = run_folder / 'results.txt'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, comment='#')
    if 'step' not in df.columns:
        df['step'] = np.arange(len(df))
    return df


def taylor_from_scan(alphas: np.ndarray, S: np.ndarray, h: float = FD_H):
    """Compute Taylor_1 and Taylor_2 approximations of S(α) around α=0.5.

    Uses finite-difference estimates of D = S'(0.5) and D2 = S''(0.5) from the
    points at α ∈ {0.5-h, 0.5, 0.5+h} (which the training-time scan grid is
    constructed to include).

    Returns (D, D2, T1(α), T2(α)).
    """
    i_mid = int(np.argmin(np.abs(alphas - 0.5)))
    i_lo  = int(np.argmin(np.abs(alphas - (0.5 - h))))
    i_hi  = int(np.argmin(np.abs(alphas - (0.5 + h))))
    S_mid = S[i_mid]
    S_lo  = S[i_lo]
    S_hi  = S[i_hi]
    actual_h = (alphas[i_hi] - alphas[i_lo]) / 2.0
    if actual_h <= 0:
        return np.nan, np.nan, np.full_like(alphas, np.nan), np.full_like(alphas, np.nan)
    D  = (S_hi - S_lo) / (2.0 * actual_h)
    D2 = (S_hi + S_lo - 2.0 * S_mid) / (actual_h ** 2)
    T1 = S_mid + (alphas - 0.5) * D
    T2 = S_mid + (alphas - 0.5) * D + 0.5 * (alphas - 0.5) ** 2 * D2
    return D, D2, T1, T2


def make_plot(run_folder: Path):
    run_folder = Path(run_folder).resolve()
    # Prefer the `data/` subfolder convention; fall back to the run-folder root
    # for older runs where npz files lived next to the rendered PNGs.
    curv_path = run_folder / 'data' / 'curvature_segment.npz'
    if not curv_path.exists():
        curv_path = run_folder / 'curvature_segment.npz'
    if not curv_path.exists():
        raise SystemExit(
            f"No curvature_segment.npz found in {run_folder} or {run_folder}/data/"
        )

    d = np.load(curv_path)
    alphas      = np.asarray(d['alphas'], dtype=np.float64)
    steps       = np.asarray(d['steps'])
    S_seg       = np.asarray(d['curv_segment'])     # [n_meas, n_alphas]
    S_u         = np.asarray(d['curv_along_u'])     # [n_meas, n_betas] (may be empty)
    betas       = np.asarray(d['betas'])            # [n_meas, n_betas]
    lambda_mid  = np.asarray(d['lambda_mid'])
    lambda_w_t  = np.asarray(d['lambda_w_t'])
    step_proj_u_t = np.asarray(d['step_proj_u_t'])
    # Along-STEP scan (per-batch Hessian along the unit step direction ĥ=h_t/‖h_t‖).
    # Present only for runs made after the along-step feature was added.
    S_step      = np.asarray(d['curv_along_step']) if 'curv_along_step' in d.files \
                  else np.empty((0, 0))
    betas_step  = np.asarray(d['betas_step']) if 'betas_step' in d.files \
                  else np.empty((0, 0))
    uHu_step    = np.asarray(d['uHu_step']) if 'uHu_step' in d.files \
                  else np.empty((0,))
    track_from  = int(d['track_from']) if 'track_from' in d.files else int(steps.min())
    track_until = int(d['track_until']) if 'track_until' in d.files else int(steps.max())

    # Sort by α so the line plots are monotone left→right.
    alpha_order = np.argsort(alphas)
    alphas_s    = alphas[alpha_order]
    S_seg_s     = S_seg[:, alpha_order]

    n_meas = len(steps)
    has_b = S_u.size > 0 and betas.size > 0
    has_step = S_step.size > 0 and betas_step.size > 0

    # Pick 12 representative measurement steps to show as panels.
    n_panels = min(12, n_meas)
    if n_meas >= n_panels:
        panel_idx = np.linspace(0, n_meas - 1, n_panels, dtype=int)
    else:
        panel_idx = np.arange(n_meas)

    # Compute Taylor approximations and signed deviations for every meas step.
    Ds   = np.zeros(n_meas)
    D2s  = np.zeros(n_meas)
    T1   = np.zeros_like(S_seg_s)
    T2   = np.zeros_like(S_seg_s)
    for i in range(n_meas):
        Ds[i], D2s[i], T1[i], T2[i] = taylor_from_scan(alphas_s, S_seg_s[i])
    Delta_alpha = S_seg_s - T1   # signed deviation from first-order Taylor

    # ---------- Figure layout ----------
    # Manual spacing — constrained_layout fights the nested subgridspecs and
    # causes the small-panel titles to overlap the row below.
    #
    # Rows are allocated dynamically (a running index) so the optional along-u
    # and along-step blocks can be added/dropped without re-deriving hardcoded
    # offsets. Each entry is (height_ratio,). Order:
    #   training | seg-panels | [along-u panels] | [along-step panels]
    #   | agg Δ(α) | [agg Δ(β) along-u] | [overlay step-vs-u] | time-evolution
    height_ratios = [1.0, 2.6]                      # training, seg-panels
    if has_b:    height_ratios.append(2.6)          # along-u panels
    if has_step: height_ratios.append(2.6)          # along-step panels
    height_ratios.append(1.1)                       # agg Δ(α)
    if has_b:    height_ratios.append(1.1)          # agg Δ(β)
    if has_step: height_ratios.append(1.3)          # overlay step-vs-u
    height_ratios.append(1.1)                       # time-evolution
    n_rows = len(height_ratios)

    fig = plt.figure(figsize=(16, 3.6 * n_rows), constrained_layout=False)
    gs = fig.add_gridspec(
        n_rows, 4,
        height_ratios=height_ratios,
        hspace=0.45, wspace=0.3,
        left=0.06, right=0.96, top=0.95, bottom=0.05,
    )
    _row = [0]
    def next_row():
        r = _row[0]; _row[0] += 1; return r

    title = parse_run_title(run_folder.name)
    fig.suptitle(f'Curvature failure-mode diagnostic — {title}\n{run_folder.name}',
                 fontsize=12, y=1.00)

    # --- Row 0: training curve overlay (matches plot_histograms.py style) ---
    # Left axis: λ_max (blue) + batch_sharpness (green). Right axis (twinx,
    # log scale): full_loss (gray). Tracking window shaded orange.
    ax_train = fig.add_subplot(gs[next_row(), :])
    df = load_training_curve(run_folder)
    if df is not None:
        ax_loss = ax_train.twinx()
        if 'full_loss' in df.columns:
            loss = df[['step', 'full_loss']].dropna()
            if not loss.empty:
                ax_loss.semilogy(loss['step'], loss['full_loss'],
                                 color='#aaaaaa', linewidth=1.4,
                                 label='loss', zorder=1)
                ax_loss.set_ylabel('Loss (log)', fontsize=11, color='#555555')
                ax_loss.tick_params(labelsize=9, colors='#555555')
                ax_loss.yaxis.label.set_color('#555555')

        if 'lmax' in df.columns:
            lm = df[['step', 'lmax']].dropna()
            if not lm.empty:
                ax_train.plot(lm['step'], lm['lmax'], color='#1f77b4',
                              linewidth=1.6, label=r'$\lambda_{max}$', zorder=2)

        if 'batch_sharpness' in df.columns:
            bs = df[['step', 'batch_sharpness']].dropna()
            if not bs.empty:
                ax_train.plot(bs['step'], bs['batch_sharpness'],
                              color='#2ca02c', linewidth=1.6,
                              label='batch sharpness', zorder=2)

        ax_train.axvspan(track_from, track_until,
                         alpha=0.10, color='orange', label='tracked')

        ax_train.set_xlabel('step', fontsize=11)
        ax_train.set_ylabel(r'Batch Sharpness / $\lambda_{max}$', fontsize=11)
        ax_train.tick_params(labelsize=9)

        lines1, labels1 = ax_train.get_legend_handles_labels()
        lines2, labels2 = ax_loss.get_legend_handles_labels()
        ax_train.legend(lines1 + lines2, labels1 + labels2,
                        loc='upper left', fontsize=8)
    ax_train.set_title('Training curve + tracking window', fontsize=10)

    # --- Example segment scans (a). 12 small subplots in a 4×3 grid. ---
    row1 = gs[next_row(), :].subgridspec(3, 4, hspace=0.65, wspace=0.32)
    for k, idx in enumerate(panel_idx):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(row1[r, c])
        ax.plot(alphas_s, S_seg_s[idx], 'o-', color='C3', ms=3, lw=1.0, label='S(α)')
        ax.plot(alphas_s, T1[idx], '--', color='gray', lw=1.0, label='Taylor₁')
        ax.plot(alphas_s, T2[idx], '-', color='black', lw=0.8, alpha=0.5, label='Taylor₂')
        ax.axvline(0.5, color='k', alpha=0.2, lw=0.5)
        ax.set_title(f'step={steps[idx]}  λ_mid={lambda_mid[idx]:.2f}', fontsize=8)
        ax.set_xlabel('α', fontsize=8)
        ax.set_ylabel('u_mid^T H u_mid', fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')

    # --- Along-u scans (b): full-batch top-eigvec direction. 12 panels. ---
    if has_b:
        row2 = gs[next_row(), :].subgridspec(3, 4, hspace=0.65, wspace=0.32)
        for k, idx in enumerate(panel_idx):
            r, c = divmod(k, 4)
            ax = fig.add_subplot(row2[r, c])
            b_grid = betas[idx]
            order = np.argsort(b_grid)
            ax.plot(b_grid[order], S_u[idx][order], 'o-', color='C0', ms=3, lw=1.0)
            ax.axvline(0.0, color='k', alpha=0.2, lw=0.5)
            ax.axhline(lambda_w_t[idx], color='C3', alpha=0.5, lw=0.8,
                       label=f'λ_w_t={lambda_w_t[idx]:.2f}')
            ax.set_title(f'step={steps[idx]}  |δ_t|={abs(step_proj_u_t[idx]):.3g}',
                         fontsize=8)
            ax.set_xlabel('β', fontsize=8)
            ax.set_ylabel('u_t^T H u_t', fontsize=8)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=7, loc='best')

    # --- Along-step scans: PER-BATCH Hessian along the unit step direction ĥ.
    # This is the stochastic instability direction; β=0 value ≈ batch_sharpness.
    # 12 panels. ---
    if has_step:
        row_step = gs[next_row(), :].subgridspec(3, 4, hspace=0.65, wspace=0.32)
        for k, idx in enumerate(panel_idx):
            r, c = divmod(k, 4)
            ax = fig.add_subplot(row_step[r, c])
            b_grid = betas_step[idx]
            order = np.argsort(b_grid)
            ax.plot(b_grid[order], S_step[idx][order], 'o-', color='C1', ms=3, lw=1.0)
            ax.axvline(0.0, color='k', alpha=0.2, lw=0.5)
            ax.axhline(uHu_step[idx], color='C3', alpha=0.5, lw=0.8,
                       label=f'ĥᵀH_Bĥ={uHu_step[idx]:.2f}')
            ax.set_title(f'step={steps[idx]}', fontsize=8)
            ax.set_xlabel('β (along ĥ)', fontsize=8)
            ax.set_ylabel('ĥ^T H_B ĥ', fontsize=8)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=7, loc='best')

    # --- Aggregate signed deviation Δ(α) across all measurement steps ---
    ax_agg_a = fig.add_subplot(gs[next_row(), :])
    med = np.median(Delta_alpha, axis=0)
    q25 = np.quantile(Delta_alpha, 0.25, axis=0)
    q75 = np.quantile(Delta_alpha, 0.75, axis=0)
    ax_agg_a.fill_between(alphas_s, q25, q75, color='C3', alpha=0.25, label='IQR')
    ax_agg_a.plot(alphas_s, med, '-', color='C3', lw=1.5, label='median')
    ax_agg_a.axhline(0.0, color='k', lw=0.5, alpha=0.5)
    ax_agg_a.axvline(0.5, color='k', alpha=0.2, lw=0.5)
    ax_agg_a.set_xlabel('α')
    ax_agg_a.set_ylabel('S(α) − Taylor₁(α)')
    ax_agg_a.set_title(f'(a) Aggregate signed deviation across {n_meas} measurement steps '
                       f'— median > 0: super-quadratic; < 0: sub-quadratic',
                       fontsize=10)
    ax_agg_a.legend(fontsize=8)

    # --- Aggregate Δ(β) along u_t across all measurement steps ---
    if has_b:
        ax_agg_b = fig.add_subplot(gs[next_row(), :])
        # Normalize β by per-step scale to overlay across measurements
        scales = np.abs(step_proj_u_t.reshape(-1, 1))
        scales = np.where(scales > 1e-12, scales, 1e-12)
        beta_norm = betas / scales            # [n_meas, n_betas]
        dev_b = S_u - lambda_w_t[:, None]     # [n_meas, n_betas]
        # Plot per-step traces (gray) + median curve. β grids are identical
        # (post-normalization), so we can index across columns directly.
        ref_beta = beta_norm[0]  # all rows share this grid post-normalization
        for i in range(n_meas):
            order = np.argsort(beta_norm[i])
            ax_agg_b.plot(beta_norm[i][order], dev_b[i][order],
                          color='gray', alpha=0.15, lw=0.5)
        # Aggregate median by sorting and averaging across steps
        order_ref = np.argsort(ref_beta)
        med_b = np.median(dev_b[:, order_ref], axis=0)
        ax_agg_b.plot(ref_beta[order_ref], med_b, '-', color='C0', lw=2.0,
                      label='median across steps')
        ax_agg_b.axhline(0.0, color='k', lw=0.5, alpha=0.5)
        ax_agg_b.axvline(0.0, color='k', lw=0.5, alpha=0.5)
        ax_agg_b.set_xlabel('β / |δ_t|')
        ax_agg_b.set_ylabel('u_t^T H(w_t + β u_t) u_t − λ_w_t')
        ax_agg_b.set_title('(b) Aggregate signed deviation along u_t', fontsize=10)
        ax_agg_b.legend(fontsize=8)

    # --- KEY PANEL: overlay of median curvature profile along the per-batch
    # step direction (ĥ, stochastic) vs along the full-batch top eigvec (u).
    # The two coincide at large batch (EoS) and decouple by a large factor at
    # small batch (EoSS) — the stochastic instability lives off u. β grids are
    # symmetric linspaces, so per-row normalization collapses to a shared axis. ---
    if has_step:
        ax_ov = fig.add_subplot(gs[next_row(), :])
        x_norm = np.linspace(-1.0, 1.0, S_step.shape[1])
        step_med = np.nanmedian(S_step, axis=0)
        step_lo  = np.nanquantile(S_step, 0.25, axis=0)
        step_hi  = np.nanquantile(S_step, 0.75, axis=0)
        ax_ov.fill_between(x_norm, step_lo, step_hi, color='C1', alpha=0.20)
        ax_ov.plot(x_norm, step_med, 'o-', color='C1', lw=1.8, ms=4,
                   label='along ĥ (per-batch H, stochastic)')
        if has_b:
            xu = np.linspace(-1.0, 1.0, S_u.shape[1])
            u_med = np.nanmedian(S_u, axis=0)
            u_lo  = np.nanquantile(S_u, 0.25, axis=0)
            u_hi  = np.nanquantile(S_u, 0.75, axis=0)
            ax_ov.fill_between(xu, u_lo, u_hi, color='C0', alpha=0.20)
            ax_ov.plot(xu, u_med, 's-', color='C0', lw=1.8, ms=4,
                       label='along u (full-batch top eigvec)')
        c_step = float(step_med[len(step_med) // 2])
        c_u = float(np.nanmedian(lambda_w_t))
        ratio = c_step / c_u if c_u else float('nan')
        ax_ov.axvline(0.0, color='k', lw=0.5, alpha=0.4)
        ax_ov.set_xlabel('normalized scan position  (0 = iterate)')
        ax_ov.set_ylabel('curvature  (eigenvalue units)')
        ax_ov.set_title(
            f'Stochastic vs full-batch curvature — median across {n_meas} steps  '
            f'(center ĥᵀH_Bĥ / λ_max = {c_step:.1f} / {c_u:.1f} = {ratio:.1f}×)',
            fontsize=10)
        ax_ov.legend(fontsize=9, loc='best')

    # --- Final row: Time evolution of endpoint deviation ---
    ax_time = fig.add_subplot(gs[next_row(), :])
    idx_left  = int(np.argmin(np.abs(alphas_s - 0.0)))
    idx_right = int(np.argmin(np.abs(alphas_s - 1.0)))
    ax_time.plot(steps, Delta_alpha[:, idx_left], 'o-', ms=3, lw=1.0,
                 color='C0', label='Δ(α=0)  ⇒ at w_{t+1}')
    ax_time.plot(steps, Delta_alpha[:, idx_right], 's-', ms=3, lw=1.0,
                 color='C2', label='Δ(α=1)  ⇒ at w_t')
    ax_time.axhline(0.0, color='k', lw=0.5, alpha=0.5)
    ax_time.set_xlabel('step')
    ax_time.set_ylabel('S − Taylor₁ at endpoint')
    ax_time.set_title('Time evolution of segment-endpoint deviation', fontsize=10)
    ax_time.legend(fontsize=8)

    # Save PNG in run folder (PDFs intentionally not emitted; the curvature
    # results folder layout keeps only PNGs at the run-folder root).
    out_png = run_folder / 'curvature_failure.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_png}")

    # Brief textual summary.
    print()
    print(f"# measurement steps:     {n_meas}")
    print(f"# alphas (segment scan): {len(alphas)}")
    if has_b:
        print(f"# betas (along-u scan):  {betas.shape[1]}")
    print(f"Median Δ(α=0) across steps:  {np.median(Delta_alpha[:, idx_left]):+.4g}")
    print(f"Median Δ(α=1) across steps:  {np.median(Delta_alpha[:, idx_right]):+.4g}")
    print(f"Mean   λ_w_t  across steps:  {np.mean(lambda_w_t):+.4g}")
    print(f"Mean   λ_mid  across steps:  {np.mean(lambda_mid):+.4g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_folder', type=str,
                        help='Path to a run folder containing curvature_segment.npz')
    args = parser.parse_args()
    make_plot(Path(args.run_folder))


if __name__ == '__main__':
    main()
