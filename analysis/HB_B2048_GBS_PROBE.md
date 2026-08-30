# Heavy-ball b2048: why GBS_med ≈ 0.33 while κ_spec ≈ 2 (probe, 2026-08-30)

**Question.** SGD-Momentum β0.9 b2048 (mlp_s `L_b2048_beta0.9_s0..4`; mlp_l `A_/A2_b2048_beta0.9_s*`)
reads κ_spec ≈ 2.0 but window-median GBS ≈ 0.33 (mlp_l 0.04–0.46), whereas SGD/Nesterov/Adam/Muon at
b2048 read GBS ≈ 2.0 and every small-batch cell has GBS ≈ κ_spec. Bug, divergence, or physics?

## Verdict

**mlp_s (the five flagship seeds): UNHEALTHY RUN — numerically dead for the last ~2/3 of the analysis
window. Not a GBS bug, not a physics difference.** In the healthy part of the window GBS = 1.84–2.00 on
every seed (top mode carries 98.6 % of the descent, applied step = intended step). At step ≈ 6000–6400
the loss reaches ~2e-9 and the parameter update drops below float32 resolution (per-parameter step
3e-11 vs half-ulp 2e-10 at the median |θ| = 3.5e-3); from there the applied step is 17–40 % of the
intended step, the period-2 coherence is lost, the bulk (rounding-noise) gradient dominates −gᵀs, and
per-step GBS collapses to ~0.3. The window median (0.33) is the median of the dead steps. κ_spec still
reads 2 only because its PSD-weighted integral is dominated by the high-amplitude healthy segment —
its own half-split diagnostic already flagged this (`kappa_spec_h1` ≈ 2.0, `kappa_spec_h2` ≈ 4.0 on all
five seeds) but the stationarity gate only watches κ_raw = lr·λ_B, which stays flat (3.79 → 3.81)
through the death. The liveness preflight (1500-step probe) could not see a death at step 6000.

**mlp_l (`A_`/`A2_b2048_beta0.9`): BUDGET ARTIFACT (pre-plateau), already flagged.** The run is still
descending (loss 4e-1 → 9e-6 over 30k steps, `dxu/su` = 1.000 throughout, no numerical death). GBS ≪ 2
there is *genuine*: −gᵀs is dominated by bulk descent (top-mode share of −gᵀs 0.3–3 %), i.e. the steps
are still loss-decreasing steps, not marginal ones. Top-mode-only GBS = 1.5 ≈ κ_spec 1.58–1.76 — both
instruments say sub-edge; they differ only in how much bulk descent they weigh. No contradiction.

**The old claim "momentum at large batch GBS ≈ 2.00" (SUMMARY.md/HANDOFF.md) is correct for healthy
runs and comes from the 3000-step slow_sweep cells**; the 20k/40k-step cells in the same sweep already
showed the same death pattern (GBS 0.03–0.7 at loss 1e-11–1e-13). See §5.

## 1. Health in the analysis window [4000, end]

Per-2000-step blocks, `L_b2048_beta0.9_s0` (mlp_s, lr 0.0065, 16 000 steps, status done, not diverged):

| block | loss med | ‖g‖ | ‖s‖ | GBS med | Σ sᵀHs / Σ(−gᵀs) | top-mode share of −gᵀs | top-only GBS | su sign-flip frac | dxu/su |
|---|---|---|---|---|---|---|---|---|---|
| [0,2k) | 1.7e-3 | 2.9e-1 | 2.3e-3 | 1.934 | 1.960 | 0.594 | 1.998 | 0.92 | 1.000 |
| [2k,4k) | 2.8e-6 | 2.8e-2 | 1.0e-4 | 1.991 | 1.993 | 0.936 | 2.002 | 0.99 | 1.000 |
| **[4k,6k)** | 3.4e-8 | 3.1e-3 | 1.1e-5 | **1.999** | 1.997 | 0.986 | 2.005 | 1.00 | 1.000 |
| [6k,8k) | 4.9e-10 | 6.7e-6 | 2.5e-7 | 0.162 | 1.999 | 0.006 | 2.011 | 0.78 | 0.894 |
| [8k,10k) | 4.1e-11 | 1.9e-6 | 8.2e-8 | 0.193 | 0.184 | 0.001 | 2.162 | 0.45 | 0.402 |
| [10k,12k) | 1.1e-11 | 1.1e-6 | 4.8e-8 | 0.285 | 0.287 | 0.001 | 2.315 | 0.34 | 0.251 |
| [12k,14k) | 4.5e-12 | 8.2e-7 | 3.6e-8 | 0.361 | 0.369 | 0.001 | 2.450 | 0.31 | 0.201 |
| [14k,16k) | 2.4e-12 | 6.9e-7 | 3.1e-8 | 0.429 | 0.435 | 0.001 | 2.477 | 0.29 | 0.173 |

`dxu` = (θ_{t+1} − θ_t)·u measured from the parameters after `opt.step()`; `su` = s·u with s the step the
optimizer wrapper says it will apply. They agree to 1.000 on every healthy cell and here until step
~6000; afterwards the applied step is a fraction of the intended one — the float32 parameters cannot
represent the update. κ_raw meanwhile: 3.794 (first 1k of window) → 3.813 (last 1k) — flat.

Controls over their full windows (`frac |dxu/su−1| > 0.1`, min loss, GBS_med):
Nesterov b2048 s0/s1/s2/s4: 0.000 / ≥3.8e-10 / 2.000; s3: 0.052 / 1.9e-10 / 1.996.
Adam b2048 s0: 0.000 / 1.4e-4 / 2.054. Muon b2048 s0: 0.000 / 2.7e-5 / 2.002.
SGD b2048 (mlp_l) s0/s1: 0.000 / 1.1e-4 / 1.955, 1.951. Nest b2048 (mlp_l): 0.000 / 4.7e-5 / 1.99.
None of the optimizers that read GBS ≈ 2 ever reach the float32 floor inside the window.

## 2. Per-step GBS distribution (whole window, s0)

median 0.346, mean 0.663, q05 0.126, q25 0.225, q75 0.515, **q95 2.017**; frac(GBS<0) = 0.000;
frac(−gᵀs<0) = 0.000; no NaN; even/odd-step medians 0.345/0.347 (no parity structure). Not bimodal
in phase — the bimodality is *in time* (≈2.0 before step 6000, ≈0.3 after). The q95 ≈ 2.0 is the
healthy segment.

## 3. Top-mode vs bulk decomposition (whole window, s0)

top-mode sᵀHs / sᵀHs (med) 0.043; top-mode (−gᵀs) / (−gᵀs) (med) 0.002; **top-mode-only GBS =
λ_B·su/(−gu) = 2.012**; bulk-only GBS = 0.327; Σ sᵀHs / Σ(−gᵀs) over the window = 1.997 (energy-
weighted, dominated by the healthy segment). Same numbers on s1: top-only 1.987, bulk-only 0.302.
So the denominator in the dead segment is bulk-dominated (99.8 %), and that bulk is rounding-scale
gradient (‖g‖ ~ 1e-6) pushed through the heavy-ball DC gain 1/(1−β) = 10. Compare Nesterov b2048 s0:
top-mode share of −gᵀs = 1.000, top-only GBS 2.000, bulk-only 1.978; SGD b2048: 0.988 / 1.976 / 1.083;
Adam b2048: 0.847 / 1.950 / 2.026.

## 4. Per-seed summary (death step = first window step with |dxu/su − 1| > 0.1)

| cell | death step | loss at death | GBS healthy med | GBS dead med | κ_spec | κ_spec h1 | κ_spec h2 | window GBS med |
|---|---|---|---|---|---|---|---|---|
| b2048 s0 | 5976 | 3.0e-9 | **1.999** | 0.306 | 2.006 | 2.006 | 4.072 | 0.346 |
| b2048 s1 | 5971 | 3.2e-9 | 1.835 | 0.284 | 1.981 | 1.981 | 4.050 | 0.326 |
| b2048 s2 | 6194 | 2.3e-9 | 1.856 | 0.282 | 1.981 | 1.981 | 4.031 | 0.325 |
| b2048 s3 | 6441 | 1.2e-9 | **1.998** | 0.308 | 2.007 | 2.007 | 4.021 | 0.359 |
| b2048 s4 | 6052 | 2.9e-9 | 1.933 | 0.287 | 1.988 | 1.988 | 4.039 | 0.326 |
| b512 s0–s4 | 4642–6504 | 6e-10–3e-8 | 1.94–2.00 | 2.23–2.44 | 1.99–2.02 | same | 2.84–3.31 | 2.14–2.36 |
| b128 s0–s3 | 7632–9401 | 1e-8–9e-8 | 1.48–1.54 | 1.53–1.59 | 1.51/1.54 | 1.51/1.53 | 1.51/1.52 | 1.53–1.58 |
| b64 s0/s1 | 22537/10498 | 6e-9/2e-6 | 1.019/1.020 | 1.022/1.031 | 1.035/1.051 | ≈ | ≈ | 1.019/1.028 |

The same death hits heavy-ball b512 (GBS goes *up* to 2.4 there — the dead-segment value is not
universal, it is whatever the rounding-noise bulk happens to give) and, later and more mildly, b128
and b64 (which read the same before and after because their top-mode share was already small and
their κ_spec is genuinely sub-2). Only the b2048/b512 heavy-ball cells have GBS_med driven by the
dead segment. `kappa_spec_h2` ≈ 4 on all b2048 seeds is the estimator's own "second half is
garbage" signal; the stationarity gate (`kappa_drift` = 0.004) did not use it.

Float32 arithmetic: model params float32 (`experiments/long_train_grid.py::build`), n = 789 258,
median |θ| = 3.45e-3 → half-ulp 2.1e-10 per parameter. Step norm 3e-8 → 3.4e-11 per parameter, six
times below the rounding threshold (at step norm 1e-5, the healthy value, it is 1.1e-8, 50× above).

## 5. Provenance of "momentum at large batch GBS ≈ 2.00"

All old SGD-Momentum b ≥ 512 runs with a `gbs` column (30 runs, `results/slow_sweep/`, `results/p1_isoR/`,
all stride 2), late-half medians:

- β0.9 b2048, **3000-step** cells: none at β0.9 (the β0.9 b2048 cells were all run 20k steps).
- β0.6 b2048, 3000 steps: lr 0.0065 → GBS 1.989 (loss 1.5e-3); lr 0.01 → 1.990 (loss 7e-4). Same lr,
  **40 000 steps** (`DEPTH_SGDM_b2048_beta0.6_*` s0/s1/s2): GBS 0.037 / 0.033 / 0.032 at loss 6–7e-11.
- β0.9 b2048, 20 000 steps: lr 0.004 → 0.231, lr 0.0065 → 0.634, lr 0.01 → 0.745, all at loss ≤ 5e-12.
- β0.9 b512, 3000 steps: lr 0.008 → 1.985 (loss 5e-8), lr 0.014 → 1.922; 20 000 steps lr 0.002 → 1.140
  (loss 5.5e-11); 5400 steps lr 0.004 → 1.746.
- β0.99 b2048, 20 000 steps, lr 0.01/0.017: 2.046 / 1.997 at loss ~3e-13 — the exception, where the
  β0.99 buffer (memory 100) apparently keeps the top-mode oscillation amplitude above the floor.

So the SUMMARY/HANDOFF sentence was read off short, healthy cells (and it is right there); the
stride-2 0.61/0.634 value the pre-registration attributed to "phase-locking" (`KSPEC_PREREG_ANNOTATIONS.md`
lines 52–55) is in fact the 20k-step lr 0.0065 cell whose late half is numerically dead — the same
mechanism as here, not aliasing.

## 6. Fresh confirmation run (appendix)

`HBCHK_b2048_beta0.9_lr0.0065`: SGD-Momentum β0.9 b2048 lr 0.0065 seed 0, mlp_s, stride 1, 4500 steps,
u0_at = 1500, run in-process via `S.run_cell` (scratch `/Users/xq/.claude/jobs/afbb0424/tmp/hb_probe/out/`,
12.4 min, status done, not diverged). Window [1500, 4500), n = 3000 — deliberately ended before the
float32 floor (loss 1.5e-4 at u0 → 8.9e-8 at the end; the ladder cells die at loss ≈ 2e-9).

- **Death step: none.** `|dxu/su − 1| > 0.05` never sustained (coordinator rule ≥100/200); `> 0.1`
  never occurs. `dxu/su` median 1.000 in every 500-step block. Healthy prefix = full window.
- **Healthy-prefix GBS: median 1.994, IQR [1.959, 2.020], mean 1.985.** Full-window GBS median 1.994
  (identical, since nothing is masked). κ_raw median 3.789 (= 2(1+β) = 3.8).
- **Top-mode vs bulk split — matches §3's healthy segment:** top-mode share of sᵀHs 0.964, of −gᵀs
  0.960 (ladder [4k,6k) block: 0.988 / 0.986); top-only GBS 2.003 (2.005); bulk-only GBS 1.866;
  Σ sᵀHs / Σ(−gᵀs) = 2.001 (1.997); |su|/|gu|/lr = 0.529 (1/(1+β) = 0.526); su sign-flip fraction 0.99.
  (Two-step GBS not available from logged scalars — see item 2 of the directive; the top/bulk split is
  the substitute and it agrees.)

500-step blocks (loss med, ‖s‖ med, GBS med, dxu/su, top-mode share of −gᵀs):

| block | loss | ‖s‖ | GBS | dxu/su | top share |
|---|---|---|---|---|---|
| [0,500) | 9.7e-2 | 1.2e-2 | 0.950 | 1.000 | 0.003 |
| [500,1000) | 4.8e-3 | 4.5e-3 | 2.011 | 1.000 | 0.389 |
| [1000,1500) | 3.9e-4 | 5.5e-4 | 1.853 | 1.000 | 0.688 |
| [1500,2000) | 1.0e-4 | 6.1e-4 | 2.009 | 1.000 | 0.980 |
| [2000,2500) | 1.6e-5 | 1.6e-4 | 1.968 | 1.000 | 0.639 |
| [2500,3000) | 5.5e-6 | 1.6e-4 | 1.997 | 1.000 | 0.967 |
| [3000,3500) | 1.6e-6 | 6.8e-5 | 1.993 | 1.000 | 0.943 |
| [3500,4000) | 4.8e-7 | 4.1e-5 | 2.002 | 1.000 | 0.980 |
| [4000,4500) | 1.7e-7 | 2.4e-5 | 1.998 | 1.000 | 0.983 |

Conclusion: a heavy-ball b2048 run measured while it is numerically alive reads GBS = 2 with the same
top-mode structure as SGD/Nesterov/Adam/Muon. The 0.33 is the dead-segment artifact of §1–§4.

## Code audit (no defect found)

- `experiments/slow_sweep.py:126-131`: gradient `g` (create_graph) at θ_t on batch idx; `gd = g.detach()`;
  `s = opt.compute_step_direction(g, params)` **before** `opt.step()`.
- `utils/optimizer.py:55-85` (`SGDMomentumOptimizer.compute_step_direction`): `s = −lr·(g + β·buf_prev)`,
  `buf_prev` read from `inner.state['momentum_buffer']` before the update. PyTorch SGD(momentum=β,
  dampening 0, nesterov False) does `buf ← β·buf + g; θ ← θ − lr·buf`, so this is exactly θ_{t+1} − θ_t.
  Nesterov (`:88-117`) uses `buf_new = β·buf + g; s = −lr(g + β·buf_new)` — matches
  `SGD(nesterov=True)`. Nothing heavy-ball-specific beyond the formula.
- `experiments/slow_sweep.py:142-146`: one HVP graph from the same loss/grads (θ_t, same batch);
  `Hs = hvp(s)`, `sHs = s·Hs`, `A = −gd·s`, at θ_t. `:174` `gbs = sHs/A`. `:213-219`: `dxu` is the applied
  step projection, measured from parameters after `opt.step()` — the independent cross-check used
  above. `:135-137` buffer read via `M.buffer_flat` (pre-update buffer); only used for `mu`/cosines.
- Consistency with κ_spec's gain: in the healthy segment |su|/|gu|/lr = 0.528–0.537 (1/(1+β) = 0.526),
  `mu/gu` = −0.522, su sign-flip fraction 1.00 — the intended step *is* −lr·g/(1+β) alternating, as it
  should be at the ω = π edge. In the dead segment |su|/|gu|/lr drifts to 0.8–1.2 and the flip fraction
  to 0.3 (bulk-dominated).
- `experiments/kspec_estimator.py:62-72, 84-86, 103-108`: window = finite `gu0` rows (from `u0_at`);
  κ_spec = λ_med × ∫|T̂| P_gu dω / ∫P_gu dω (PSD-weighted); `gbs_med` = plain median over the same rows;
  h1/h2 split computed but not gated on. `kappa_drift`/`stationary` are κ_raw-based.

## What resolves it going forward (not implemented here — scope)

1. Health mask in assembly: restrict both GBS and κ_spec to rows with |dxu/su − 1| < 0.05 (or step
   norm above the float32 floor, ‖s‖/√n ≳ 10× half-ulp), and report the masked fraction. On the
   existing five seeds this alone gives GBS 1.84–2.00 vs κ_spec 1.98–2.01.
2. Liveness gate: add "not numerically dead" (loss > ~1e-8 or dxu/su ≥ 0.9 at end of probe) and,
   for the full cell, either float64 parameters or a window that ends before the floor is reached
   (heavy-ball b2048 reaches it by step ~6000 at lr 0.0065; Nesterov never does within 16k).
3. Use `kappa_spec_h1` vs `h2` disagreement as a stationarity/validity flag alongside `kappa_drift`.

Scratch: `/Users/xq/.claude/jobs/afbb0424/tmp/hb_probe/{probe.py,subwin.py}`. No tracked code modified.
