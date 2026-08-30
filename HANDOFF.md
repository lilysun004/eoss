# HANDOFF — start here

**Project.** Empirical study of the edge of stochastic stability: what a stochastic optimizer's
settled top sharpness (κ = η·λ_B) is, and whether there's a universal, path-computable scalar that
equals a fixed value at the edge for **any** optimizer/batch/β. Branch `lily`, worktree
`.claude/worktrees/gbs-search`. Lean setup: CIFAR-10, `mlp_s` (789k params), num_data=2048, MSE, CPU.
Code in `experiments/`, results (gitignored) in `results/`. Run env:
`export DATASETS=/Users/xq/Desktop/moonshot/eoss/datasets EOSS_SKIP_CHECKSUM=1` and the venv
`/Users/xq/Desktop/moonshot/eoss/.venv`.

## Established results (solid — don't re-derive)
1. **GBS = 2 is the at-the-edge signature** (GBS = E_B[sᵀH_Bs/(−gᵀs)]): all five optimizers at the
   coherent edge (GOLD sweep 2026-08-30, health-masked — heavy-ball large-batch GBS≠2 readings were
   float32-dead-run artifacts). Small-batch: EVERY optimizer sits below 2; ceiling law
   κ_B = min(coherent edge, C(B,arch)·(1−β)) — `GOLD_RESULTS.md`, `analysis/MEMORY_EDGE_LAW.md`.
2. **"The buffer moves the house, not the weather."** In a paired SGD+SGDM sweep at matched batch,
   *every* fluctuation statistic (catapult clustering, rate, excursion-size distribution) is
   optimizer-independent (set by batch noise); only the **operating-point position** (κ/GBS) tracks
   the buffer. Position is buffer-set; weather is not.
3. **The sub-edge point is a REGULATED ATTRACTOR, not a phase.** λ is restored to it from both above
   and below (park-vs-attractor test, `park_test.py`). There is **no force-free / "metastable phase."**
4. **Position = a saturating min, not a ratio.** Free-exponent regression rejected R = memory/τ_rot
   (τ_rot exponent ≈0); the (β,batch) table shows large-batch κ\* = **2(1+β) exactly**. So
   κ\* = min( 2(1+β) [β-edge, GBS=2, stability, universal], reach(batch) [loss-geometry, non-universal] ).

## THE κ_spec / κ_ms RESULTS (2026-07-12 session — COMPLETE) → read `KSPEC_RESULTS.md`
The pending κ_spec test RAN and PASSED its committed pre-registration, and the follow-on
mean-square program resolved the middle band. One-paragraph version:
**κ_spec = 2 at the coherent edge, formula-free, two optimizers, three thresholds** (SGDM b512/
b2048: 2.007 ± CV 0.006 with gain = 1/(1+β) to 3 decimals; Nesterov anchor ratio 1.000, gain =
(1+2β)/(1+β) to 3 decimals, κ_spec = 2.00; sub-edge cells all < 2, two-sided holds). The STRONG
form (every plateau spectrally marginal) is REFUTED: mixed-ω plateaued cells read 0.33/0.54/1.51
with gain ≈ 1 (buffer decoheres; measured open-loop R² +1.00 → −43 down the ladder). The
second-moment program (κ_ms, pre-registered ADDENDA 1–5) found: **no moment order gives a
universal =2 pin (Reading A dead), but the mean-square WALL is real and measured** — onset
brackets from bitwise replay checkpoints put every cell 1.15–1.35× below its wall (κ_ms_emp
1.6–2.0, → 2 exactly at the coherent end), with a noise-growing margin and one clean anomaly
(b8 β0.9 regulation budget ≈ 1.05, death≈onset). Estimator reconciliation saga + calibration
status in KSPEC_RESULTS.md / ADDENDA 4–5.
- **Wide-grid collapse figure is UNLOCKED** (committed gate passed) — not yet run.
- **Open quantitative target: the margin law** — margin(c*−1) vs curvature-noise CV(h)²
  (archived r=0.895 law is the candidate), plus recomputing the κ_ms column under one
  calibrated construction (est-(i) with its measured +0.3 bias band at mixed cells, empirical
  onsets as ground truth).

## Doc map
- **`KSPEC_RESULTS.md`** — the 2026-07-12 session's full results (κ_spec PASS, κ_ms walls, margins, budgets). **Read after this file.**
- **`KSPEC_PREREG_ANNOTATIONS.md`** — all five pre-registration addenda (chain of custody).
- **`KSPEC_DESIGN.md`** — the original κ_spec spec (now executed): hypothesis, banked result, pre-registered gate + pass
  criteria, runner-patch spec (done), estimator plan, cells. Historical.
- **`LESSONS.md`** — methodology / good practices earned the hard way this project. Read before running
  anything; several rules here would have saved weeks.
- **`SUMMARY.md`** — full detailed historical record (all six parts). ARCHIVE. Parts on the
  marginal/metastable "phase" (Part IV/VI phase language) were **RETRACTED** — see result #3 above;
  read it for context on *how* claims were tested/killed, not for current claims.
- Memory (`~/.claude/.../memory/gbs-edge-dichotomy.md`) — the running one-file record of findings.

## Key code
`experiments/slow_sweep.py` (dense per-step runner, now with signed projections),
`slow_sweep_driver.py` (grid + liveness bisect), `p1_isoR.py` (contour-driver pattern to reuse),
`park_test.py` / `transplant.py` (the attractor tests), `phase_analysis.py` (passive stats, frozen
decision rule), `utils/optimizer.py` (add Nesterov here), `utils/measure.py` (HVP, eigvecs).
