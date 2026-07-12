# HANDOFF — start here

**Project.** Empirical study of the edge of stochastic stability: what a stochastic optimizer's
settled top sharpness (κ = η·λ_B) is, and whether there's a universal, path-computable scalar that
equals a fixed value at the edge for **any** optimizer/batch/β. Branch `lily`, worktree
`.claude/worktrees/gbs-search`. Lean setup: CIFAR-10, `mlp_s` (789k params), num_data=2048, MSE, CPU.
Code in `experiments/`, results (gitignored) in `results/`. Run env:
`export DATASETS=/Users/xq/Desktop/moonshot/eoss/datasets EOSS_SKIP_CHECKSUM=1` and the venv
`/Users/xq/Desktop/moonshot/eoss/.venv`.

## Established results (solid — don't re-derive)
1. **GBS = 2 is the at-the-edge signature** (GBS = E_B[sᵀH_Bs/(−gᵀs)]): SGD at all batches, momentum
   at large batch (GBS≈2.00). Small-batch momentum/Adam sit **below** the edge.
2. **"The buffer moves the house, not the weather."** In a paired SGD+SGDM sweep at matched batch,
   *every* fluctuation statistic (catapult clustering, rate, excursion-size distribution) is
   optimizer-independent (set by batch noise); only the **operating-point position** (κ/GBS) tracks
   the buffer. Position is buffer-set; weather is not.
3. **The sub-edge point is a REGULATED ATTRACTOR, not a phase.** λ is restored to it from both above
   and below (park-vs-attractor test, `park_test.py`). There is **no force-free / "metastable phase."**
4. **Position = a saturating min, not a ratio.** Free-exponent regression rejected R = memory/τ_rot
   (τ_rot exponent ≈0); the (β,batch) table shows large-batch κ\* = **2(1+β) exactly**. So
   κ\* = min( 2(1+β) [β-edge, GBS=2, stability, universal], reach(batch) [loss-geometry, non-universal] ).

## THE LIVE THREAD — κ_spec (this is what to work on) → read `KSPEC_DESIGN.md`
The two momentum plateaus 2(1±β)/η are **one** closed-loop marginality condition λ·|T(ω\*)| = 2 at two
frequencies of the momentum transfer T(ω) = η/(1−βe^{−iω}): ω\*=π (coherent period-2)→(1+β), ω\*=0
(DC)→(1−β). So the **universal path-computable scalar (no optimizer math):**
`κ_spec = λ_B·|T̂(ω*)| = 2 at all batches`, T̂ = measured gradient→step transfer, ω\* = measured mode
frequency. **R is just the decoherence parameter that selects ω\*** (explains R's half-collapse).
- **CONFIRMED free:** ω\* migration is real — increment lag-1 autocorr r₁ goes +0.95 (small-batch DC)
  → −0.99 (b2048 β0.9, period-2); endpoint κ_spec = 1.95 ≈ 2. The mechanism is banked.
- **PENDING (your job):** does κ_spec = 2 hold *quantitatively* across the interpolation band? Needs a
  targeted ~15-cell rerun — the old sweep logged UNSIGNED cosines (phase unrecoverable). **The
  signed-logging patch is DONE and verified (commit 9fd10f8):** `slow_sweep.py` now logs signed
  in-frame `gu/su/mu` + fixed-frame `gu0/su0`.
- **Next steps (spec'd in KSPEC_DESIGN.md):** (a) wire a ~12-cell ladder driver (β0.9 b8→b2048 +
  β0.99 b8) — reuse `experiments/p1_isoR.py`'s pattern AND make a liveness-bisect pre-flight mandatory
  (high-β/large-batch cells die at canonical lr — this has cost us cells 4×); (b) offline κ_spec
  analysis (Welch T̂ from signed gu/su, spectral-integral κ_spec, in-frame-vs-fixed-frame check),
  evaluated against the **pre-registered marginality gate + pass criteria** in KSPEC_DESIGN.md;
  (c) add a Nesterov optimizer to `utils/optimizer.py`, run the trio (zero-math third-threshold test);
  (d) wide-grid collapse figure ONLY if the gated ladder passes.

## Doc map
- **`KSPEC_DESIGN.md`** — the live κ_spec spec: hypothesis, banked result, pre-registered gate + pass
  criteria, runner-patch spec (done), estimator plan, cells. **Primary working doc.**
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
