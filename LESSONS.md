# LESSONS — methodology earned the hard way (read before running)

This project killed **three headline candidates** to its own controls (a fast-coordinate γ instrument,
a "metastable phase," and a "coupling mediates R" mechanism). Every one died the same way. These rules
are the second contribution of the work — apply them to κ_spec and everything after.

## The one rule that generalizes
**Don't conclude from an instrument that can't resolve the two things you're deciding between.**
Every false headline was a conclusion drawn from a measurement that couldn't distinguish the hypothesis
from its alternative:
- thin source ladder: "parks near source" vs "attractor near source" — indistinguishable → looked like
  a phase; a ladder spanning the target *both ways* showed it was an attractor (slope≈0).
- catapult detector dead-time (refractory=20): "regular" vs "clustered" — the dead-time *manufactured*
  B<0; at honest refractory the sign-flip vanished.
- absolute vs fluctuation-scaled threshold: "quieter basin" vs "different baseline variance" — the
  ratio reversed when scaled to each cell's own floor.
The fix is always the same move: **widen the measurement until the alternatives separate** (span both
ways, shrink dead-time, fluctuation-scale, add the control). If you can't widen it, report "instrument
can't resolve this," not a verdict.

## Controls (no instrument ships without one)
- **Matched-batch control.** Batch noise dominates almost every statistic. Run the SGD twin at the
  *same* (batch, lr) and report the SGD-vs-SGDM **contrast/ratio** — it cancels the noise. The one
  analysis all week that included this is the one whose result survived.
- **Circularity / shared-error check.** If the mediator and the outcome are built from the same
  estimated object (e.g. both use the per-step u_B), correlated measurement error inflates partial
  correlations. Re-run with a *decorrelated* version of one (we used λ_full — a different eigenvector);
  if the effect collapses, it was shared error. Split-half estimation (odd/even-step geometry) is the
  clean version — build it into logging, don't discover you need it later.
- **Actuator-independent confirmation.** An intervention that works on one regime may be a
  *constraint-side* actuator that can't move the other (the lr-pulse displaces λ against an active
  constraint but not a slack one). Confirm with a second actuator that works everywhere (the transplant
  moves θ directly, η/β untouched — so it can't tip a cell into divergence either).

## Pre-registration (against forking paths / post-hoc excuses)
- Write the **decision rule + pass criteria BEFORE seeing the number**, in the analysis script's
  docstring (e.g. `phase_analysis.py`, `KSPEC_DESIGN.md`). "The cells that failed weren't marginal" is a
  post-hoc escape hatch — close it with a **blind gate** (label cells by an *independent* instrument
  before computing the test statistic) and a **two-sided** prediction (X on gated cells, not-X on the
  rest).
- A statistic selected *after* seeing candidates (instantaneous → median → energy-weighted) is
  **hypothesis-generating**, not confirmed — it earns "confirmed" only **out-of-sample** on cells that
  didn't participate in the selection.

## Measurement hygiene
- **Log raw signed primitives; compute nothing clever online.** Unsigned cosines destroyed the phase
  (κ_spec) — irrecoverably. Save the vectors/signed dots; all spectra/fits/estimators are offline and
  re-runnable as they're refined. (This also means: if you might want a quantity later, log its raw
  ingredients now — vectors, not summaries.)
- **Gate on activity, not just "not diverged."** A run can be stationary (GBS plateaued) but *dead*
  (dynamics stopped: step-norm collapsed, or u_B frozen with κ≈0). The regime question is only posed for
  *live* cells. Compute the gate from the dense series (step-norm late/early ratio, κ drift, frozen-u).
- **Beware endogenous predictors.** Trajectory-measured quantities (τ_rot from the path) leak the
  outcome (dead dynamics → frozen u_B → low R). Prefer the exogenous version (R_noise = fixed-θ
  resampled rotation) for causal claims, even if it collapses worse; report both.
- **Timescales must fit the window.** A relaxation-rate fit over a window ≪ the true τ reads a slow
  return as "no return" (biased the phase verdict toward slack). Size relax windows ≥ 5τ from the
  *measured* autocorrelation; state park as a timescale bound (1/k vs the reference), not "didn't move."
- **Estimator edge-sensitivity.** arccos(r₁) is hypersensitive at r₁→1 (the DC end) — use the full
  spectral estimate, not a fragile point statistic, where the gain lives at an endpoint.
- **Verify "too clean" numbers.** "exactly 0 ± 0" was a null-space floor artifact. Suspiciously clean =
  degeneracy until proven physics. Cross-check magnitudes (a 90× discrepancy in "k" between two fit
  methods = they're measuring different modes, not one constant).

## Compute hygiene
- **Liveness-bisect pre-flight (standing rule).** Designed grids lost ~half their cells to the live-lr
  window **four times** (b2048 calibration, β0.99 endpoints, high-β large-batch twice). Make bisecting
  lr down from canonical (until non-diverging AND non-crawling) a mandatory pre-flight in every cell
  spec. The `slow_sweep_driver` has the divergence-check logic to reuse.
- **Targeted runs over re-sweeps.** A per-cell scalar test (κ_spec) needs ~15 cells (a ladder + a trio +
  seeds), not 178. Gate the wide/expensive figure on the small result.
- **Runs are background jobs immune to session limits once launched.** Smoke-test the plumbing on a
  short cell first (catches the bugs that waste hours), then launch and monitor. Save primitives with
  periodic flushes so a killed cell keeps partial data.
