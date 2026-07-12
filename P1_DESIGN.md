# P1 — Iso-R contours: pre-registered design (falsify R's *form* + out-of-sample mediation)

**Goal.** The strongest test of R's functional form. R = memory/rotation claims the two ingredients
enter *only as a ratio*. So move both together to hold R fixed and ask whether **position stays fixed
along the contour**. Everything downstream (the collapse claim, the central figure) depends on this.

## Pre-registered readout & criterion (fixed BEFORE running)
- **Primary:** for each target-R contour, `var(position) ALONG the contour / var(position) ACROSS
  contours < 0.3` (position = κ = η·λ_B; robustness on GBS and on λ_full). Contours must also be
  *ordered* (position monotone in R across contours).
- **Both axes, reported side by side:** R_traj = (1/(1−β))·(1−cos_uu) [trajectory rotation] AND
  R_noise = (1/(1−β))/τ_noise [fixed-θ resampled, EXOGENOUS]. **R_noise is the causally interpretable
  axis** (R_traj is partially endogenous: dead dynamics → frozen u_B → low R_traj, outcome leaking into
  predictor). If contours agree, report R_traj (tighter). If they disagree, **R_noise wins the causal
  claim even if it collapses worse**, and the disagreement is itself a result.
- **Companion prediction (free, more diagnostic than the ratio):** since coupling is now the proximate
  variable, cells on the same contour should match in **energy-weighted coupling** too, not just
  position. A contour where coupling matches but position doesn't (or vice versa) localizes exactly
  where R→coupling→position breaks. Log coupling as a column; pre-register "coupling constant along
  contour" alongside "position constant."

## Out-of-sample mediation replication (pre-registered into THESE cells)
The energy-weighted-coupling mediator was hypothesis-generating (selected post-hoc on the 70 discovery
cells; κ-mediation inflated by shared u_B). It earns "confirmed" ONLY out-of-sample. On P1's new cells
(which did NOT participate in mediator selection), pre-register and evaluate:
- **M1:** partial( position, coupling | R ) **excludes 0** (coupling mediates), AND
- **M2:** on the **split-half-decorrelated** position (λ from ODD steps' u_B, coupling from EVEN steps),
  the mediation SURVIVES — if it collapses under split-half, the discovery flip was correlated
  measurement error.
- **M3 (the open one):** whether R's *direct* partial (position ~ R | coupling) shrinks to ~0
  out-of-sample (→ "R acts through coupling") or stays negative (→ R carries structure beyond measured
  coupling; decorrelated λ_full already shows −0.71, so expect the latter unless split-half changes it).

## Runner logging additions (REQUIRED — makes both audits computable)
Add to the dense per-step log: (i) **energy-weighted coupling** primitives — per step save `cos_su` and
`step_norm` (already logged) AND compute `E_uB = (step_norm·cos_su)²` so the coherence-weighted mediator
is reconstructible; (ii) **split-half u_B** — every measured step, estimate the top eigvec twice from
disjoint half-batches (u_B^odd, u_B^even), log λ and cos(step,·) for each, so position and mediator can
be built from *decorrelated* geometry. Keep raw dense series (no online summaries).

## Contour construction (compute at setup from the clean-gate τ_rot table)
1. From the 70 clean-gate cells, tabulate **median τ_rot(B)** and **τ_noise(B)** per batch (validated-
   live only; activity+stationarity gate as in the re-run).
2. Target R ∈ {~0.5, ~2, ~8}. For each target, find **3–4 (β, B) combinations** that hit it via
   R(β,B) = (1/(1−β))/τ_rot(B) — do NOT assume τ_rot; use the measured table. (e.g. R≈8 plausibly via
   β0.9/b8, β0.95/small-mid, β0.99/mid — but read off the table, don't guess.)
3. Run each at its **validated-live lr** (the sweep's per-(B,β) live lr; activity gate handles divergence).
   2 seeds. Reuse the slow_sweep harness + dense logging + the two additions above.

## Interpreting failure (why this stays main-thread)
Position drifting *along* a contour can mean (a) the ratio form is wrong (→ the 2D (memory, τ_rot) map
gives the true exponents) or (b) a measurement/window artifact (τ_rot estimation error, lr-window
mismatch between the combined cells). These require judgment — do not auto-conclude "form wrong" without
ruling out (b) via the coupling companion (if coupling ALSO mismatches along the contour, it's real
physics, not τ_rot noise).

## Queued free items (before compute reset)
- **S2 collapse** re-run on the clean 70 (position-vs-R_traj vs -R_noise residual spread).
- **FDT with unified k**: one k definition (same window, same detrend, same units) on the archetype
  cells — transplant-k (long relaxation toward attractor) and slow_kick-k (short pulse-response) differ
  90×, plausibly *different relaxation modes*; unify before any constancy statement.

## Sequencing
Clean-gate re-run (done: mediation reversed to partial, R keeps direct effect on decorrelated λ_full) →
this P1 design (done) → **launch P1 after the shared session/compute reset** → S3 (discriminating
optimizers: Nesterov vs heavy-ball, Adam β1=0 sweep β2, EMA-momentum) and S5 (Muon, nonparametric
memory) queue behind P1, same logging additions, pre-registered predictions.
