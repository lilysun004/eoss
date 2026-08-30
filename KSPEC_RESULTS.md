# κ_spec + κ_ms results (2026-07-12 session) — the two-moment resolution

All pre-registrations in `KSPEC_PREREG_ANNOTATIONS.md` (5 addenda, each committed before the
data it governs). Analysis code: `experiments/kspec_*.py`, `experiments/ms_*.py`. Raw tables in
`kspec_results/`. 18 training cells (12-cell SGDM ladder + 6-cell Nesterov trio), stride 1,
signed projections; **zero divergences**; all replays bitwise-exact.

## Headline 1 — κ_spec = 2 at the coherent edge, formula-free, two optimizers, three thresholds

Committed KSPEC_DESIGN criteria on blind-gated cells: **PASS.**
median κ_spec = 2.007 (gated ladder cells b512/b2048: 2.015, 2.008, 2.006, 1.981), CV 0.006
vs criterion CV < 0.009; every sub-edge-labeled cell < 2 (two-sided holds).
The measured gradient→step gain at b2048 = 0.526 = 1/(1+β) to 3 decimals, and for Nesterov
(anchor first: raw κ = 1.357 = 2(1+β)/(1+2β), ratio 1.000 both seeds) the measured gain =
1.474–1.476 = (1+2β)/(1+β) to 3 decimals — **both transfer functions fell out of the Welch
cross-spectrum with the formulas grep-certified absent from the estimator.** Nesterov trio:
κ_spec = 2.003/2.001 (b2048), 2.220/2.206 (b128, coherent-gated, honest slight overshoot),
0.681 (b8 sub-edge; its measured decohered gain 1.82 ≈ (1+β) — the immediate-gradient term
survives decoherence, the buffer term does not).

## Headline 2 — the strong form ("every plateau is spectrally marginal") is refuted

All 12 ladder cells are stationary (class i), but mixed-ω plateaued cells are NOT first-moment
marginal: κ_spec = 0.33 (b8), 0.54 (b32), 1.51 (b128), 0.033 (β0.99 b8) with sub-1% CIs and
seed/split-half agreement. The in-frame gain at small batch ≈ 0.87–0.99 (SGD-like): the buffer
decoheres; the 100× DC gain never materializes in-frame (the ω*=0/(1−β) endpoint is untested
in-frame — and unreachable there, which is itself the mechanism finding). Decoherence measured
directly: open-loop reconstruction R² = +1.000 (b2048) → +0.97 (b512) → +0.78 (b128) → −5 (b32)
→ −43 (b8).

## Headline 3 — the mean-square wall exists, is measurable, and the plateau sits INSIDE it
## by a noise-dependent margin (Reading B; Reading A dead everywhere except large batch)

Ground truth = onset brackets (continue real training from bitwise-replayed plateau
checkpoints at c× lr, 3000 steps; onset = loss excursion > 3× plateau max — the registered
wall marker; death = wall + regulation budget):

| cell | κ_raw | κ_spec | wall c*_emp (onset) | death | κ_ms_emp = 2/c* | budget |
|---|---|---|---|---|---|---|
| b8 β0.9 | 0.34 | 0.33 | (1.22, 1.30] ≈ 1.26 | 1.30 | ≈ 1.59 | **≈1.03–1.07 (razor-thin)** |
| b32 β0.9 | 0.62 | 0.54 | (1.0, 1.2] borderline | 1.6 | ≈ 1.7–2.0 | ≈ 1.45 |
| b128 β0.9 | 2.11 | 1.51 | (1.1, 1.2] ≈ 1.15 | >1.5 | ≈ 1.74 | ≥ 1.3 |
| b512 β0.9 | 3.66 | 2.01 | ≈ 1.01 (est., calibrated) | — | 1.97 | — |
| b2048 β0.9 | 3.80 | 2.00 | ≈ 1.006 (est. + old grid) | ~2.6 (old) | 1.99 | wide |
| b8 β0.99 | 0.034 | 0.033 | ≈ 1.2–1.35 (borderline; horizon caveat) | >1.8 | ≈ 1.5–1.7 | ≥ 1.4 |

**No moment order yields a universal "=2 at the operating point": Reading A is dead** (the
registered kill-outcome for the pin form). What stands constructively:
- κ_ms (the wall) → 2 exactly at the coherent end, and the wall is CLOSE in lr-multiplier
  everywhere (1.15–1.35×) even where κ_raw is 6–60× below 2 — the second moment is the right
  axis at small batch (b8: 11× below the first-moment ceiling, 1.26× from its MS wall).
- Margin (c*−1): ~0.01 (b2048) / 0.15 (b128) / 0.05–0.2 (b32) / 0.26 (b8) / ~0.3 (β0.99) —
  grows with curvature noise, ≈0 where coherent. The archived deficit-vs-CV(h)² law is the
  standing candidate for the margin law (next session's quantitative target).
- Regulation budget: anomalously thin ONLY at b8 β0.9 (death ≈ onset); b32/β0.99 have normal
  budgets — the "thin budget ∝ R" prediction FAILED as a monotone law, one clean anomaly stands.

## Estimator reconciliation (methods; chain of custody in ADDENDUM 4/5)

Round 1: (i) explicit pooled operator 1.48 / (ii) projected replicas 1.53 / (iii) full-space
replicas 1.05 at b128 — protocol stop-and-debug. Ground truth (onset 1.15 ± .05) split them:
(iii)'s sub-wall curve is ~100% bulk→V injection floor (flat-control ≈ main below wall;
excess-over-control crossing ≈1.05–1.15 ✓), (i)/(ii) overestimate by ~0.3 (i.i.d.-pool +
residual frame truncation; top-3-enriched frame did NOT move them — 1.499/1.528 — so the bias
is not top-3 blindness; suspected temporal-correlation blindness of the i.i.d. pool).
Calibration status: (i)/(ii) exact at deterministic/coherent cells (b2048 1.006 ≈ old-grid
wall; pooled-V reproduces fixed-frame there and at b128), +~0.3 biased at mixed cells,
INVALID at b8 (pooled frame does not converge: held-out capture 0.51 at K=120 — the unstable
family is not low-dimensional; a finding, not a failure mode). Paper's Eq-21 with clean
fixed-u h-stats: exact at the deterministic end (ratios 0.99–1.02), off 0.4–1.4× and
seed-inconsistent at small batch — the 1D frame-blind law does not predict the interpolation
band; the rotating-family physics is required.

## Verdict against the original north star

The all-B, all-optimizer scalar that "equals 2 at every plateau" does not exist — at any
moment order (registered kill, now measured rather than suspected). What exists and is now
measured: **κ_spec = 2 wherever the loop is phase-coherent (universal across optimizers,
formula-free), a mean-square WALL κ_ms → 2 at the coherent end and within 1.2–1.4× everywhere,
and a noise-scaled safety margin** — position = wall − margin(noise), with the margin law the
one remaining open quantitative object (deficit-vs-CV(h)² candidate).

---

# UPDATE (2026-07-13/14 sessions): five optimizers, ground-truth walls, and the Tier-2 verdict

## Flagship table, final form (kappa_spec at the coherent edge, FORMULA-FREE)
| optimizer | cell | n seeds | raw threshold | measured gain (theory) | kappa_spec |
|---|---|---|---|---|---|
| SGDM      | b512  | 5 | 3.65-3.67 | 0.549 (1/(1+b)=0.526)* | 1.99-2.02 |
| SGDM      | b2048 | 5 | 3.77-3.81 | 0.526 (=1/(1+b) exact) | 1.98-2.01 |
| Nesterov  | b2048 | 5 | 1.357 (=2(1+b)/(1+2b) exact) | 1.474 (=(1+2b)/(1+b) exact) | 2.001-2.003 |
| Adam b1=.9| b2048 | 5 | kappa~=34 (eps-audit PASS: 7% over 3 decades) | 0.0568-0.0574 (NO formula exists; ideal-EMA 0.0526 is 8% off -- the measured filter is the real one) | 1.94 in-frame / 2.00 fixed-frame |
(*b512's small gain excess = its mild off-pi spectral weight, consistent.)
Adam frame-primary registration gap disclosed (ADDENDUM 7); uniform in-frame rule applied.

## Boundary findings (results, not caveats)
1. **adam05 (b1=0.5) registered-FAIL = the instrument's validity boundary.** kappa_spec read 0.5
   while the cell is AT its wall by ground truth (plateau itself catapults to loss ~103; kappa~
   = 5.5 vs ideal edge 6). Domain of the LTI cross-spectral instrument: filter memory >>
   preconditioner adaptation timescale. b1=0.9 satisfies it; b1=0.5 does not.
2. **Adam's effective filter deviates from the textbook EMA by +8%, systematically** (gain
   0.0568-0.0574 across 5 seeds; ideal 0.0526). The loop is marginal against its OWN measured
   filter -- the reason the formula-free instrument works where formulas would be wrong.
3. **The nest_b128 overshoot (2.21-2.25, 5 seeds, all estimator variants) is a noise-elevated
   wall, not an estimator artifact:** raw position sits above the deterministic Nesterov law by
   an elevation that decays ~13/b (7.7% -> 5.1% -> 2.5% -> 0.0% across b128/256/512/2048) at
   SATURATED coherence (r1 <= -0.98 from b256 up); measured gain tracks the pi-formula within
   3% throughout. kappa_spec faithfully reports the elevated operating point.

## Ground-truth wall dataset (onset brackets, 28 rows, 4 optimizers; tier2_dataset.{json,csv})
Coherent cells (r1 <= -0.9), ALL optimizers: margin 0.025-0.05, budget 1.12-1.19 -- optimizer-
independent to bracket resolution. Noisy cells: margins 0.10-0.64 growing with noise; b8 beta0.9
budget 1.03 (razor-thin, the one anomaly); beta0.99 budget >= 1.4. Adam mid-batch cells sit VERY
deep inside (adam_b128 censored at margin > 1.2). Adam b2048 margin ~0.15 at cv2h=0: the
preconditioner is an intrinsic noise source invisible to fixed-theta curvature statistics
(its full-batch plateau weather reaches loss 0.09 vs 3e-7 for SGDM) -- flagged for fit v2.

## Tier-2 verdict (pre-registered protocol, ADDENDUM 8; frozen row table)
- **Kill-test: PASSED on every fit** (optimizer-identity dummies: best-fit p = 0.72; per-
  optimizer residuals do not sort). Margins below the wall are OPTIMIZER-INDEPENDENT.
- **Partial single-X collapse: margin = 0.54 * sqrt(CV(h)^2)** (amplitude scaling), R^2 = 0.59,
  LOOCV RMSE 0.14, through the origin (coherent-cell margins 0.025 = bracket grid resolution).
- Largest residual: beta0.99 b8 (2x above the curve) -- a MEMORY direction carried by no
  registered X; the exploratory tau_su candidate failed (su decorrelates in ~1 step everywhere).
  Open item, out-of-sample confirmation required for any new X.
- Contrast pair of scalings: margin-below-stochastic-wall ~ noise AMPLITUDE (sqrt CV^2);
  position-above-deterministic-law (nest elevation) ~ noise VARIANCE (~1/b).
- Budget: too quantized to fit (12 deaths); descriptively 1.12-1.19 coherent, 1.03-1.45 noisy.
**Bottom line: universality of the margin is CONFIRMED in the registered falsification sense
(nothing sorts by optimizer); its full functional form is only partially captured (amplitude
law + an open memory residual). That partial outcome is reported as-is.**

---

# 2026-08-30 verification pass (registered order: gate -> A2 -> constant comparison -> Muon)

## 1. mlp_l blind gate (assigned once, blind; results/kspec_arch/gates.json)
At-edge: A_sgd_b2048 s0/s1 (kappa_raw 1.972/1.968 vs pred 2.0) and A2_b2048_beta0.9_s1
(3.305 vs 3.8, -13%). Everything else sub-edge/mixed, including all four first-pass
budget_artifact cells and both A2 Adam cells (raw-frame kappa 37 vs the 3.8 raw-frame law --
the gate is not built for preconditioned frames, same disclosure as on mlp_s).

## 2. A2 verdict (extended budget 30000, same lrs; branch named per open-queue item 2)
**Branch name: "slow-approach" -- plateaued by drift gate (kappa_drift +0.005) but still
BELOW the wall; not the mlp_s picture of parking AT the wall, and not pre-plateau either.**
- A2_b2048_beta0.9: kappa_raw 2.94 (s0) / 3.31 (s1) vs wall 3.8; kappa_spec 1.584/1.758.
  s1 is gate-pass at -13% raw yet kappa_spec 1.758 narrowly MISSES the [1.8, 2.2] band:
  reported as a miss. Reading: at 30000 steps mlp_l SGDM has NOT reached its operating
  point; kappa_spec correctly reports sub-marginal (it tracks kappa_raw/wall: 3.31/3.8 x 2
  = 1.74 ~ 1.758). The instrument and the gate agree with each other and disagree with the
  15% gate tolerance -- the tolerance, not the physics, made s1 "at-edge".
- **Crossing intervals per ADDENDUM 10.1(i)** (bracket kappa_trace, last-quiet -> first-excited):
  - A2_b2048_beta0.9_s1: quiet at kappa 3.78 (c=1.15), excited entering at 4.28 (c=1.3)
    -> crossing interval (3.78, 4.28] CONTAINS 3.8 -> consistent with 2(1+beta) at the
    achieved resolution. The first-pass "4.0 vs 3.8" reading is superseded by this interval.
  - A2_b2048_beta0.9_s0: quiet THROUGH kappa = 3.80 sustained (c=1.3, 2000 steps, max_loss
    1.5e-5) -> wall strictly above 3.80; censored, no upper crossing observed.
  No elevation claim; no point-match claim. Wall location consistent with, and bounded
  below by, the deterministic 3.8.
- A2_adam_b2048: **the formula-less fixed-frame threshold at 2 REPLICATES: kappa_spec_fixed
  = 1.984 / 1.983** (mlp_s: 1.999/1.998). Raw-frame in-frame reading 0.28/0.31 as expected
  (these cells logged in raw frame; preconditioned dynamics invisible to it). Adam onset
  bracket censored quiet through 1.3 -- consistent with the mlp_s deep-inside picture.

## 3. ADDENDUM 9 / 10.1(iv) constant comparison -- outcome (1) REFUTED, (2) vs (3) open
Resolved mlp_l rows: b32 s0 margin 0.323 vs pred 0.255; b32 s1 0.049 vs 0.291 (seeds
STRADDLE the law); nest_b128 0.025 vs 0.077; nest_b512 0.025 vs 0.032 (coherent band at
grid floor, as on mlp_s). Censored rows already decisive against SAME-constant: the whole
b8 family sat quiet at c=2.3 -> margin > 1.3 vs predictions 0.37-0.48 (>2.7x, as a LOWER
bound). **Same-form-same-0.54 (outcome 1) is refuted on mlp_l.** Distinguishing universal-
form-different-coefficient (2) from no-sqrt-form (3) needs the resolved onsets -- hotter
brackets (3.0/4.0/5.5) launched, plus the fluctuation-scaled excitation rule as the
registered alternative if even 5.5 stays quiet.

## 4. Muon (bracket-primary per ADDENDUM 10/10.1(iii))
- **Spectral, mlp_s: kappa_spec 1.333/1.350/1.344 (3 seeds) -- tightly reproducible AND
  stationary=false in all three (kappa_drift -0.46 to -0.48). Registered reading:
  raw-frame LTI INSTRUMENT-INVALID (adam05 class); the stable 1.34 is a reproducible
  artifact of a drifting operating point, not a marginality reading.** mlp_l seconds this:
  kappa_spec 1.88/1.80 at r1 = -1.00, omega* = pi, but nonstationary (drift -0.32/-0.42).
- Ground truth: everything quiet through c=1.5 on BOTH architectures (max_loss <= 1e-3,
  kappa 0.2-0.3 mlp_s / 0.05-0.07 mlp_l) -> onset > 1.5, margin > 0.5 censored. kappa_raw
  0.12 (mlp_s) / 0.03 (mlp_l): in raw lr*lambda units Muon is nowhere near an edge, yet
  gbs_med = 2.002 -- the batch-sharpness measure IS pinned at 2. The wall, if any, lives in
  a non-raw geometry (layer-spectral branch (b), still deferred). Hotter brackets
  (2.0/3.0/4.5) launched to bound the onset.
- Location in the write-up: boundary findings, next to adam05 -- instrument-domain mapping,
  not a flagship row.

## 5. Hotter-bracket resolution (2026-08-30, cs 3.0/4.0/5.5 b8 family; 2.0/3.0/4.5 Muon)
**Constant comparison lands on outcome (3) under the registered rule: no sqrt(cv2h) collapse
on mlp_l.** Resolved rows, ratio margin / sqrt(cv2h): b32 s1 0.09, b32 s0 0.68, adam_b8 s1
1.17, b8_beta0.9 s0 4.7, b8_beta0.99 s1 4.1 -- no single constant, not monotone in cv2h, and
same-cell seeds still differ by a censoring class (adam_b8 s0 and b8_beta0.9 s1 quiet through
5.5 while their twins resolve). Together with the outcome-(1) refutation above: **the mlp_s
amplitude law margin = 0.54*sqrt(CV(h)^2) does not transfer to mlp_l in constant OR form, as
measured by the ADDENDUM 5 excitation rule.**

**Instrument caveat, both directions, now demonstrated within one dataset:** the fixed
3x-plateau-max excitation bar is not fluctuation-calibrated. Quiet-plateau side: the Muon
"onsets" written into the datasets (A_muon 1.05-1.3, L_muon 1.5-2.0) are RULE ARTIFACTS --
max_loss grows smoothly ~c^2 (diffusion scaling, e.g. A_muon s0: 2.3e-4 -> 5.9e-3 across
c=1.05 -> 4.5) over a 3x bar of 2.4e-4, with no excitation event and kappa <= 0.24 throughout;
treat those margin rows as invalid, Muon onset remains UNRESOLVED at > 4.5 in every real
sense. Loud-plateau side: b8-family bases 0.45-1.04 set bars of 1.4-3.1, so late onsets are
partly bar-height; adam_b8 s1 (7.9 vs 3.1) is a genuine excursion, b8_beta0.9 s0 at c=5.5
(1.42 vs 1.36) is marginal. The registered alternative -- a fluctuation-scaled excitation
rule -- is now the required instrument before any Tier-2 refit on mlp_l; rerunning it changes
Y for every noisy cell on BOTH architectures, so the mlp_s fit must be re-derived under the
same rule (registered protocol, not a patch).
