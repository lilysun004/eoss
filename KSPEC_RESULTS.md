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
