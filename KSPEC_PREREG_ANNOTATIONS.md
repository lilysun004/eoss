# κ_spec pre-registered per-cell annotations — committed BEFORE any ladder cell is analyzed

Written while the ladder is running (2026-07-12), REVISED once (same session, still before any
ladder dense.npz was opened) after review caught the first draft mis-registering the mixed-cell
prediction. Only old-sweep data and pre-flight probes have been touched.

**Disclosures:** (1) the estimator plumbing smoke ran on pre-flight probe
`PRE_b8_beta0.9_lr0.002` (3000 steps, transient-heavy, not a ladder cell): κ_spec≈0.28,
gain≈0.92, r₁≈+0.25. The gate criterion predates it and uses no κ_spec. (2) The stationarity
table below was computed on OLD sweep cells only.

## The three-way prediction map (keyed on PLATEAU-NESS, not coherence class)

The first draft registered "mixed ⇒ predicted κ_spec < 2", which inverts the theory: the
spectral-integral hypothesis says a *plateaued* cell operating at mixed ω is marginal with
respect to its own broadband spectrum — the interpolation-band money case where only κ_spec can
read 2. Corrected registration:

- **(i) Plateaued cells, any ω (incl. mixed): predict κ_spec ≈ 2.** Strongest form of the
  hypothesis; mixed-ω plateaued cells (b128 headline) are its sharpest test.
- **(ii) Genuinely sub-plateau cells (κ non-stationary / still climbing): predict κ_spec < 2.**
- **(iii) Endpoint cells (coherent b2048 / DC β0.99-b8 if the blind gate labels it so):
  κ_spec ≈ 2 AND the measured gain must land at the known endpoint values as a cross-check**
  (gain ≈ 1/(1+β) at ω=π, ≈ 1/(1−β) at ω=0 — checked on the gate/anchor side; the estimator
  never contains these).

**Stationarity criterion (fixed now, computed from raw κ only, blind to κ_spec):** split the
plateau window into thirds; plateaued iff |median κ(third 3)/median κ(third 1) − 1| < 0.10.

**Stationarity measured on OLD equivalents (2026-07-12):** b8 −4.8/−4.0%, b32 −0.9/−0.3%,
b128 +2.7/+1.5%, b512 +0.7%, b2048 +0.0% → ALL stationary. So the entire β0.9 ladder is
expected class (i): **the strong form is on the line at every batch**, including b8.

**The b8 anomaly, registered rather than classified away:** b8 β0.9 plateaus at κ≈0.34, 70%
ABOVE its DC-law value 2(1−β)=0.20, with mixed r₁≈+0.25. Readings: (a) it is not in the pure DC
regime — its marginality is at intermediate ω (the κ_spec story, prediction ≈2); (b) the probe
preview (gain≈0.92 where ≈5.9 would be needed) hints the strong form may FAIL here — if the full
cell confirms gain≈1 with a stationary κ, then b8 is a plateaued cell that is NOT spectrally
marginal, and class (i) is falsified at small batch → the honest headline becomes "κ_spec = 2 on
the coherent-to-mixed branch only" or a min()-structure survival. Either outcome is informative
and will be reported as measured. (c) A harness-vs-paper-law disagreement would also hit the
Nesterov anchor — check the anchor before interpreting trio κ_spec (ruling 3 below).

## Secondary registered test — the sub-2 axis made quantitative (non-tautological form)

"κ_spec shortfall = gain-corrected raw shortfall" is circular (κ_spec/2 ≡ λ/λ_marg by
definition). The quantitative version registered instead: **κ_spec must agree cell-by-cell with
median plateau GBS** — two independent marginality instruments (HVP quadratic form s'Hs/(−g's)
vs gradient→step transfer spectroscopy), both = 2 at any binding edge, and predicted to fall
short by the SAME factor on any sub-marginal cell. Report: per-cell (GBS_med, κ_spec) with
correlation and slope across the ladder; prediction slope ≈ 1 through the origin-anchored fit.
Caveat registered: old stride-2 large-batch data shows GBS_med 0.61 at the b2048 at-edge cell —
suspected stride-2 phase-locking of a period-2-alternating GBS (same aliasing family as the ω=π
signal); the stride-1 ladder is the clean read, and if stride-1 GBS_med at b2048 still ≠ 2 that
is itself a finding about GBS phase-sensitivity at coherent cells, to be reported not patched.

## Interpretation rulings (fixed before data)

1. The gate's "OR independent attractor/at-edge evidence" clause will NOT be invoked; park-test
   attractor evidence is regulation evidence, not marginality evidence.
2. The COMMITTED instrument layer is unchanged: gates.json labels per kspec_gate.py, and the
   KSPEC_DESIGN PASS criteria (median κ_spec ∈ [1.8,2.2] AND CV(κ_spec) < 0.5·CV(κ_raw) on
   gate=at-edge cells) are evaluated exactly as committed. The three-way map above is the
   theory-faithful prediction layer evaluated ALONGSIDE it. Both verdicts get reported;
   neither is dropped post-hoc.
3. Nesterov trio anchor (STOP gate, kspec_ladder.py --anchor): Nesterov b2048 raw plateau κ
   within 15% of 2(1+β)/(1+2β) (β=0.9 → 1.357), evaluated BEFORE any trio κ_spec. Small-batch
   position reported informationally only. Anchor fail ⇒ stop and debug, no trio κ_spec claims.

## Per-cell table (predictions; blind gate still assigns the committed labels on NEW data)

β0.9: coherent law 3.80, DC law 0.20. β0.99: coherent 3.98, DC 0.02.

| cell | lr | old κ | stationary | expected class | prediction |
|---|---|---|---|---|---|
| L_b8_beta0.9    | 0.0020 | 0.34 | yes | (i) plateaued-mixed | κ_spec ≈ 2 (strong form; probe hints fail → informative) |
| L_b32_beta0.9   | 0.0050 | 0.63 | yes | (i) plateaued-mixed | κ_spec ≈ 2 |
| L_b128_beta0.9  | 0.0060 | 2.10 | yes | (i) plateaued-mixed | κ_spec ≈ 2 — HEADLINE interpolation test |
| L_b512_beta0.9  | 0.0080 | 3.66 | yes | (i)+(iii) coherent | κ_spec ≈ 2, gain → 1/(1+β) |
| L_b2048_beta0.9 | 0.0065 | 3.81 | yes | (i)+(iii) coherent endpoint | κ_spec ≈ 2, gain → 1/(1+β), fixed-frame ≡ in-frame |
| L_b8_beta0.99   | bisect TBD | TBD | TBD | (iii) DC endpoint if gated so | κ_spec ≈ 2 from raw κ ≈ 0.02–0.03; ω-resolution flag applies |

If L_b8_beta0.99 does not gate at-edge AND is non-stationary, the DC endpoint goes untested
(untested ≠ failed; the report must say which).

---

# ADDENDUM (same session, 2026-07-12, registered BEFORE any gamma_2 is computed):
# mean-square marginality (kappa_ms) — the second-moment rung of the hierarchy

**Motivation (from the ladder result + paper):** kappa_spec is a FIRST-moment certificate (coherent
in-frame response) — confirmed at the coherent endpoint (2.007 ± CV 0.006, gain = 1/(1+β) to 3
decimals, formula-free) and decisively NOT 2 at small/mid batch (0.033–1.54), where the measured
in-frame gain ≈ 1 (buffer decoheres; the 100× DC gain never materializes in-frame). The paper
(EoSS_Momentum.pdf, Eq. 11/18/21) derives the (1−β) small-batch law from MEAN-SQUARE stability —
variance accumulation across rotating frames, invisible to any per-frame first-moment gain — and
its exact 1D boundary predicts the full interpolation band. Three prior in-house results point the
same way: Kesten alpha=2 crossing at R≈1 (alpha=2 IS the MS boundary), restoring-from-above at 5×
below the deterministic edge (an MS-divergence force needs no coherent oscillation), and the
"interior regulator" open question.

**The quantity:** kappa_ms ≡ 2/c*_2, where c*_2 = the lr-multiplier at which the MEASURED
closed-loop tangent cocycle at a frozen plateau checkpoint goes mean-square marginal:
gamma_2(c) := lim (1/T) Σ_t log s_t = 0, with s_t = RMS over N tangent replicas propagated
through the optimizer's own linearized update (autodiff HVPs; optimizer enters ONLY through its
own step — no β-formula anywhere in the estimator) with i.i.d. fresh-batch reduced Hessians
M_t = V^T H_B V (top-K curved subspace, v3 null-space fix), shared renormalization.

**Estimator spec (fixed before data):** K=8 primary (K=1, K=16 robustness on 2 cells);
pool P=384 i.i.d. batch draws at frozen (θ*, buffer*); N=128 replicas; T=3000 steps after
300 burn-in; c-grid {0.5,0.7,0.85,0.95,1.0,1.05,1.15,1.3,1.5,2.0}; c*_2 by linear interpolation
of the gamma_2(c) sign change; block-split SE. gamma_1 (per-replica log-norm rate, same pool)
reported alongside for the moment-hierarchy contrast. Checkpoints regenerated by deterministic
replay of the ladder cells (replay verified against the logged loss trace before use); checkpoint
steps in the LIVE phase (ring-down excluded at b512/b2048).

**Pre-registered claims:**
1. PRIMARY: c*_2 ≈ 1 at EVERY plateaued ladder cell — small, intermediate, AND large batch.
   Criteria (mirroring the committed kappa_spec ones): median(kappa_ms) over plateaued cells
   ∈ [1.8, 2.2] AND CV(kappa_ms) < 0.5·CV(raw kappa); per-cell band kappa_ms ∈ [1.6, 2.5]
   reported per cell.
2. CONTRAST: gamma_1's crossing c*_1 sits AWAY from 1 at the noise-dominated cells (first moment
   already shown not to pin them).
3. Paper-law check (same pool, zero extra compute): Eq. 21 with (a, σ_b²) measured as
   (mean, variance) of M[0,0] (curvature along the frozen top eigvec under fresh batches) should
   predict the measured plateau κ per cell. The λ_B / along-step proxies already tried FAILED
   (wrong h_t — biased mean/suppressed variance and bulk-contaminated respectively, recorded in
   the session log); M[0,0] is the model's actual h_t.

**Registered failure mode (stated in advance):** if c*_2 ≠ 1 at the intermediate cells — if they
sit strictly inside even the mean-square boundary — then the middle band is genuinely
drive-limited, NO stability functional of any moment order pins it, and the paper's honest
answer is "no all-B stability scalar exists; here is the measured boundary of where stability
quantities apply." That outcome still closes the paper.

**Registered caveats:** (a) frozen-point oscillation confound (known from v3) is worst at
AT-EDGE cells (b512/b2048) — those are already certified by kappa_spec; the informative set for
kappa_ms is the small/middle band where the confound is mild (small oscillation amplitude).
(b) Pool bootstrap (P=384) approximates the batch distribution; robustness: re-run one cell at
P=192 to check pool-size sensitivity. (c) b2048 is FULL batch → σ_b²=0, M_t deterministic —
gamma_2 = gamma_1 there by construction; its kappa_ms is a consistency read, not an MS test.

---

# ADDENDUM 2 (2026-07-12 late session): registered BEFORE the noise-dominated gamma_2 readings
# (b32/b8/b8-beta0.99) are seen. Seen so far: b128 (c*_2=1.52/1.53, kappa_ms=1.31, seeds agree)
# and b2048 (c*_2=1.006, kappa_ms=1.988 both seeds -- instrument calibrates to 2 at the known
# at-edge cell). The b128 reading already leans toward the registered failure mode.

## Two registered readings for the remaining cells (both constructive, fixed in advance)

- **Reading A — MS-marginality is the universal pin:** c*_2 ≈ 1 (kappa_ms ≈ 2) on the plateaued
  noise-dominated cells. Then kappa_ms is the all-B stability scalar.
- **Reading B — kappa_ms is the universal WALL, position = wall minus a noise-scaled margin:**
  c*_2 > 1 tracking the cell's EMPIRICAL divergence multiplier (where training actually dies),
  not 1. Then the certificate is validated as the wall; the plateau sits BELOW the wall by
  margin c*_2 − 1, and the margin is the object to explain — the archived deficit-vs-CV(h)^2
  law (r=0.895 through the origin, all optimizers) is the standing empirical candidate
  (escape-rate/Kramers-type noise-scaled safety margin below a mean-square wall; the renewal
  picture as a POSITION law, not a phase claim).
- Distinguishing column (added to the report): **empirical divergence multiplier bracket** per
  cell from existing data (old-sweep lr grids, liveness-bisect probes, Exp-2/arbiter: SGDM b8
  stable at lr 0.0021, diverged at 0.0024 → wall ≈ 1.15× operating). Certificate VALIDATED if
  c*_2 falls inside/near the bracket; MISSING PHYSICS if it disagrees with where cells die.
  Caveat registered: the empirical wall confounds lambda re-adaptation at hotter lr (the frozen
  cocycle holds lambda fixed), so brackets from lr-grid divergence are upper-bound-flavored at
  cells where lambda adapts; the bisect-probe brackets (short horizon, less adaptation) are the
  cleaner comparator.

## Frame-capture audit — GATE on trusting the decisive cells (registered before their numbers)

The pool is M = V^T H_B V with V FROZEN top-K at the checkpoint. If per-batch top directions
u_B poke outside span(V) — the R-mechanism itself — gamma_2 under-counts energy injection and
c*_2 biases UP at exactly the noise-dominated cells. Audit (ms_frame_audit.py): captured mass
||V^T u_B||^2 over sample batches per cell. RULE: if median captured mass < 0.8 at a cell, its
K=8 verdict is NOT read; re-run gamma at enlarged K (K=32) and read that instead (and if still
<0.8 at K=32, report "frame-limited, verdict withheld"). The b128 c*_1 ≈ c*_2 "no moment
separation" attribution (tiny fixed-u noise) is HELD until its captured-mass number is in
(fixed-u cv2=0.015 may itself be frame-blindness).
