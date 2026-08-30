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

> **[ANNOTATION 2026-08-30 — registered caveat resolved, phase-locking explanation WRONG.]** The
> stride-1 ladder read GBS_med 0.33 at heavy-ball b2048; the cause is neither aliasing nor
> phase-sensitivity but a numerically dead run (loss ~1e-9 → float32 update < half-ulp from step
> ~6000; `analysis/HB_B2048_GBS_PROBE.md`). Health-masked, GBS = 2.00 = κ_spec. The old 0.61
> stride-2 reading was the same death. Secondary test outcome on the GOLD sweep (health-masked,
> stationary cells): corr 0.91, origin slope 0.87 (`kspec_results/gold/agreement.json`).

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

---

# ADDENDUM 3 (2026-07-12, registered before any pooled-frame number is computed):
# pooled-frame construction + three-estimator reconciliation protocol for kappa_ms

Context at registration: K=8 readings seen so far -- at-edge b512/b2048 kappa_ms=1.97-1.99
(audits 0.98/1.00 PASS), b128 kappa_ms=1.31 (audit 0.92 PASS), b32/b8/b8-beta0.99 NO CROSSING
<= c=2 with FAILED or pending audits (b32: 0.45 @K8, 0.69 @K32 -> frame-limited, withheld).
Eq-21 with fixed-u inputs: matches at deterministic end, scattered/seed-inconsistent at small
batch. The frozen-single-Hessian frame is the wrong coordinates at small batch (rotating family).

## Frame rule (one rule, all cells)
V = batch-POOLED frame: top-1 eigvec of H_B for ~160 fresh construction batches (+ frozen
full-H top-8 anchor columns), SVD-orthonormalized; K* = smallest leading-dimension with
HELD-OUT (40 fresh batches, disjoint from construction) mean capture ||VV^T u_B||^2 >= 0.9.
K* reported per cell (small at b2048 = degenerate family; large at b8 = spread family; if
held-out capture does NOT converge by pool size, the operator genuinely needs the bulk ->
explicit-operator route dies honestly, replica route (iii) is what survives).

## Three estimators, trust-ordered; reconciliation BEFORE any cell is read
 (i)  explicit pooled operator: Sigma <- E_pool[J Sigma J^T] power iteration, rho(c); direct
      Eq-13 analogue. Projection bias UP.
 (ii) projected replicas (current gamma_2 machinery under pooled V). Same bias direction.
 (iii) FULL-SPACE replicas (no projection of dynamics; fresh-batch HVP per replica-step at the
      frozen point), subspace READOUT ||V^T z||. Reference estimator.
Protocol: run all three at b128_s0 and b8_beta0.9_s0.
 - (i) ~ (iii) at both -> projection safe, (i) everywhere.
 - (i) > (iii) at b8 but agree at b128 -> gap = measured frame leakage; QUOTE (iii) as c*_2.
 - disagree at b128 -> STOP, debug pooling/bookkeeping before reading any cell.
Sanity anchors: pooled-V at b2048 must reproduce kappa_ms ~ 2; pooled-V at b128 must match the
K=8 c*_2=1.52-1.53 -- if it moves, the earlier "audit-clean" b128 label was too generous and is
re-opened (reported, not patched).

## Bookkeeping rule
ONE estimator, ONE frame-construction rule, ALL cells: after reconciliation picks the winner,
the at-edge calibration readings are RECOMPUTED under it; only then are the b8/beta0.99 verdicts
and the margins (Reading A vs Reading B of ADDENDUM 2) read.

---

# ADDENDUM 4 (2026-07-12 ~23:00): RECONCILIATION ROUND-1 FREEZE + bracket mapping
# (committed BEFORE the b128 divergence bracket runs)

## Round-1 numbers, frozen as-measured (chain of custody; no silent updates)
- b128_s0: (i) c*_2=1.484, (ii) c*_2=1.528 (= fixed-frame K8 reading), (iii) c*_2=1.051
  [(iii) curve smooth: -0.0052, -0.0038, +0.0074, +0.0156, +0.0344, +0.2407 at
  c=0.85..2.0 -- NOT the flat-then-kink floor signature]. DISAGREEMENT detected by protocol.
- b8_beta0.9_s0 (iii), partial: gamma2 POSITIVE at c=0.85 (+0.029) and c=1.0 (+0.034) --
  frozen-linearized loop MS-unstable at the operating point (real cell stable via nonlinear
  regulation), pending SEs + flat-control; suspicion registered: few-replica upward bias
  under b8's heavy-tailed M_t.
- b8 frame non-convergence is a FINDING (methods): held-out capture 0.268 at K=8 ->
  0.510 at K=120, K*=-1. The unstable family at b8 is not low-dimensional; the operator
  route needs a growing frame there, (iii) is the only valid estimator -- by measurement,
  not by branch choice.

## Live hypothesis for the (i,ii)-vs-(iii) gap
(i)/(ii) are frame-blind to per-batch 2nd/3rd stiff directions (top-1-capture criterion);
(iii) sees them. Decisive test: top-3-per-batch enriched frame -> (i)/(ii) should DROP toward
~1.07 (convergence from above) if frame-blindness is right.

## GROUND-TRUTH BRACKET (gating item, launched first): b128 divergence bracket FROM THE
## CHECKPOINT (theta*, buffer* restored; short horizon 3000 steps so lambda adaptation is
## limited -- the registered clean comparator), c in {1.1, 1.3, 1.5}.
Pre-committed mapping (temptation to re-interpret post-hoc is maximal right now):
- DIES at ~1.1x  -> (iii) VALIDATED; plateau is MS-MARGINAL; the 50%-margin reading was a
  frame artifact; paper's marginality claim confirmed in operator form; interior-attractor
  question reframes as "marginality maintained by what feedback".
- SURVIVES to ~1.5x -> margin REAL; min(wall, reach) stands; (iii) has residual
  contamination despite the smooth curve -- hunt continues.
- SURVIVES 1.3x but DIES 1.5x -> AMBIGUOUS (nonlinear extension precedent: old arbiter saw
  survival ~1.6x past linear threshold). Registered follow-up: transplant-above-plateau at
  1.2x (linear-regime displacement: relaxes back = margin real; energy grows = marginal).

## (iii) quoting rule
(iii) c*_2 is NEVER quoted bare: always alongside the flat-control tangent's apparent
V-readout growth (bulk->V injection baseline; a partial floor shifts the crossing LEFT, and
1.051-vs-1.15 is exactly the size such a floor produces). Quote as excess-over-control.

## Framing pre-write (both endings, before the bracket chooses)
- If MS-marginal: kappa_spec=2 certifies the coherent first-moment wall where phase survives;
  kappa_ms=2 certifies the mean-square wall where it doesn't; "every plateau is at ITS wall,
  binding moment order set by the coherence regime" -- the moment hierarchy returning as the
  answer. - If margin real: kappa_spec=2 at the coherent edge + kappa_ms as the universal
  WALL with a noise-scaled margin (deficit-vs-CV(h)^2 law) = min(wall, reach) final form.

---

# ADDENDUM 5 (2026-07-12 ~23:20, committed BEFORE the b8/beta0.99/b32 bracket results print;
# those runs are already queued behind the b128 c=1.2 refinement)

## Operational wall definition (three markers now in play; fixed before the decisive cell)
c*_2-EMPIRICAL = ONSET OF VARIANCE GROWTH, not survival. Per bracket run, report three markers:
 1. ONSET (the linear MS wall): loss excursion above the quiet baseline within the run --
    operationalized as max_loss > 1e3 x the source cell's plateau-loss median near the
    checkpoint, OR final kappa < 0.9 x initial kappa (lambda shaving = regulation responding
    to growth). The b128 readings under this rule: c=1.1 quiet (below wall), c=1.3 excited ->
    onset in (1.1, 1.3), c*_emp ~ 1.2 +/- 0.1.
 2. SEVERITY (catapult size growth vs c).
 3. DEATH (divergence) = wall + cubic/regulation budget, NOT the linear wall (old arbiter
    lesson: SGD survived 1.6-1.8x past linear threshold).
Survival brackets overestimate the linear wall by a cell-dependent regulation budget.

## Registered branch: REGIME-DEPENDENT Reading A/B
Old arbiter data puts the b8 beta0.9 wall at ~1.05-1.2x operating while kappa_raw sits ~11x
below the coherent ceiling. If the b8 ONSET lands there: a cell far below its FIRST-moment
edge but AT/NEAR its SECOND-moment edge -- the two-moment picture confirmed at the
noise-dominated endpoint by ground truth. Then the coherent conclusion is REGIME-DEPENDENT:
margin ~1.2 at intermediate B (Reading B), marginal at small B (Reading A) -- the
interior-attractor margin shrinking as noise grows. This is a registered ending, not a
surprise to be reconciled post-hoc.

## Regulation budget (new registered column, computable from brackets + old data)
budget(cell) = survival multiplier / onset multiplier. b128 >= 1.25 (survived 1.5/onset ~1.2).
Old SGD ~1.7. Prediction (third appearance of the R story): coherent cells have a deep cubic
rescue budget; decohered cells a THIN one (buffer cannot couple to the shaving mechanism).
Column to report: budget vs R.

## Estimator program = calibration, not truth-source
Enriched-(i)/(ii) and control-corrected-(iii) are now judged against measured onsets:
enriched-(i) should drop 1.5 -> ~1.2 at b128; control-corrected (iii) should rise 1.05 -> ~1.2.
If they meet at the empirical onset from opposite sides, reconciliation complete; if
enriched-(i) stalls, quote it with its measured bias band (no indefinite convergence-chasing).
FINAL TABLE RULE: the paper's kappa_ms column is recomputed for ALL cells under the single
calibrated construction; the multi-construction numbers quoted during reconciliation are
methods-appendix material only.

---

# ADDENDUM 6 (2026-07-13 ~00:50, committed BEFORE any Adam cell runs): Adam trio predictions

Runner: slow_sweep Adam mode (smoke-verified: whitened dxu = su to 1e-7). Frame = LIVE per-step
robust preconditioned geometry (adjudicator protocol: d = 1/sqrt(sqrt(vhat)+0.1*median),
whitened observables g~=d*g, s~=s/d, m~=d*m, u from d*H_B*d power iteration). P-drift logged
per step (cos(d_t,d_ref)) + sparse full snapshots -> offline local-linearization check (block
the Welch analysis if drift is fast relative to the window). beta slot = beta1 = 0.9,
beta2 = 0.99 fixed.

Pre-registered predictions:
1. Adam b2048 (at its preconditioned edge per the adjudicator arc): kappa_spec = 2 with the
   measured gain playing a role NO formula predicts -- Adam's effective filter (beta1-EMA
   composed with slowly-adapting 1/sqrt(vhat)) has no closed form; strongest form of the
   zero-optimizer-math claim. Gate for Adam cells: plateau laws with beta -> beta1 (coherent
   2(1+b1), DC 2(1-b1)) on PRECONDITIONED kappa~, per the adjudicator's kappa_precond ~ 2(1+b1)
   finding; labeled as an assumption in the report.
2. Adam b8 (sub-edge per adjudicator, R_precond ~ 9): kappa_spec < 2, gate sub-edge; onset
   bracket adds its point to the margin dataset.
Preflight: liveness-bisect from lr 0.001 (b8 validated; b128/b2048 start there), precond-kappa
crawl floor 0.08.

---

# ADDENDUM 7 (2026-07-13, committed BEFORE tonight's cells run): second overnight queue

## Disclosures / resolutions from today's audit round
- ADDENDUM 6 did NOT declare the Adam frame primary (registration gap, disclosed): uniform
  rule applies -- in-frame kappa_spec is primary everywhere (Adam b2048: 1.94 primary,
  1.999 fixed-frame robustness).
- nest_b128 overshoot: ALL registered estimator variants agree (2.21-2.25, 5 seeds) -> genuine
  anomaly. Decomposition: measured gain 1.514-1.519 vs pi-formula 1.474 (within 3%); position
  kappa_raw = 1.46 sits 9% ABOVE the deterministic Nesterov law; bracket wall (1.1, 1.25]
  confirms the operative wall sits above the deterministic law. Discriminating prediction for
  tonight's nest_b256/b512 cells: if the overshoot DECAYS toward 2 as coherence purifies
  (r1 -> -1), it is the mixed-residual/noise-elevated-wall story; if it PERSISTS at fully
  coherent Nesterov cells, it is Nesterov-specific incompleteness of the single-frequency
  reduction.

## Fifth-threshold prediction: Adam beta1=0.5, b2048 (adam05_b2048)
Ideal EMA gain at omega=pi: (1-b1)/(1+b1) = 0.333 (vs 0.0526 at b1=0.9); preconditioned edge
kappa~* = 2/0.333 = 6. Registered: measured gain ~ 0.33, kappa_spec ~ 2, with NO Adam formula
in the estimator. Tests that the Adam result tracks beta1 rather than being a b1=0.9
coincidence. (Also registered: adam_b2048 gain-vs-ideal 0.057 vs 0.0526 -- 3 more seeds decide
systematic-vs-noise; if systematic, it is a REAL deviation of Adam's effective filter from the
textbook EMA, reported as a finding.)

## Tonight's other registered additions
- Adam onset brackets (b2048, b128) via full-state-dict checkpoint restoration (replay now
  saves opt.inner.state_dict() -- exact Adam v-hat restored, lr overridden after load).
- Margin-dataset densification: SGDM b64 (new batch, liveness preflight), nest s1 brackets,
  nest_b256/b512 brackets after their cells.
- kappa_ms seed padding (est-i/ii under the one construction) at b128 + at-edge cells.
- Muon b2048 LEFTOVER-COMPUTE ONLY: dense signed logging with RAW-H frame; mu column invalid
  (Muon buffer not exposed) -- geometry/frame decisions are explicitly deferred to analysis;
  tonight only logs primitives.
Standing rules: no SUMMARY/KSPEC_RESULTS writes, no regressions/fits tonight.

---

# ADDENDUM 8 (2026-07-13, registered BEFORE any Tier-2 fit is run): the two-tier north star
# and the Tier-2 fitting protocol (fit itself deferred to the analysis session)

**Tier 1 (done):** kappa_spec = 2 at the coherent edge, all optimizers, formula-free.
**Tier 2 (target):** ONE empirical curve Y = f(X) across ALL optimizers for the deviation.

## Y (primary): the GROUND-TRUTH onset margin from brackets, margin := c_onset - 1
(onset = registered excitation rule, ADDENDUM 5). Companion panel: regulation budget :=
c_death/c_onset. The deficit 2 - kappa_spec and the estimator margin c*_2 - 1 are consistency
OVERLAYS, never the fitted Y (immunity to the frame/floor biases cataloged in ADDENDA 4-5).

## X candidates (passive path statistics of the undisturbed plateau, zero hyperparameters):
r1 (in-frame increment lag-1 autocorr), 1 - |r1|, spectral weight off the pi-peak
(1 - P_hi where P_hi = PSD fraction at omega > 3pi/4), and pool CV(h)^2 (fixed-u curvature
noise). The Tier-3 lesson stands: raw landscape noise CANNOT be X (beta-sweep); the
optimizer's filtering must be in X implicitly via path measurement, never via beta.

## Circularity guard (rule, fixed now): Y and X from DIFFERENT instruments.
Y is intervention-based and built from LOSS/lr observables only (bracket excitation/death);
X from passive statistics (may be u_B-derived since Y is not). No shared estimated ingredient
on both axes (the shared-eigenvector mediation lesson, applied prophylactically).

## Expected structure (from the first overnight): TWO related curves, both anchored at (0,0)
at the deterministic end -- margin vs X and budget vs X, opposite regimes (coherent: ~0.05
margin/zero budget; noisy: 0.2-0.3 margin/wide budget). If both collapse on the SAME X, that
is the R-story's final quantitative form (R was the hyperparameter-flavored proxy for this X).

## Registered fallback + failure criterion:
- Fallback: a two-variable surface (decoherence x noise amplitude) shared across optimizers
  still counts as Tier-2 universality.
- FAILURE criterion: after fitting on measured X's only, an optimizer-identity dummy adds
  significant explanatory power (residual sorts by optimizer) -> the honest claim shrinks to
  per-family fits and the residual structure is the next mechanism hunt.
Deliverable shape: two-panel figure (margin | budget) vs the winning X, all optimizers pooled,
overlays for the estimator-based Y variants.

---

# ADDENDUM 9 (2026-07-14, committed BEFORE the architecture battery launches):
# second-architecture replication battery (mlp_l: 512-wide x 4-layer, 2.37M params = 3x mlp_s,
# deeper AND wider; same data/loss/harness; EOSS_MODEL env -- zero porting)

CNN excluded on measured COST, not porting: cnn preset exists but fwd+bwd b2048 = 3.5 s on CPU
(days per at-edge cell). One axis tonight: architecture via mlp_l.

## Battery (~18 cells, A_ prefix, results/kspec_arch/): all lrs re-bisected (windows do NOT
## transfer -- the twice-fatal failure mode; preflight gates everything)
- Tier 1: SGD b2048, SGDM b0.9 b2048, Nesterov b2048, Adam b2048 x2 seeds.
  PREDICTIONS: kappa_spec ~ 2 on all; measured gains ~ 1, 1/(1+b), (1+2b)/(1+b), and Adam's
  own (near-ideal-EMA, +systematic deviation allowed as on mlp_s).
- Tier 2: onset brackets on those + sub-edge SGDM b8/b32, Adam b8, SGDM b0.99 b8 x2 seeds.
  QUESTIONS (branches pre-stated): (a) same FORM margin ~ sqrt(CV(h)^2)? (b) same CONSTANT
  0.54? same-slope => physical constant; different-slope-same-form => universal law,
  setup-dependent coefficient -- BOTH reported as measured. (c) coherent band again at
  margin ~ 0.03 / budget ~ 1.12? (d) beta0.99 memory residual replicates (2x above amplitude
  law) => the missing memory variable is physics, not an mlp_s quirk.
- Structure (free): r1 migration up the ladder; nest elevation ~1/b (nest b128/b512 x1 seed);
  sub-edge kappa_spec < 2.
Honest branch: if kappa_spec ~ 2 replicates but the margin constant shifts, Tier 1 is
architecture-independent and Tier 2's coefficient is not -- report both as measured.
Skip tonight: Muon, CE-loss/dataset axes, any fitting.

---

# ADDENDUM 10 (2026-07-21, committed BEFORE any post-battery runs launch):
# (A) extended-budget completion of the two pre-plateau mlp_l Tier-1 cells;
# (B) Muon validation program with the deferred frame decision made now.

Context on record from the battery's first pass (data already landed, no new peeking):
A_b2048_beta0.9 (kspec 1.56/1.70) and A_adam_b2048 (0.29/0.24) read NOT-2, diagnosed
pre-plateau -- brackets show kappa still climbing toward the wall (SGDM 2.75->3.7 vs
2(1+b)=3.8) at the 10000-step budget; mlp_l sharpens slower than mlp_s. The blind gate is
expected to label these cells sub-edge/mixed AS-IS; those first-pass rows stay in the tables.

## (A) A2_ cells: extended budget, NOT hotter lr (registered choice)
Hotter lr moves the cell to a different wall and breaks comparability with the mlp_s
counterparts; budget extension tests the same cell later on the same trajectory.
- A2_b2048_beta0.9 and A2_adam_b2048, seeds 0/1, SAME preflighted lrs (0.0065 / 0.001),
  max_steps 30000, u0_at 20000 (window length 10000 = first-pass window length).
- PREDICTIONS: once at-plateau by the blind gate, kappa_spec in [1.8, 2.2]; gain ~ 1/(1+b)
  for SGDM; Adam near-EMA with the +several-% systematic deviation seen on mlp_s allowed.
- Brackets after, same grid (1.05, 1.15, 1.3), ckpt at 3x window start rule -> use step 20000.
- FAILURE branch (pre-stated): still pre-plateau at 30000 (kappa still climbing at window
  start) -> report "mlp_l sharpening timescale exceeds budget; cell censored". No silent lr
  change, no third attempt tonight.

## (B) Muon: frame decision registered NOW
Ground truth is FRAME-FREE: onset brackets (replay + real-optimizer restart) need no
transfer-function frame. Brackets are therefore the PRIMARY Muon validation; the spectral
instrument is secondary and gated.
1. Frame decision: raw parameter frame is the only admissible spectral frame for Muon --
   orthogonalization is not a diagonal preconditioner and exposes no inv-sqrt; inventing a
   whitened frame post hoc would be formula-smuggling. Registered expectation: the polar-
   factor step is state-dependent, so the LTI instrument may fail as it did for adam05
   (filter-memory vs adaptation-timescale). Pre-stated branches:
   - gated non-stationary or LTI-invalid -> kappa_spec reading registered INSTRUMENT-INVALID
     (adam05 class); the bracket carries the at-wall/margin claim alone.
   - gated stationary + coherent -> kappa_spec interpreted at face value against 2.
   The existing single cell (L_muon_b2048_s0: kappa_spec 1.33, stationary=false, gain 10.8)
   is already in the first class pending replication.
2. Cells (mlp_s): L_muon_b2048_s1, s2 (lr 0.001, momentum 0.95, same protocol as s0) for
   reproducibility of the reading + stationarity flag; then brackets on s0/s1,
   cs = (1.05, 1.15, 1.3, 1.5) -- wide grid because no prior wall location exists for Muon.
   Margin/budget rows enter the Tier-2 dataset as DATA ONLY (no refit tonight).
3. Cells (mlp_l, if the machine clears the queue): preflight-bisect Muon lr from 0.001
   (windows do NOT transfer), then A_muon_b2048 s0/s1 + brackets, same rules.
- PREDICTIONS: Muon plateaus INSIDE a measurable MS wall (onset bracket finite, margin > 0);
  no prediction on kappa_spec until the stationarity gate speaks. If margin lands on the
  0.54*sqrt(cv2(h)) curve without optimizer identity, the kill-test result extends to a
  non-diagonal-geometry optimizer -- reported as measured either way.

---

# ADDENDUM 10.1 (2026-07-21, committed while phase-2 runs, BEFORE the A2 brackets, hotter
# brackets, cv2h pools, or any Muon spectral data exist): readout disciplines + classifier
# amendment + registration corrections.

## (i) A2 crossing-kappa readout (pre-stated before the A2 cells finish)
The first-pass reading "SGDM b2048 s1 crosses at kappa ~ 4.0 vs 2(1+b) = 3.8" is NOT yet a
wall-location replication: 4.0 > 3.8 is ABOVE the deterministic edge, the (1.15, 1.3] bin is
wide (the crossing kappa interpolates inside it), and this cell family is the one flagged
pre-plateau -- and mlp_s taught us walls sit off the formula in BOTH directions (noise-
elevated nest, Jensen-elevated SGD). Registered readout for the A2 rerun of exactly this
cell: report the bracket kappa_trace crossing as an INTERVAL (last kappa before the first
excitation event, first kappa after) together with the c-bin that produced it; "replicates
2(1+b)" may only be claimed at the resolution of that interval -- never as a point-match
within an untested tolerance. If the A2 crossing interval sits wholly above 3.8, that is a
real small elevation and is reported as one; wholly containing 3.8 -> consistent; the
first-pass 4.0 carries the pre-plateau + bin-coarseness caveats permanently.

## (ii) First-event-fatal bracket guard (classifier amendment, registered before the
## 1.6/1.9/2.3 hotter brackets run)
At hot multipliers a cell may be past DEATH, not just onset, and the onset/death distinction
blurs when the first excursion is terminal. The margin dataset needs ONSET. Rule: if the
lowest event c in a cell's bracket set is a death with no non-fatal excitation at or below
it, the row is logged as "onset <= c, death <= c, unresolved between" -- onset_hi = c and
c_death = c retained, but margin is CENSORED (NaN, no geometric midpoint) and a machine-
readable flag `onset_death_unresolved` is set. Applied uniformly on reassembly: any existing
mlp_s/mlp_l rows this reclassifies are reported as reclassified (column addition + censoring
rule only; the frozen row-inclusion table is untouched).

## (iii) Muon frame registration -- weakness stated (correction to ADDENDUM 10 B1)
"Raw frame is the only admissible spectral frame" overclaims. Correct statement: raw frame
is chosen as PRIMARY because it is the unique hyperparameter-free, state-free option. Muon
DOES have a natural non-raw geometry -- per-layer spectral, from the orthogonalization's
structure; a layer-spectral frame instrument is DEFERRED as branch (b), not excluded. A
kappa_spec failure in the raw frame therefore reads "raw-frame LTI instrument invalid",
never "no frame could work" -- only one frame was tried. Bracket-primary structure stands.

## (iv) ADDENDUM 9 margin-constant comparison: THREE pre-registered outcomes (wording fixed
## now, before any mlp_l cv2h number exists)
(1) same sqrt(cv2(h)) form + same 0.54 constant -> the constant is physical;
(2) same form + different constant -> the law is universal, the coefficient setup-dependent;
(3) no sqrt(cv2) form at all on mlp_l -> the amplitude law is mlp_s-specific and Tier-2
takes real damage. All three reported as measured; (2) is the expected outcome and is fine.

## (v) budget_artifact flag (bookkeeping made machine-readable)
The pre-plateau first-pass readings (A_b2048_beta0.9_s0/s1, A_adam_b2048_s0/s1) stay in the
tables but carry a `budget_artifact` column in the tier2 datasets, superseded by the A2
cells -- prose alone will not protect a future aggregation script.

## Verification order when the chain reports (registered): blind gate -> A2 kappa_spec
## table -> ADDENDUM 9 constant comparison -> Muon branches. Muon lands in the boundary-
## findings section next to adam05 regardless of outcome -- its job is mapping the
## instrument's domain, not padding the optimizer count.
