# Edge of Stochastic Stability — Findings Report

*Branch `lily`. What controls where a stochastic optimizer's sharpness parks, and is there a universal
path-computable signature of the edge for any optimizer. Lean setup: CIFAR-10, `mlp_s` (789k params),
num_data=2048, MSE, CPU. Code under `experiments/`; results (gitignored) under `results/`.*

---

## The answer in one paragraph

The operating point (top sharpness λ, or dimensionless κ = η·λ) is set by a **constrained
optimization**: GD-at-EoS implicitly solves `min L s.t. sharpness ≤ edge`. Two KKT regimes:
**marginal** = the stability constraint is **active** (κ pinned *at* the edge, GBS = 2 is the
saturation condition read off the path, Lagrange multiplier > 0); **metastable** = the constraint is
**slack** (κ parked in the interior, set by drive-exhaustion, not by stability). A single dimensionless
control parameter **R = state-memory / unstable-direction-rotation-time = (1/(1−β)) / τ_rot** decides
which: R ≲ 1 → active (SGD, large-batch momentum); R ≫ 1 → slack (small-batch momentum & Adam). The
buffer's role is sharp and surprising: **it moves the house, not the weather** — the noise-driven
fluctuation field is optimizer-independent, and the buffer only sets *where* the system parks within
it. The order parameter is the multiplier (shadow price of stability), which is why a genuine binary
coexists with perfectly continuous fluctuations. The north-star universal scalar (GBS = 2) exists
**exactly on the active branch**; off it, position isn't a stability quantity, so no stability metric
pins it.

---

## Part I — The R regime map (the solid backbone)

- **GBS = E_B[sᵀH_Bs / (−gᵀs)] = 2 is an at-the-edge signature**, not an across-optimizer constant.
  Every cell at its edge reads ~2: SGD at all batch sizes, **and momentum at large batch** (b512/b2048,
  β ≤ 0.9, all GBS ≈ 2.00 — the alternating-buffer result: at the period-2 edge the buffer sums to
  g/(1+β) and the (1+β) cancels in the ratio). Small-batch momentum/Adam park **below** the edge
  (GBS ≈ 0.3, κ ≈ 10–20% of edge).
- **R is causal and geometry-general.** Fixed-batch β-sweep: τ_rot ≈ const (rotation is a landscape/
  noise property, not β), R varies purely through buffer memory, and edge-reachability collapses
  monotonically. Adam is metastable too (R_precond ≈ 9; parameter-free preconditioned-top-mode
  GBS = 0.49, not 2). Corrections log (a first-class methods result): under-training, raw-vs-
  preconditioned Hessian, null-space floor, fixed-u/AR at small batch, circular η·λ/edge yardstick.

## Part II — The paired dense sweep: "the buffer moves the house, not the weather"

A **paired SGD + SGDM** sweep (matched (batch, lr), β ∈ {0, 0.6, 0.9, 0.99}, dense per-step
λ/GBS/aₜ/norms/alignments; 178 cells; `slow_sweep*.py`) let every statistic be compared as a
**matched-batch column ratio** — SGD vs SGDM at identical batch-noise, only the buffer differing. The
result is clean and one-sided:

- **Every passive EVENT statistic is optimizer-independent** (identical SGD vs SGDM at matched batch):
  catapult **clustering** (burstiness — retired as a *detector dead-time artifact*: B<0 dies at
  refractory r=2), catapult **rate** (fluctuation-scaled SGDM/SGD ratio drifts with threshold, opposite
  the naive read), and the **excursion-size distribution** (p99/p50, Hill α, kurtosis all track *batch*,
  not the buffer).
- **The only robust matched-batch discriminator is EQUILIBRIUM POSITION** (κ/GBS at edge vs ~5× below at
  identical noise).

Physical statement: the fluctuation field (weather) is set by batch noise and is the same for SGD and
momentum; the buffer only relocates the operating point within it. This **retroactively explains the
whole instrument graveyard** — every failed instrument was a *fast/noise* statistic (γ kicks, AR-poles,
burstiness, rates, tails); every surviving one is *position* (κ, GBS, alignment). It also means passive
statistics **cannot** answer the phases-vs-continuum question — which is why the verdict rests on causal
kicks. (β0.99 b8 is a real **dead/no-window** region of the map: diverges > 2e-4, crawls κ≈0.02 below —
at R≈90 even the basin disappears.)

## Part III — The KKT frame (why position vs weather is exactly the right split)

Quasi-potential picture: `dλ/dt ≈ α(λ) − c·E[x²](λ)`, α = progressive-sharpening drive, E[x²] =
amplitude of the unstable coordinate (cubic self-stabilization, Damian et al.). E[x²] needs
*amplification*, which needs coupling to u_B. **SGD couples** (fresh gradient tracks u_B) → E[x²] walls
up as λ→edge → constraint **active**, κ* pinned at the edge. **Small-batch momentum can't couple**
(R≫1, buffer averages over a fast-rotating u_B) → no amplification wall → κ* set by where the *drive*
dies (α→0: interpolation/alignment saturation) in the **slack interior**. The two regimes are the
inequality active vs slack — an LP-style binary, order parameter = the multiplier, coexisting with
continuous weather. `κ* = min(κ_constraint [=edge, computable, GBS=2], κ_exhaustion [loss-geometry +
history, not a stability quantity])`.

## Part IV — Causal tests of the constraint (the decisive layer)

The fast-coordinate kick (γ on the iterate) is **dead at small batch** (archetype 2×2: SGD_b8 and
SGDM_b8 both γ≈0 in every projection — the kick loses its identity in ~τ_rot≈1 step, regime-independent;
ill-posed, not merely rotation-contaminated). So we probe the **slow variable** (sharpness λ, a scalar
— no rotating-coordinate problem):

- **Marginal side — lr-pulse F(δλ) `slow_kick.py`:** displacing λ against the active constraint gives a
  clean force-vs-displacement curve — **active-constraint / thermostat confirmed for every SGD cell**:
  interior return **0.72–0.83 (b32), 0.81–0.95 (b128), 0.98–1.01 (b512)** (λ relaxes back = attractor),
  then a **hard wall** (divergence) at **eps ≈ 0.5** (lr×1.5) — a *measured wall position* per cell.
  (b8 is noisy, return 0.9–2.7, no clean wall — small-batch chaos.) The relax window is sized to ≥5τ,
  so a slow return isn't false-read as "slack."
- **The lr-pulse is a *constraint-side* actuator** — it works on the active (marginal) cell but
  **cannot displace a slack (metastable) λ** (displacement ≈ noise, return = noise/noise). Itself
  consistent with slackness, but not a positive test — hence the transplant.
- **Metastable side — transplant actuator `transplant.py` (η-clean, direct λ displacement):** SGD's own
  progressive-sharpening checkpoints form a graded λ-ladder; transplant each θ into the SGDM optimizer
  (buffer zeroed, warm-up excluded) and watch λ, with an SGD-into-SGD control at the *matched* source
  (cancels the loss confound) and loss logged throughout. **Resolved verdict — a gradient in R, not a
  sharp phase line:**
  - **Deep endpoint (b32 β0.9, R≈9): the KKT-slack signature.** Interior source (λ 101.7, plateau 84.5)
    → SGDM **PARKS at 98.7** (moves only ~17% toward plateau) while the SGD control *climbs* to 217.8.
    Force-free interior vs active twin — a genuine causal phases contrast, loss-controlled. *Caveat:
    single interior source* (the fine targets collapsed onto one sparse checkpoint) — suggestive, needs
    more deep-endpoint sources to firm.
  - **Shallow point (b8 β0.6, R≈2): restoring present.** All interior sources (104–131, plateau 96) →
    SGDM **RETURNS to ~86–95**, SGD control climbs to 141–164 — both regulated toward *different*
    attractors (continuum-like). β0.6 sits near the active boundary, so this is expected.
  So the active→slack transition is **real but gradual in R**: a genuine force-free slack region emerges
  at the high-R endpoint (phases-like there), with continuous restoring at moderate R.

## Part V — The instrument graveyard (a real methods contribution)

Fast/noise statistics that each looked promising and each died to a control: γ-kick (rotation-blind at
small batch), AR-pole (u rotates), frozen-cocycle-at-a-point (oscillation-confounded), burstiness
(detector artifact), event rate (baseline-variance artifact), excursion tails (track batch not buffer),
sharpening-suppression (drive dead near interpolation). The pattern: **fast statistics carry noise
information, not regime information.** The lesson (now a rule): no instrument ships without a
matched-batch control; the one analysis that included it is the one that survived.

## Part VI — Honest state & the north-star answer

- **Solid:** the R-map; GBS = 2 at the edge including large-batch momentum; the weather-universality
  (buffer moves the house not the weather); the KKT frame; the marginal F(δλ) with measured walls
  (return 0.7–1.0, wall eps≈0.5).
- **Resolved (with a caveat):** the transplant shows the active→slack transition is **real but gradual
  in R** — the deep-R endpoint (b32 β0.9) has a **force-free slack interior** (transplant parks while the
  SGD twin climbs = causal phases contrast, loss-controlled), while moderate R (b8 β0.6) still restores
  (continuum-like). So: **R-continuum whose high-R endpoint is a genuine KKT-slack region**, not a sharp
  thermodynamic phase boundary. *Firm-up needed:* the deep-endpoint PARK rests on one interior source
  (sparse SGD checkpoints); the clean follow-up is denser checkpoint saving so the deep cells get 3–4
  interior sources, plus a deeper cell (b128/b512 β0.9 at a metastable lr) to trace where park↔return
  flips along R.
- **North star:** the dream "one scalar = const for all optimizers/batches" does **not** hold
  unconditionally — but the *conditional* universal exists exactly where the stability constraint binds:
  **at the edge ⟺ GBS = 2**, optimizer-agnostic and path-computable. Whether a given (optimizer, batch)
  binds is what R decides. Off the active branch, position is loss-geometry + history, not a stability
  quantity — which is itself the honest final answer to why no universal stability scalar can pin it
  there.

Either way the paper closes: **one constrained-optimization picture, two KKT regimes, R the parameter
that decides whether the stability constraint can bind, and the weather-universality result as the
evidence that the phase lives in the drift, not the noise.**
