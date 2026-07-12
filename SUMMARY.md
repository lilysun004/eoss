# Edge of Stochastic Stability — Findings Report

*Branch `lily`. What controls where a stochastic optimizer's sharpness parks, and is there a universal
path-computable signature of the edge for any optimizer. Lean setup: CIFAR-10, `mlp_s` (789k params),
num_data=2048, MSE, CPU. Code under `experiments/`; results (gitignored) under `results/`.*

---

## The answer in one paragraph

The operating point (top sharpness λ, or dimensionless κ = η·λ) is a **regulated attractor for every
optimizer** — displaced λ is restored to it from both above and below (verified causally, see Part IV)
— but **where** that attractor sits is set by the buffer. SGD and large-batch momentum are pinned *at*
the stability edge (GBS = 2, the saturation condition read off the path); small-batch momentum & Adam
are regulated to a point **5× below** the edge. A single dimensionless control parameter **R = state-
memory / unstable-direction-rotation-time = (1/(1−β)) / τ_rot** sets the position continuously (R ≲ 1 →
at edge; R ≫ 1 → far below). The buffer's role is sharp and surprising: **it moves the house, not the
weather** — the noise-driven fluctuation field is optimizer-independent (identical SGD vs momentum at
matched batch), and the buffer only relocates *where* the regulated point sits within it. This is an
**R-continuum, not two phases**: a candidate KKT "constraint-slack" (force-free) regime at high R was
tested directly and **overturned** — the sub-edge point is a genuine restoring attractor, not a parking
lot. The north-star universal scalar (GBS = 2) exists **exactly where the stability constraint binds
(at the edge)**; off it, position is loss-geometry + buffer, not a pure stability quantity.

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

## Part III — Why position is buffer-set (the mechanism)

Quasi-potential picture: `dλ/dt ≈ α(λ) − c·E[x²](λ)`, α = progressive-sharpening drive, E[x²] =
amplitude of the unstable coordinate (cubic self-stabilization, Damian et al.). E[x²] needs
*amplification*, which needs coupling to u_B. **SGD couples** (fresh gradient tracks u_B) → E[x²] walls
up as λ→edge → κ* pinned at the edge (GBS = 2). **Small-batch momentum can't couple** (R≫1, buffer
averages over a fast-rotating u_B) → the amplification is weaker → κ* is regulated to a point far below
the edge (`κ* = min(edge, exhaustion-point)`, exhaustion set by loss-geometry/alignment saturation).
**Correction (the audit that killed a nicer story):** an earlier version of this section framed the two
ends as a KKT **active-vs-slack** binary (constraint active at the edge, *force-free slack* interior at
high R) with the Lagrange multiplier as an order parameter. The direct park-vs-attractor test (Part IV)
**refuted the slack half** — the high-R point is a restoring attractor, not force-free — so this is a
**continuum** in where the regulated point sits, not a phase binary. The mechanism above still holds
(coupling strength, set by R, moves the attractor); only the "slack phase" language is retracted.

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
  (cancels the loss confound) and loss logged throughout. **k(R) phase test — 5 validated-live cells
  spanning R (only θ moved, no dial toward the dead region; k = rate λ relaxes toward plateau, vs a
  drift-null floor + SGD-twin control):**

  | cell | R | SGDM k / drift-null (max) | reading |
  |---|---|---|---|
  | b8 β0.9, b32 β0.9 | **9** | 0.0, 0.7 | looked like PARK — **but a thin all-λ≈85 ladder** (SGD λ starts above plateau) |
  | b128 β0.9 | 3 | 5.4 | mixed |
  | b8 β0.6, b32 β0.6 | 2 | 4.6, 4.2 | mixed |

  The k(R) run *looked* like a force-free region at high R — **but three audits overturned it.** (i) The
  "MIXED" β0.6 cells actually **converge to one attractor** (~105) from both sides — the low k was a
  `fit_k`-target artifact (it targeted the training-tail plateau, not the true settle point). (ii) SGDM
  κ0 is **reproducible across seeds** (CV≈0.003, as tight as SGD) — not a scattered parking lot. (iii)
  The k(R) ladders were thin (few interior sources, no below-plateau arm) so "parks near source" couldn't
  be told from "attractor near source."
- **Decisive PARK-vs-ATTRACTOR test `park_test.py` (the real answer):** sources from SGDM's *own* descent
  span the plateau **both ways** (λ 46→102 vs plateau 65), transplanted into fresh SGDM with **zeroed AND
  pre-warmed buffer**, N=6000, readout = slope d(settle)/d(source) (≈1 park/slack, ≈0 attractor). Result,
  **both R≈9 cells**: every source — far below *and* far above the plateau — **returns to the plateau**
  (b8: 47→67, 102→64; b32: 62→87, 106→86), **slope = −0.05 / −0.01 → ATTRACTOR**, and the **pre-warmed
  buffer agrees** (not a zeroed-buffer transient). So the metastable operating point is a **genuine
  regulated attractor** — λ is restored from both sides — **not a force-free slack region.** The earlier
  "PARK at 75" was the thin-ladder artifact (all sources near the attractor). **The force-free phase is
  overturned.**

## Part V — The instrument graveyard (a real methods contribution)

Fast/noise statistics that each looked promising and each died to a control: γ-kick (rotation-blind at
small batch), AR-pole (u rotates), frozen-cocycle-at-a-point (oscillation-confounded), burstiness
(detector artifact), event rate (baseline-variance artifact), excursion tails (track batch not buffer),
sharpening-suppression (drive dead near interpolation). The pattern: **fast statistics carry noise
information, not regime information.** And every *false headline* shared one signature — a conclusion
drawn from a measurement that **couldn't distinguish the hypothesis from its alternative** (thin ladder:
park vs attractor-at-source; detector dead-time: regular vs clustered; absolute threshold: quiet vs
different-baseline). Every audit that killed one was the same move: **widen the measurement until the
alternatives separate** (span the target both ways; shrink the dead-time; fluctuation-scale the
threshold; add the matched-batch control). That is the transferable rule, sharper than "no instrument
without a control": *don't conclude from an instrument that can't resolve the two things you're deciding
between.*

## Part VI — Honest state & the north-star answer

- **Solid:** the R-map; GBS = 2 at the edge including large-batch momentum; the weather-universality
  (buffer moves the house not the weather); the constrained-optimization *mechanism* (Part III — with
  the active-vs-slack *phase* framing retracted); the marginal F(δλ) with measured walls (return 0.7–1.0,
  wall eps≈0.5); the sub-edge point is a **regulated attractor** (park-vs-attractor test).
- **Resolved (park-vs-attractor test — the real freeze):** there is **no force-free / KKT-slack phase.**
  The metastable operating point is a **regulated attractor** — displaced λ returns to it from both below
  and above, at R≈9, both batches, and robust to buffer pre-warming (slope d(settle)/d(source) ≈ 0). So
  the final picture is a clean **R-continuum**: *every* optimizer regulates λ to an attractor; the buffer
  (via R) sets *where* that attractor sits — continuously from the edge (R≲1) to ~5× below (R≫1) — not
  *whether* restoring exists. The earlier "phases" reading was a measurement artifact (thin ladder +
  wrong `fit_k` target), caught by the audits before it shipped. What stands, robustly: **the position
  is buffer-set while the weather is not** ("moves the house, not the weather"), and **GBS = 2 is the
  universal edge signature wherever the constraint binds.**
- **R is a strong but *approximate* axis (iso-R test):** holding R fixed and moving β and B together,
  position is *not* constant — along-iso-R / across-R variance ratio ≈ **0.5–0.6** (free 70-cell test
  0.58; designed R~2 contour 0.50), failing the pre-registered tight-pin criterion (<0.3). Position
  tracks R *direction* robustly (higher R → lower position) but with real residual, so the honest claim
  is **"R is the best single organizing axis, not a precise law"** — a measured monotone f(memory/
  rotation), not a tight functional. Mechanism audit (house standard): instantaneous step-alignment does
  *not* mediate (attenuated/artifact); energy-weighted *coherent coupling* partially mediates on the
  decorrelated position (λ_full) but R **keeps a strong direct effect** (partial −0.71) — coupling is a
  real partial proximate variable, not the whole story, and the specific functional is hypothesis-
  generating (needs out-of-sample confirmation; the designed contours were under-powered, 6/14 live,
  high-β large-batch cells need live-lr search). Coupling is *tighter* along contours than position, so
  the residual localizes to the coupling→position link.
- **The open question the retraction *created* (future-work centerpiece):** the attractor result is
  *stranger* than the slack story it replaced. Something restores λ to ~65 **from above** in a cell whose
  stability edge is ~5× away, **and from below** with no visible sharpening drive, reproducibly across
  seeds — and *neither* term of Part III's quasi-potential (edge amplification, progressive-sharpening
  drive) explains restoring-from-above far from the wall. This **unidentified interior regulator**
  (candidates: residual E[x²] coupling at the noise floor, loss-geometry curvature selection, buffer
  equilibration dynamics) is the genuine discovery the park test made *by failing to find slack*.
- **Terminology / a tension to own:** we retain "**metastable**" as a historical label for the sub-edge
  regime, but note it is now a **regulated attractor, not a decaying/damping-held basin**. This does
  raise a real tension with the early lean-run observation that small-batch momentum runs *eventually
  diverge*: plausibly both hold — a regulated attractor with **rare noise-driven escapes over much longer
  horizons** than the park test's window — but the attractor is what the direct causal test shows, and
  reconciling the two quantitatively (escape-rate over long horizons) is open, not asserted here.
- **North star:** the dream "one scalar = const for all optimizers/batches" does **not** hold
  unconditionally — but the *conditional* universal exists exactly where the stability constraint binds:
  **at the edge ⟺ GBS = 2**, optimizer-agnostic and path-computable. Whether a given (optimizer, batch)
  binds is what R decides. Off the active branch, position is loss-geometry + history, not a stability
  quantity — which is itself the honest final answer to why no universal stability scalar can pin it
  there.

The paper closes on: **one regulated attractor whose position R sets continuously — pinned at the edge
where the stability constraint binds (GBS = 2), regulated below it where the buffer can't couple to the
unstable direction — with the weather-universality result as the evidence that the physics lives in the
drift (where the attractor sits), not the noise (which is optimizer-independent).**
