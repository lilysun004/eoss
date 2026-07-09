# Edge of Stochastic Stability — Findings Report

*Investigation into a universal, path-computable quantity that stabilizes at the edge of stability for any optimizer. Branch `lily`. Lean setup: CIFAR-10, `mlp_s`, num_data=2048, MSE, CPU, seeds fixed. All experiment code under `experiments/`, results (gitignored) under `results/`.*

---

## 0. The question and the one-paragraph answer

**The dream (from `instructions.tex`):** find a single quantity, computable from the optimization path alone (steps, gradients, curvature — no per-optimizer math), that stabilizes at a known value at the edge of stability for *any* optimizer. Motivating candidate: the **generalized batch sharpness**
`GBS = E_B[ sᵀH_B s / (−gᵀs) ]` — the update-curvature ratio, which is 2 at any marginally-stable operating point. Empirically GBS≈2 at full batch for all optimizers, but **< 2 at small batch for stateful optimizers** (momentum, Adam, Muon).

**The answer:** the universal quantity exists, but it is **conditional**. Marginality of the closed loop — equivalently **GBS = 2 ⟺ frozen-cocycle Lyapunov γ = 0 ⟺ a perturbation persists rather than decays** — is the universal edge signature *in the optimizer's own geometry*, and it holds **whenever the optimizer's step can track the unstable direction**. Whether it can is set by a single, causal, geometry-general dimensionless control parameter:

> **R = (optimizer state-memory timescale) / (unstable-direction rotation timescale) = (1/(1−β)) / τ_rot**

- **R ≲ 1 → marginal regime:** the system sits *at* its stability edge, GBS = 2, held there by continuous feedback.
- **R ≫ 1 → metastable regime:** the state buffer averages over a fast-rotating unstable direction and **can't couple to it**, so the system parks in a damped sub-edge basin (GBS < 2), stabilized by noise and rare catapults rather than marginal oscillation.

So the "GBS < 2" deficit is **not** a failure of the theory — it is the metastable regime, and R predicts which regime a given (optimizer, batch, β) is in. The metastable basin is quantitatively described by **correlated linear random-matrix (moment-Lyapunov) theory**, not iid Kesten and not (dominantly) nonlinear truncation.

---

## 1. Two regimes, defined dynamically

Both look identical on the loss curve (a fluctuating plateau). They differ in *what holds the system there*, and the distinction is causal and measurable:

| | **Marginal** | **Metastable** |
|---|---|---|
| Where it sits | *at* the stability edge | *below* the edge, inside a basin |
| What holds it | continuous negative feedback (thermostat) | damping; escapes via rare catapults |
| Linearized growth | marginal oscillation persists | perturbation **decays** (γ<0), rare escapes |
| Signatures | GBS≈2; γ_frozen=0 at c*≈1; perturbation persists | GBS<2; γ_frozen<0; perturbation decays; heavy-but-truncated tails |
| Members (small batch) | SGD | momentum, Adam, (Muon likely) |
| Members (large batch) | SGD, momentum, Adam — everyone | — |

**One-line version:** marginal = *held at the edge by feedback*; metastable = *held below the edge by damping, leaking over it by noise*.

---

## 2. R — the control parameter (and that it is *causal*)

R = (buffer memory `1/(1−β)`) / (rotation time `τ_rot`), where `τ_rot = 1/(1−|cos(u_B(t), u_B(t+1))|)` measures how fast the top eigenvector of the per-batch Hessian rotates step-to-step.

**Batch sweep** (`mechanism_buffer_rotation.py`, SGD-Momentum β=0.9):

| batch | τ_rot | **R** | cos(buffer,u_B) | GBS |
|---|---|---|---|---|
| 8 | 1.08 | **9.2** | 0.054 | 0.33 |
| 128 | 3.26 | 3.1 | 0.147 | 1.44 |
| 2048 | ∞ | **0.0** | 0.734 | 1.99 |

At small batch u_B rotates almost completely every step; the 10-step buffer averages over all those rotated directions and **can't concentrate energy in the current unstable direction** (alignment 0.054 vs a random-chance baseline ≈0.001, so ~50× chance but tiny). At full batch u_B is stable, the buffer aligns (0.73), GBS→2. Tellingly, even at small batch the bare **gradient** aligns ~5× better than the buffer (0.30 vs 0.054) — which is exactly why SGD (step = fresh gradient, always tracks) reaches its edge and momentum (step = averaged buffer) doesn't.

**Causal test — β-sweep at FIXED batch=8** (`beta_sweep_mechanism.py`). This is the key control: it varies R *purely through buffer memory* while holding the landscape and rotation fixed.

| β | R | τ_rot | cos(step,u_B) | GBS | η·λ/edge |
|---|---|---|---|---|---|
| 0.0 (SGD) | 0.87 | 1.14 | 0.207 | 1.28 | **113%** |
| 0.3 | 1.26 | 1.13 | 0.189 | 0.99 | 68% |
| 0.6 | 2.18 | 1.14 | 0.162 | 0.66 | 35% |
| 0.9 | 8.6 | 1.16 | 0.135 | 0.35 | 10% |
| 0.99 | 87 | 1.15 | 0.071 | 0.16 | 0.9% |

**τ_rot is essentially constant (1.13–1.16)** — rotation is a property of the batch/landscape, *not* β — so R varies purely through buffer memory, and **reachability collapses monotonically as R rises**, with β=0 continuously recovering SGD's edge-reaching. This is the difference between "R is the control parameter" and "R correlates with batch size": it's the former. **R is causal.**

**R is geometry-general.** For Adam, computing R in its *preconditioned* geometry (rotation of the top eigenvector of `P^{−1/2}H_B P^{−1/2}`, P from Adam's state) gives **R_precond = 9.2** — the preconditioner rotates the geometry but does *not* slow the unstable direction (τ_rot_precond=1.09 ≈ raw). So Adam's buffer can't track its unstable direction either. R holds across optimizers in each one's own geometry.

---

## 3. The marginal regime, characterized

Marginality is the **universal edge signature**, verified three independent ways for every cell that reaches its edge:

- **GBS = 2** (loss-Taylor / batch-sharpness). SGD b8 = 1.93, SGD b2048 = 2.06; momentum b2048 = 1.87 (with η·λ = 3.75 ≈ 2(1+β) = 3.8, sitting exactly at its heavy-ball edge).
- **Frozen-cocycle Lyapunov γ = 0 at lr-multiplier c*≈1** (`frozen_cocycle_v3.py`, autodiff of the closed-loop Jacobian, restricted to the top-K curved subspace). At-edge cells: SGD_b2048 c*=1.017, momentum_b2048 c*=1.011 — both marginal at c*≈1. This is the reviewer's stochastic-edge prediction confirmed.
- **AR-pole ρ = 1** at large batch (system-ID of the iterate path). SGD_b2048 ρ=1.02 with phase≈π (the textbook period-2 sign-flip); momentum_b2048 ρ=0.99.

All three coincide because at marginal stability the unstable mode sustains a non-decaying oscillation. Notably SGD operates **above** its deterministic edge (η·λ up to ~3 = 1.5× the 2/η threshold) — the Jensen/stochastic-stabilization effect: multiplicative curvature noise makes `E[log|1−ηh|] ≤ 0` even when `E[h] > 2/η`.

**Perturb-and-relax** (`perturb_relax.py`, kick along the unstable direction, no edge formula, no β): for SGD (β=0) the perturbation **persists/grows** (γ_relax ≈ +0.015) — marginal.

---

## 4. The metastable regime, characterized

Small-batch stateful optimizers (momentum, Adam) sit **below** their edge:

- **Sub-edge sharpness & GBS:** momentum b8 GBS=0.31 (κ=η·λ=0.37, ~10% of its edge); Adam b8 GBS=0.46. Confirmed at **canonical scale** (`mlp`, num_data=8192, 30k steps): momentum stays κ=0.31 vs SGD's 3.61 in the same run — measured on the raw Hessian (momentum's correct geometry), so **not** a measurement artifact.
- **Genuine sub-edge equilibrium, not slow arrival:** momentum's κ trends **flat** (0.37→0.34 over 30k steps), not creeping toward its edge — it's a real quasi-equilibrium.
- **Perturbation decays** (`perturb_relax.py`): γ_relax goes +0.015 (β=0) → −0.012 (β=0.3, R≈1) → −0.072 (β=0.9), flipping sign at the R≈1 crossover — the damped-basin signature, causal and yardstick-free.
- **Catapults:** rare loss spikes reset sharpness; the excursion tail (`causal_regime_tests.py`) has kurtosis *decreasing* with β (4.38→0.82) — heaviest tail at marginal, lightening with damping (see §6).

### 4a. Adam — the same regime, reached only after a geometry correction

Adam nearly escaped as "marginal" via a measurement error, and adjudicating it (`adam_adjudicator.py`) is one of the cleaner results:
- Raw η·λ(H_B) = 0.44 → looks sub-edge; preconditioned η·λ(P^{−1/2}HP^{−1/2}) = **3.4–6.8** → looks *at* its edge (Cohen et al. 2022 found Adam trains at its preconditioned edge — but at *standard* batch; b8 is a regime they never probed).
- **Decisive, parameter-free (no β₁ input):** the preconditioned-top-mode GBS = **0.49, not 2** — Adam is sub-edge in its *own* geometry, top mode. And **R_precond = 9.2 ≫ 1**. So the R-mechanism *predicts* Adam metastable — no contradiction.
- Resolution of "κ_precond=6.8 yet sub-edge": 6.8 is the curvature that's *available*, but Adam doesn't *ride* it (cos(step, u_precond)=0.14). High curvature exists; the buffer can't couple to it.

**So the dichotomy unifies: small-batch stateful optimizers (momentum AND Adam) are metastable, for the same reason (R≫1), and R is geometry-general.** No contradiction with the literature — large batch (u_B stable, R small) → marginal; b8 → metastable.

### 4b. The metastable tail law — correlated-linear random-matrix theory

Along the unstable coordinate, momentum is a random linear recursion `x_{t+1}=a_t x_t + noise`; the stationary tail is `P(|x|>u) ~ u^{−α}`. The `kesten_test.py` shuffle discriminator separates two explanations:

| β | R | E[a²] | α_iid (pure Kesten) | α_ordered (real temporal order) | α_real |
|---|---|---|---|---|---|
| 0.0 | ~1 | 3.68 | (linear diverges) | (diverges) | 3.55 |
| 0.3 | 1.3 | 0.88 | 2.15 | 2.24 | 3.85 |
| 0.6 | 2.0 | 0.73 | 3.77 | 4.05 | 4.89 |
| 0.9 | 8.6 | 1.34 | **0.38** | **3.46** | 4.88 |
| 0.99 | 87 | 1.30 | **0.14** | **3.51** | 6.40 |

**In the deep metastable basin (β=0.9, 0.99): iid Kesten predicts an absurdly heavy tail (α≈0.1–0.4, infinite mean, because per-step E[a²]>1), but restoring the real temporal order collapses it to light (α≈3.5), matching reality (≈5–6).** So the light tail is explained by **temporal correlation** (the buffer decorrelates/negatively-correlates the growth factors) — **linear correlated random-matrix theory survives**; no nonlinear-truncation model needed in the deep basin. Two tie-ins: **α_iid crosses 2 (the mean-square-stability boundary) right at R≈1**, and **α_real increases monotonically with β** (Hill ordering: lighter tail with more damping, heaviest at marginal). This is the metastable half's first confirmed quantitative law — pure path statistics (growth factors off the trajectory), no optimizer math.

*(Near-marginal cells β=0.3/0.6 lean toward nonlinear truncation because nonlinearity is strongest there and E[a²]<1 so iid isn't even heavy; the clean test is the deep basin.)*

---

## 5. The corrections log — a first-class methods result

Every apparent counterexample to universal marginality died the same death: **the wrong yardstick / a measurement artifact.** This list is itself a contribution (it inoculates against exactly these errors):

1. **Under-training.** SGD_b8 read GBS=1.25 at 3k steps → 1.93 at 30k. The plateau must be verified stationary, not a fixed step count.
2. **Raw vs preconditioned Hessian.** Adam's edge is on `P^{−1/2}HP^{−1/2}`, not H — nearly flipped Adam's verdict.
3. **Null-space floor.** The frozen-cocycle Lyapunov pinned at *exactly* 0 because the free tangent drifts into the Hessian's near-null space (interpolation manifold). Fixed by restricting to the top-K curved subspace so γ can go negative.
4. **Fixed-u AR-pole (and step-PCA) breaks at small batch** because the unstable direction *rotates*. The SGD control read ρ=0.70 (should be ≈1) — so AR-pole was **dropped** as a signature; γ_relax replaces it.
5. **Frozen-point vs oscillation.** EoS is an oscillation; freezing at the EMA point θ̃ (the flat central-flow path) reads spuriously damped for *every* cell, marginal included. Only the realized-trajectory instruments are clean at small batch.
6. **Circular η·λ/edge yardstick.** "10% of edge" uses the deterministic edge formula `2(1+β)/η` — the very thing the investigation discredited, and whose validity degrades with β. Replaced by the direct causal perturb-relax test.
7. **Kick amplitude.** A 4×-natural kick catapults *every* cell out of its basin (γ>0 for all) — discrimination needs the small-amplitude linear regime.
8. **iid vs correlated growth factors.** iid Kesten grossly over-predicts the tail; the buffer's temporal correlation is the whole point (same lesson as the earlier arbiter failure).

---

## 6. Instruments and their status

| Instrument | What it measures | Status |
|---|---|---|
| GBS | update-curvature ratio (=2 at marginal) | **Solid**, geometry-invariant; use preconditioned-top-mode form for Adam |
| Frozen-cocycle γ / c* | closed-loop Lyapunov via autodiff | **Solid** for at-edge cells (c*≈1); oscillation-confounded at small batch (use EMA/along-traj carefully) |
| Perturb-and-relax γ | does a kick persist (marginal) or decay (metastable)? | **Best causal small-batch test**; fragile single-seed (multi-seed sweep underway) |
| R (mechanism) | state-memory / rotation-time | **Solid and causal** (β-sweep), geometry-general |
| AR-pole ρ | top pole of the iterate path | **Dropped** at small batch (u rotates; control fails) |
| Sharpening-suppression | is the boundary binding? | **Confounded** near interpolation (no sharpening drive) |
| Catapult / Kesten tail | tail exponent α, correlated vs iid | Qualitative ordering solid; deep-basin = correlated-linear |

---

## 7. Caveats and uncertainties

- **Lean setup.** Almost everything is `mlp_s` / 2048 / MSE / single seed. Canonical-scale (`mlp`/8192/±CE) checks confirm the momentum sub-edge finding and the Adam raw-vs-preconditioned point, but the canonical SGD control was still non-stationary at 30k (GBS climbing through 2, not a clean plateau) — I do **not** claim a precise "GBS=2 at canonical scale," only that SGD *reaches/exceeds* its edge while momentum does not.
- **Single-seed fragility.** Perturb-relax γ hovers near zero with signs that flip across amplitude/seed; the *comprehensive* SGD-Momentum sweep now running (batch×β×lr, multi-seed with fit errors, escape thresholds, full catapult distributions) is designed to give the publication-grade, error-barred version. Until it lands, treat single-cell γ signs as indicative, not definitive.
- **The alignment "can't track" is small but real:** cos(buffer,u_B)=0.04 is ~35× the random-chance baseline (≈0.001) — i.e. "can't track," but not literally zero.
- **Metastable tail:** correlation is the dominant effect in the deep basin (α: iid 0.14 → ordered 3.5), but there's a modest residual gap to reality (ordered 3.5 vs real 5–6) that could be residual nonlinearity or Hill-estimation noise. So: "correlated-linear explains the bulk; a small residual remains."
- **Muon untested** on the corrected yardsticks — it has its own (orthogonalized-update) geometry and the same wrong-yardstick exposure Adam had; presumed unresolved, not confirmed metastable.
- **Deferred (needs an active sharpening drive, i.e. pre-interpolation / more data / CE):** Arrhenius escape-rate scaling, the renewal balance (sharpening rate ≈ catapult rate × drop), and what pins the plateau level. These are stated as predictions, not findings; Kesten is the one confirmed quantitative instance.

---

## 8. Bottom line

The original dream — one scalar pinned at a constant for *all* optimizers at *all* batch sizes — does **not** exist unconditionally, because not every (optimizer, batch, β) is at an edge. What *does* exist:

1. **A universal edge signature** (marginality of the closed loop: GBS=2 ⟺ γ_frozen=0 ⟺ perturbation persists), holding conditionally on trackability.
2. **A causal, calculable, geometry-general control parameter R** = state-memory / unstable-direction-rotation-time, that predicts *which regime* any optimizer/batch/β is in — and it's computed from path + optimizer-state observables only (steps, buffers, per-batch curvature), no per-optimizer edge formula.
3. **A characterization of both regimes:** marginal (feedback-held oscillation, GBS=2) and metastable (damped sub-edge basin described by correlated-linear random-matrix theory, with α=2 at R≈1).
4. **A methods contribution:** the corrections log — the yardstick errors that make marginality look violated, and how to avoid them.

The comprehensive SGD-Momentum sweep (running) will turn the regime map into an error-barred, richly-logged dataset over batch × β × lr, with every cheap quantity logged throughout training (R, sharpness, GBS, batch-sharpness, α_g, the u_t·u_{t+1} rotation overlap, all buffer/gradient/step alignments and norms) plus the three causal tests with seed statistics and full raw series.
