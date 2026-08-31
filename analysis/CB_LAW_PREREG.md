# CB_LAW_PREREG.md — the noise-gap edge law (registered 2026-08-30, BEFORE the confirmation runs)

**Exploratory finding (existing ceilings, both archs):** at every stationary small-batch memory-edge
plateau, η_eff·(λ_B − λ_full) ≈ 2, where η_eff = η/(1−β) (Muon: momentum), λ_B = top batch-Hessian
eigenvalue, λ_full = top full-set eigenvalue (both measured in-run). Equivalently
**κ_B = κ_full + 2(1−β)** — so C(B) = 2/(1 − λ_full/λ_B) was the spectral-gap ratio, not a new constant.
b8: HB β0.6/0.9/0.99 → 2.01/2.41/2.22; Nesterov → 2.55; Muon(0.95) → 2.07; SGD → 1.87 (raw ceilings span 80×).
mlp_l b32 → 2.01. Known failures: λ_full/λ_B → 1 cells (memory edge dissolves → coherent edge takes over
= the min-law), converged/float32-dead cells, Muon b32/b128 (0.55/0.12 — gap ~0; b16 discriminates).

## Registered predictions for the NEW cells (mlp_s, stride 1, health-masked, late-half medians)

PASS criterion per cell: measured η_eff·(λ_B − λ_full) ∈ [1.4, 2.6] (2 ± 30%) at a stationary
(|drift| < 0.15) plateau. Secondary: the gap κ_B − κ_full = 2(1−β) tracks (1−β) across the β grid at
fixed batch (log-log slope 1.0 ± 0.2 vs (1−β)). Muon memory M read from its momentum (0.95 → 20, 0.9 → 10).
SGD cells: which-edge test only (its own GBS=2 edge expected to bind; gap ≈ 2 would be a bonus, not required).

| cell | optn | β/mom | b | predicted κ_B − κ_full |
|---|---|---|---|---|
| CB_hb03_b8 | SGD-Momentum | 0.30 | 8 | 1.40 |
| CB_hb08_b8 | SGD-Momentum | 0.80 | 8 | 0.40 |
| CB_hb095_b8 | SGD-Momentum | 0.95 | 8 | 0.10 |
| CB_sgd_b16 | SGD | — | 16 | which-edge test |
| CB_hb05_b16 | SGD-Momentum | 0.50 | 16 | 1.00 |
| CB_hb09_b16 | SGD-Momentum | 0.90 | 16 | 0.20 |
| CB_hb097_b16 | SGD-Momentum | 0.97 | 16 | 0.06 |
| CB_nest_b16 | SGD-Nesterov | 0.90 | 16 | 0.20 |
| CB_muon095_b16 | Muon | 0.95 | 16 | 0.10 |
| CB_muon09_b16 | Muon | 0.90 | 16 | 0.20 |

KILL condition (registered): if the β grid at fixed batch shows the gap NOT ∝ (1−β) (slope outside
[0.8, 1.2]) or ≥ 4 of the 8 memory cells fall outside [1.4, 2.6], the gap law is dead and the residual
b8→b128 trend (2.4 → 1.8) is to be reported as the obstruction. Standing rules: liveness-bisect per
cell; raw primitives only; no doc edits by the pipeline; censored cells listed.
