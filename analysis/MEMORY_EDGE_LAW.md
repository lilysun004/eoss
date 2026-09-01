# MEMORY_EDGE_LAW.md — the small-batch ceiling is an effective-learning-rate edge (data, 2026-08-30)

**Claim under test (registered here, before the gold SGD/Muon columns exist):** the plateau batch-sharpness κ_B = η·λ_B of any
optimizer satisfies  **κ_B = min( coherent edge , C(B, arch)·(1−β) )**, i.e. either the current-batch step is marginal
(GBS = 2; heavy-ball coherent edge 2(1+β), Nesterov 2(1+β)/(1+2β), SGD ≈ 2) or the *effective-lr* edge η·λ_B/(1−β) = C(B)
binds, in which case GBS < 2. Memory enters ONLY as η_eff = η/(1−β). C(B) is a batch/architecture constant to be explained.
Source: every stationary (|drift| < 0.15 between 2nd and 4th quarter of the health-masked window) SGD-family cell in
`results/{slow_sweep,p1_isoR,beta_sweep*,kspec,kspec_arch,...}`; late-half medians; health mask |dxu/su−1| ≤ 0.05.
Lean setup (CIFAR-10 num_data=2048, MSE). Old cells are stride-2; kspec/kspec_arch stride-1.

## Per (arch, batch, optimizer, β): median over stationary cells

| arch | b | optn | β | n_cells | lr range | κ_B | **κ_B/(1−β)** | GBS | coherent edge | which edge binds? |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_l | 32 | SGD-Momentum | 0.9 | 2 | 0.005–0.005 | 0.320 | **3.20** | 0.23 | 3.80 | memory edge (η_eff) |
| mlp_l | 128 | SGD-Nesterov | 0.9 | 1 | 0.006–0.006 | 1.461 | **14.61** | 1.29 | 1.36 | coherent (GBS→2) |
| mlp_l | 512 | SGD-Nesterov | 0.9 | 1 | 0.008–0.008 | 1.397 | **13.97** | 1.58 | 1.36 | coherent (GBS→2) |
| mlp_l | 2048 | SGD | 0.0 | 2 | 0.05063–0.05063 | 1.977 | **nan** | 1.96 | 2.00 | coherent (GBS→2) |
| mlp_l | 2048 | SGD-Momentum | 0.9 | 4 | 0.0065–0.0065 | 3.066 | **30.66** | 0.15 | 3.80 | memory edge (η_eff) |
| mlp_l | 2048 | SGD-Nesterov | 0.9 | 2 | 0.0065–0.0065 | 1.355 | **13.55** | 1.99 | 1.36 | coherent (GBS→2) |
| mlp_s | 8 | SGD | 0.0 | 3 | 0.01–0.01 | 2.752 | **nan** | 1.82 | 2.00 | coherent (GBS→2) |
| mlp_s | 8 | SGD-Momentum | 0.6 | 9 | 0.0025–0.0065 | 1.216 | **3.04** | 0.80 | 3.20 | memory edge (η_eff) |
| mlp_s | 8 | SGD-Momentum | 0.9 | 8 | 0.0015–0.002 | 0.338 | **3.38** | 0.30 | 3.80 | memory edge (η_eff) |
| mlp_s | 8 | SGD-Momentum | 0.99 | 2 | 0.0001–0.0001 | 0.034 | **3.36** | 0.13 | 3.98 | memory edge (η_eff) |
| mlp_s | 8 | SGD-Nesterov | 0.9 | 2 | 0.002–0.002 | 0.369 | **3.69** | 0.45 | 1.36 | memory edge (η_eff) |
| mlp_s | 32 | SGD | 0.0 | 3 | 0.009–0.009 | 2.649 | **nan** | 1.76 | 2.00 | coherent (GBS→2) |
| mlp_s | 32 | SGD-Momentum | 0.6 | 8 | 0.005–0.009 | 2.402 | **6.00** | 1.38 | 3.20 | memory edge (η_eff) |
| mlp_s | 32 | SGD-Momentum | 0.9 | 13 | 0.0015–0.005 | 0.634 | **6.34** | 0.63 | 3.80 | memory edge (η_eff) |
| mlp_s | 64 | SGD-Momentum | 0.9 | 2 | 0.0055–0.0055 | 1.195 | **11.95** | 1.03 | 3.80 | memory edge (η_eff) |
| mlp_s | 128 | SGD | 0.0 | 4 | 0.006–0.011 | 2.214 | **nan** | 1.92 | 2.00 | coherent (GBS→2) |
| mlp_s | 128 | SGD-Momentum | 0.6 | 7 | 0.0015–0.011 | 3.173 | **7.93** | 1.84 | 3.20 | coherent (GBS→2) |
| mlp_s | 128 | SGD-Momentum | 0.9 | 15 | 0.0015–0.011 | 2.097 | **20.97** | 1.48 | 3.80 | memory edge (η_eff) |
| mlp_s | 128 | SGD-Momentum | 0.95 | 2 | 0.006–0.006 | 1.104 | **22.08** | 1.22 | 3.90 | memory edge (η_eff) |
| mlp_s | 128 | SGD-Momentum | 0.99 | 3 | 0.0015–0.003 | 0.189 | **18.92** | 0.93 | 3.98 | memory edge (η_eff) |
| mlp_s | 128 | SGD-Nesterov | 0.9 | 5 | 0.006–0.006 | 1.470 | **14.70** | 1.75 | 1.36 | coherent (GBS→2) |
| mlp_s | 256 | SGD-Nesterov | 0.9 | 2 | 0.007–0.007 | 1.427 | **14.27** | 1.81 | 1.36 | coherent (GBS→2) |
| mlp_s | 512 | SGD | 0.0 | 2 | 0.002–0.004 | 1.930 | **nan** | 1.54 | 2.00 | coherent (GBS→2) |
| mlp_s | 512 | SGD-Momentum | 0.6 | 1 | 0.002–0.002 | 1.874 | **4.68** | 0.74 | 3.20 | memory edge (η_eff) |
| mlp_s | 512 | SGD-Momentum | 0.9 | 7 | 0.002–0.008 | 3.650 | **36.50** | 1.97 | 3.80 | coherent (GBS→2) |
| mlp_s | 512 | SGD-Momentum | 0.99 | 2 | 0.008–0.014 | 0.436 | **43.59** | 1.25 | 3.98 | memory edge (η_eff) |
| mlp_s | 512 | SGD-Nesterov | 0.9 | 2 | 0.008–0.008 | 1.391 | **13.91** | 1.91 | 1.36 | coherent (GBS→2) |
| mlp_s | 2048 | SGD | 0.0 | 4 | 0.004–0.017 | 2.000 | **nan** | 2.00 | 2.00 | coherent (GBS→2) |
| mlp_s | 2048 | SGD-Momentum | 0.6 | 4 | 0.004–0.0065 | 3.199 | **8.00** | 0.03 | 3.20 | coherent (GBS→2) |
| mlp_s | 2048 | SGD-Momentum | 0.9 | 8 | 0.004–0.01 | 3.774 | **37.74** | 1.84 | 3.80 | coherent (GBS→2) |
| mlp_s | 2048 | SGD-Momentum | 0.99 | 4 | 0.004–0.017 | 2.376 | **237.60** | 1.72 | 3.98 | memory edge (η_eff) |
| mlp_s | 2048 | SGD-Nesterov | 0.9 | 5 | 0.0065–0.0065 | 1.357 | **13.57** | 2.00 | 1.36 | coherent (GBS→2) |

## C(B, arch) = median κ_B/(1−β) over cells where the memory edge binds

| arch | b | C(B) | n | CV |
|---|---|---|---|---|
| mlp_l | 32 | **3.20** | 1 | 0.00 |
| mlp_l | 2048 | **30.66** | 1 | 0.00 |
| mlp_s | 8 | **3.37** | 4 | 0.07 |
| mlp_s | 32 | **6.17** | 2 | 0.03 |
| mlp_s | 64 | **11.95** | 1 | 0.00 |
| mlp_s | 128 | **20.97** | 3 | 0.06 |
| mlp_s | 512 | **24.14** | 2 | 0.81 |
| mlp_s | 2048 | **237.60** | 1 | 0.00 |

**Rows to ignore in the C table (auto-classified, not edge cells):** mlp_s b512/b2048 β0.99 (lr-dependent κ_B, loss ~1e-13 = float32-dead), mlp_s b512 β0.6 lr0.002 (converged, loss 1e-5), mlp_l b2048 β0.9 (pre-plateau budget artifact, RESULTS_MASTER §2). Valid C(B) rows: mlp_s b8/b32/b64/b128, mlp_l b32.

mlp_s: C(8)≈3.3, C(32)≈6.2, C(64)≈12, C(128)≈21 — grows roughly as B^0.65–0.7; reaches the coherent edge 2(1+β)/(1−β) = 38 (β0.9) by b512.
mlp_l: C(8)≈1.3 (cells nonstationary, drift −0.3…−0.5 — provisional), C(32)≈3.2. Same batch trend, different constant.

## Checks of the min-structure (predictions vs observed)

- mlp_s b128 β0.6: C(128)(1−β)=21×0.4=8.4 > 2(1+β)=3.2 → coherent edge binds → predicted κ_B 3.2, GBS 2: observed κ_B 3.2, GBS 1.9 ✓
- mlp_s b128 β0.9: 21×0.1=2.1 < 3.8 → memory edge → predicted κ_B 2.1, GBS<2: observed 2.1, GBS 1.5 ✓;  β0.95: 22×0.05=1.1: observed 1.1 ✓;  β0.99: 0.21: observed 0.19 ✓
- mlp_s b128 Nesterov β0.9: coherent edge 1.357 < 2.1 → coherent binds: observed κ_B 1.47, GBS 1.65–1.8 ✓ (slightly above)
- mlp_s b32 β0.6: 6.2×0.4=2.5 < 3.2 → memory edge: observed 2.4, GBS 1.38 ✓;  b8 β0.6: 3.3×0.4=1.3: observed 1.31, GBS 0.85 ✓
- SGD (β=0, memory 1): the memory edge is irrelevant (C(B)·1 ≫ 2); κ_B climbs toward the SGD edge (b8 lr0.01: 2.78, GBS 1.85 still rising; b128 lr0.011: 2.25, GBS 1.94; b2048: 2.00, GBS 2.00) ✓
- NOT explained: large-batch β0.99 cells (lr-dependent, loss ~1e-13 = float32-dead) and β0.6 b512 lr0.002 (converged, loss 1e-5) — excluded as non-edge cells.

## What C(B) is NOT
- not the full-Hessian DC edge (η·λ_full/2(1−β) = 0.1–0.6 at b8, 1.8 at b32)
- not 2·λ_B/λ_full (that ratio shrinks with B while C grows)

## Registered predictions for the gold sweep
- G_sgd_b8/b32: κ_B near the SGD edge (≈2.7 at the hottest live lr), GBS → 2 (drift-flagged if still climbing) — NOT at C(B).
- G_sgdm_b8/b32/b128 (β0.9): κ_B/(1−β) = 3.3 / 6.2 / 21 ± 15%.
- G_nest_b8/b32: κ_B/(1−β) ≈ 3.7 / ~6.5; G_nest_b128+: coherent edge 1.357.
- Muon: no β; if the law is about path-measured memory, Muon's small-batch κ_B ceiling should match a memory read from its step autocorrelation (test after data).

Companion: `analysis/noise_amplification.{py,json}` (in-frame amplification A, fixed-frame restoring rate c, diffusion exponent α), `analysis/collapse_figure.py`.

---
## UPDATE 2026-08-30 (evening): C(B) resolved — the noise-gap law, confirmed out-of-sample

C(B) is not a new constant: **C(B) = 2/(1 − λ_full/λ_B)**, i.e. the memory edge is
**η_eff·(λ_B − λ_full) = 2 ⟺ κ_B = κ_full + 2(1−β)** — memory compounds only the batch-specific
(noise) excess of the top batch curvature over the persistent full-batch part. This also derives the
min-structure: λ_full/λ_B → 1 at large batch dissolves the memory edge and hands over to the coherent edge.
Registered before data (`analysis/CB_LAW_PREREG.md`), confirmed on 10 fresh cells (`CB_RESULTS.md`):
heavy ball 6/6 in-band across β 0.3–0.97 and b8/b16 (gap×mem 1.95–2.58, predicted gaps hit to 7–29%,
slope 0.91); Nesterov marginally high (2.55–2.76); **Muon fails (0.94/0.64) with a momentum-independent
ceiling (κ_B = 0.208 at mom 0.9 AND 0.95)** — its effective memory is not 1/(1−mom); boundary finding.
Residual structure to own: gap×mem trends up with β (1.95 at β0.5 → 2.55–2.58 at β≥0.9) — a ~30% second-order
effect, unexplained.

**Adam (2026-08-31, ADAM_GAP_RESULTS.md):** 3-outcome prereg resolved (H3): whitened gap_w = 3.9/4.6/4.6
at b8/b16/b32 — neither 2 (EMA DC gain) nor 0.2 (window). Batch-INDEPENDENT gap ~4.6 = structured
boundary case (cf. Muon's momentum-independent ceiling); β₂ preconditioner memory suspected. The gap
law's domain: classical momentum family (HB exact, Nesterov +6%); adaptive/orthogonalized optimizers
have their own constant but not the 2-calibration.
