# TAXONOMY_PREREG.md — blind taxonomy test on two unseen optimizers (registered 2026-09-06, BEFORE runs)

Debate outcome (unifier vs splitter forks): splitter's blind-taxonomy design, merged with the unifier's
zero-cost state-function scoring. Two never-run optimizers; every prediction below derives from the
written rules R1–R4, committed before any cell runs.

## Prediction-generating rules (fixed first)
- **R1 mechanism class:** noise-maintained iff the update passes gradient magnitude through (any linear
  filter or per-coordinate rescaling); normalization-enforced iff update norm is fixed by construction.
- **R2 LTI small-batch plateau:** gap = κ_B − κ_full = 2/G_DC ± 30%, G_DC = the optimizer's DC gain
  (measured, frozen G2 protocol; audit qualifier: hottest-live-lr O(1) constant).
- **R3 coherent edge:** GBS = 2 ± 0.1; raw κ threshold = 2/|T(π)|-gain.
- **R4 basin:** LTI class → kicks fatal at 32–128×A0; adaptive (preconditioned) class → survives 128×A0.
- Noise-maintenance read (frame-consistent, learned from P1a's bug): post-swap λ vs pre-swap λ_full
  (same frame), ≥ +20% within 6k = noise-maintained; within ±10% = not.

## The two optimizers and their structural classification (R1, before data)
- **SGD-Momentum-Damped (DHB), β = dampening = 0.9:** strictly LTI; buffer window 10 steps but DC gain
  **1** (weights (1−β)Σβ^k sum to 1). Class: noise-maintained, LTI (cliff expected).
- **RMSProp (β₂ = 0.99):** Adam minus the gradient EMA; per-coordinate rescaling. Class: noise-maintained,
  adaptive (robust, no cliff).

## Registered predictions (10 cells, mlp_s, health-masked, frozen protocols verbatim from prior preregs)
| cell | prediction |
|---|---|
| TX_dhb_b8, TX_dhb_b16 | **gap = 2.0 ± 0.6 in raw κ units (M = G_DC = 1: SGD-like DESPITE the 10-step buffer)**; rival window-rule prediction 0.2 — 10× separation, THE core discriminator |
| TX_dhb_b2048 | GBS 2.0 ± 0.1; **κ_raw threshold ≈ 2(1+β)/(1−β) = 38 ± 20%** (formula-flagged); plateau at hottest live lr |
| TX_dhb_kick_b8 | LTI cliff: diverges at a 32–128×A0 kick |
| TX_dhb_swap_b8 (→2048) | noise-maintained: λ (vs pre-swap λ_full) ≥ +20% within 6k |
| TX_rms_b8, TX_rms_b16 | whitened gap **batch-independent**; 3-outcome: ≈ 2 ± 30% (Adam's 4.6 was the EMA's doing) / ≈ 4.6 ± 30% (preconditioner owns it) / neither → kill |
| TX_rms_b2048 | GBS 2.0 ± 0.1 |
| TX_rms_kick_b8 | survives all kicks incl. 128×A0 (adaptive class) |
| TX_rms_swap_b8 (→2048) | noise-maintained (≥ +20%/6k, whitened frame); **state-function endpoint: lands within ±25% of TX_rms_b2048's own de-novo plateau** (target measured in this same sweep, before the swap cell is scored) |

**Kill conditions:** (K1) DHB gap ∉ [1.4, 2.6] → the gap law fails on a fourth strictly-LTI optimizer →
exposed as SGD-family curve fitting. (K2) RMSProp lands in "neither" → no handle on adaptive methods.
(K3) any R1 class assignment contradicted by R4/noise-maintenance reads → the taxonomy is not
structurally predictive.

**Unifier's zero-cost addendum, recorded:** Muon's P1 "failure" (λ flat/−18% after b8→2048) is
consistent with the state-function reading — its de-novo b2048 set point (λ = 123) sits BELOW its b8
plateau (259); the swap-up run relaxed to 106 = 0.86 of that target. Post-hoc reinterpretation, recorded
here; treated as registered only for future cells (TX_rms_swap_b8 above).
