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

---
> **[VERDICT 2026-08-30 22:40, after data — CB_RESULTS.md, commit f4fa3e7.]** Kill condition NOT
> triggered (3/9 memory cells outside [1.4, 2.6]; threshold was ≥4). Secondary slope test PASSED
> (heavy-ball gap ∝ (1−β)^0.91, registered [0.8, 1.2]). Per-cell gap×mem: HB 2.07/2.28/2.58 (b8
> β0.3/0.8/0.95), 1.95/2.55/2.55 (b16 β0.5/0.9/0.97) — **6/6 PASS**; Nesterov 2.76 (marginal fail,
> +6% over band; 2.55 at b8 — small systematic elevation vs HB); **Muon 0.94/0.64 FAIL**, with
> κ_B = 0.208 identical at momentum 0.95 and 0.9 → Muon's ceiling is momentum-independent; its
> effective memory is not 1/(1−mom) (orthogonalization decouples buffer memory from stability).
> SGD b16 which-edge test: GBS = 2.04 → own edge binds, as registered. Standing law: for the
> classical momentum family, **κ_B = κ_full + 2(1−β)**; Muon = boundary finding (frame/memory open).

---
## Adam addendum (registered 2026-08-31, BEFORE the Adam gap runs)

Adam (β₁ 0.9, β₂ 0.99) was excluded above: its `lam_batch` is whitened while `lam_full` was raw — the
gap was not computable. `slow_sweep` now logs `lam_full_w` (whitened full-subset top eig, warm power
iter, Adam only). Cells: adam_b8/b16/b32 (mlp_s, lr from preflight, 30k steps). **3-outcome test on
gap_w = κ̃_B − κ̃_full at the stationary plateau:**
- (H1) memory factor = DC gain of the gradient EMA = 1 → **gap_w = 2 ± 30%** at every batch (Adam ≡ SGD in the whitened frame);
- (H2) memory factor = window 1/(1−β₁) = 10 → **gap_w = 0.2 ± 30%**;
- (H3) neither → Adam joins Muon as a boundary case (β₂ preconditioner memory confound; check `pdrift`).
Whichever holds must hold at ALL THREE batches to count. No other quantity will be consulted for the verdict.

> **[ADAM VERDICT 2026-08-31 23:05, after data — ADAM_GAP_RESULTS.md.]** Outcome **(H3)**: gap_w =
> 3.88 / 4.58 / 4.59 at b8/b16/b32 — outside both registered bands (2±30%, 0.2±30%) at every batch
> (b8 additionally nonstationary, drift +0.18). Structured failure: the whitened gap is
> **batch-independent (~4.6)**, mirroring Muon's momentum-independent ceiling — Adam has a constant
> curvature-noise gap in its own frame, but the 2/(1−β₁) calibration does not transfer; β₂
> preconditioner memory is the registered suspect. Adam joins Muon as an instrument/frame boundary
> case for the gap law.

---
## G2 addendum: the FORMULA-FREE gap law (registered 2026-09-06, BEFORE the CB2 runs)

Exploratory result (all stationary ceiling cells on disk): replacing 1/(1−β) with the path-MEASURED
low-frequency gain G2 = |T̂(ω→0)|/η (Welch gu0→su0, fixed frame; protocol FROZEN: nperseg =
min(2048, 2^floor(log2(n/6))), skip bin 0, average lowest max(3, K/64) bins PSD-weighted) gives
**(λ_B − λ_full)·η·G2 = 2** for the LTI family in the DC-dominated domain (b ≤ 32): SGD 1.88,
HB 13 cells 1.63–2.44 (median 2.2, incl. mlp_l 1.89/1.93), Nesterov 2.52–2.76 (elevation +15–25%).
G2 validates against 1/(1−β) (9.4–9.5 at β0.9; 1.00 SGD). Non-LTI: Adam prod 3.3–3.6 (measured
whitened DC gain 0.8), Muon 0.3–0.6 with band-SENSITIVE G2 (not LTI). Domain limit: b ≥ 64 has no
DC drive (G2 seed-inconsistent) — coherent-regime handover.

**CB2 confirmation cells + registered predictions (prod = (κ_B − κ_full)·G2, same frozen protocol):**
| cell | optn | β/mom | b | prediction |
|---|---|---|---|---|
| CB2_hb085_b8 | SGD-Momentum | 0.85 | 8 | prod ∈ [1.5, 2.6] |
| CB2_hb06_b16 | SGD-Momentum | 0.60 | 16 | prod ∈ [1.5, 2.6] |
| CB2_nest06_b16 | SGD-Nesterov | 0.60 | 16 | prod ∈ [2.0, 3.0] (elevation persists) — prod ∈ [1.5, 2.0) would instead say the Nesterov elevation was a β0.9 artifact; either is informative, outside [1.5, 3.0] is a FAIL |
| CB2_muon05_b8 | Muon | 0.50 | 8 | ceiling κ_B within ±20% of mom-0.95 b8 value 0.259 (momentum-independence) AND prod < 1 (stays outside the LTI law) |
KILL: either HB cell outside [1.5, 2.6] → the G2 law does not survive out-of-sample; report and stop.

> **[CB2 VERDICT 2026-09-06 10:55, after data — G2_CONFIRM_RESULTS.md, with the RED_TEAM_AUDIT
> lr-conditioning qualifier attached.]** No kill: hb085_b8 = 2.19, hb06_b16 = 2.03 (both in [1.5, 2.6];
> (1−β) scaling now confirmed at 8 β values × 2 batches out-of-sample). nest06_b16 = 1.96 → registered
> bin [1.5, 2.0): **the Nesterov elevation was a β0.9 artifact** (boundary value, 1.96 vs bin edge 2.0 —
> noted). muon05_b8: prod 0.19 < 1 ✓ but κ_B = 1.102 ≫ ±20% of 0.259 (λ-normalized ~2.8×) and drift
> −0.25 (nonstationary) → **momentum-independence of Muon's ceiling NOT supported at mom 0.5**; the
> 0.9/0.95 agreement was a large-memory coincidence; Muon stays unmodeled. All constants read as
> hottest-live-lr O(1) values per the audit — the sharp-edge question remains open (bracket test).
