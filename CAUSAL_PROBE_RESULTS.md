# CAUSAL_PROBE_RESULTS.md — batch-swap ablations + kick trains (2026-09-06)

Frozen predictions: analysis/CAUSAL_PROBE_PREREG.md (committed before runs). DATA ONLY.

## Swap cells (P1 up / P2 down)

| cell | swap | pre-swap λ_B med | λ_B @ +2k (%Δ) | λ_B end-4k med (drift/2k) | diverged |
|---|---|---|---|---|---|
| CPU_sgd_b8up | 8→2048 @ 12000 | 270.5 | 109.4 (-60%) | 119.8 (+4%) | False |
| CPU_sgdm_b8up | 8→2048 @ 12000 | 180.0 | 64.7 (-64%) | 75.3 (+11%) | False |
| CPU_adam_b8up | 8→2048 @ 12000 | 4465.3 | 34250.0 (+667%) | 34825.4 (+362%) | False |
| CPU_muon_b8up | 8→2048 @ 12000 | 266.0 | 122.7 (-54%) | 110.9 (-24%) | False |
| CPD_sgdm_dn | 2048→8 @ 5000 | 582.9 | nan (+nan%) | 582.9 (+2%) | True |
| CPD_adam_dn | 2048→8 @ 5000 | 32676.7 | 4015.8 (-88%) | 4705.1 (+8%) | False |
| CPD_muon_dn | 2048→8 @ 5000 | 264.1 | 390.1 (+48%) | 272.9 (-7%) | False |

## Kick cells (P3): per amplitude tier, median over kicks

| cell | A0 | tier ×A0 | n | Δλ_B(+300) % vs pre-600 med | net return cumsum(dxu)/amp @600 |
|---|---|---|---|---|---|
| CPK_sgd_b8 | 1.06e-02 | 2 | 1 | +1.4% | +0.91 |
| CPK_sgd_b8 | 1.06e-02 | 8 | 1 | +2.4% | -0.71 |
| CPK_sgd_b8 | 1.06e-02 | 32 | 1 | +nan% | -3965.34 |
| CPK_sgdm_b8 | 4.83e-03 | 2 | 1 | -0.3% | -1.71 |
| CPK_sgdm_b8 | 4.83e-03 | 8 | 1 | +1.1% | +4.55 |
| CPK_sgdm_b8 | 4.83e-03 | 32 | 1 | -3.5% | +1.35 |
| CPK_sgdm_b8 | 4.83e-03 | 128 | 1 | +nan% | -575.85 |
| CPK_nest_b8 | 8.58e-03 | 2 | 1 | -5.7% | -2.81 |
| CPK_nest_b8 | 8.58e-03 | 8 | 1 | +2.4% | -0.01 |
| CPK_nest_b8 | 8.58e-03 | 32 | 1 | -1.4% | -1.25 |
| CPK_nest_b8 | 8.58e-03 | 128 | 1 | +nan% | +248.32 |
| CPK_adam_b8 | 2.09e-03 | 2 | 5 | +6.0% | -14.51 |
| CPK_adam_b8 | 2.09e-03 | 8 | 5 | -0.1% | +3.88 |
| CPK_adam_b8 | 2.09e-03 | 32 | 5 | -4.6% | -0.91 |
| CPK_adam_b8 | 2.09e-03 | 128 | 5 | +16.7% | -0.17 |
| CPK_muon_b8 | 7.87e-04 | 2 | 5 | -2.7% | -1.98 |
| CPK_muon_b8 | 7.87e-04 | 8 | 5 | +1.4% | +4.84 |
| CPK_muon_b8 | 7.87e-04 | 32 | 5 | +1.4% | -0.41 |
| CPK_muon_b8 | 7.87e-04 | 128 | 5 | +0.8% | -1.16 |
