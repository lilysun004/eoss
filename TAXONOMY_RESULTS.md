# TAXONOMY_RESULTS.md — blind taxonomy test, DHB + RMSProp (2026-09-07)

Predictions & rules: analysis/TAXONOMY_PREREG.md (committed before runs). DATA ONLY.

| cell | lr | κ_B late | κ_full | gap | GBS | drift | status | notes |
|---|---|---|---|---|---|---|---|---|
| TX_dhb_b8_s0 | 0.01 | 3.223 | 1.094 | 2.129 | 0.32 | +0.02 | done |  |
| TX_dhb_b16_s0 | 0.01 | 4.754 | 2.693 | 2.061 | 0.49 | +0.03 | done |  |
| TX_dhb_b2048_s0 | 0.05 | 37.930 | 37.965 | -0.035 | 1.97 | +0.00 | done |  |
| TX_rms_b8_s0 | 0.001 | 3.005 | 0.754 | 2.251 | 1.94 | +0.21 | done |  |
| TX_rms_b16_s0 | 0.001 | 2.887 | 1.072 | 1.815 | 1.78 | +0.17 | done |  |
| TX_rms_b2048_s0 | 0.001 | 1.919 | 1.901 | 0.019 | 1.90 | -0.02 | done |  |
| TX_dhb_kick_b8_s0 | 0.01 | 3.102 | 1.052 | 2.050 | 0.30 | -0.01 | done | kicks=20 survived-all |
| TX_dhb_swap_b8_s0 | 0.01 | 1.413 | 1.262 | 0.151 | 0.00 | -0.53 | done | pre_lf(lr·λ)=1.119 post_lam=144 |
| TX_rms_kick_b8_s0 | 0.001 | 2.473 | 0.745 | 1.728 | 1.64 | +0.05 | done | kicks=20 survived-all |
| TX_rms_swap_b8_s0 | 0.001 | 1.983 | 1.717 | 0.267 | 1.90 | -0.11 | done | pre_lf(lr·λ)=0.809 post_lam=1959 |
