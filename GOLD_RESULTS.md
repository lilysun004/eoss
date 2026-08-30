# GOLD_RESULTS.md — kappa_spec AND GBS on the same run (2026-08-30)

**These numbers SUPERSEDE every earlier kappa_spec / GBS table** (KSPEC_RESULTS.md, RESULTS_MASTER.md §1/§3, SUMMARY.md, HANDOFF.md, kspec_results/*.json). Earlier tables mixed seeds, budgets and runner eras; here each cell is ONE run and both quantities are read from the SAME `dense.npz` over the SAME analysis window.

**Analysis window (pre-registered 2026-08-30 11:15, before any gold cell finished).** Primary numbers use the *healthy prefix* [u0_at, death_step): a step is healthy iff |dxu/su − 1| ≤ 0.05 (the applied step along the top mode equals the intended step — a float32-fidelity check, no optimizer knowledge); death_step = first sustained failure (≥100 unhealthy in 200 steps). Motivation: `analysis/HB_B2048_GBS_PROBE.md` — heavy-ball b2048 reaches loss ~1e-9 by step ~6000, the update drops below float32 half-ulp, and per-step GBS becomes rounding noise (0.33) while κ_spec's PSD-weighted integral hides it. Full-window numbers are kept in the table (`κ_spec full / GBS full`) so nothing is hidden; cells with < 512 healthy steps are censored, listed, never dropped.

**Protocol.** mlp_s (789k), CIFAR-10 num_data=2048, MSE, CPU, seed 0, stride 1 everywhere. Optimizers: SGD, SGD-Momentum β=0.9, SGD-Nesterov β=0.9, Adam β₁=0.9, Muon momentum=0.95. Batches 8/32/128/512/2048 with max_steps/u0_at = 30000/8000, 30000/8000, 20000/5000, 16000/4000, 16000/4000 (== kspec_ladder.LADDER). lr per cell from a liveness-bisect preflight (`results/kspec_gold/preflight.json`; live = not diverged, κ_late ≥ 0.4·2(1−β), step-norm slope > −0.02). Estimator: `experiments/kspec_estimator.py` (formula-free, grep-certified): κ_spec = median(λ_B)·∫|T̂(ω)| dμ(ω), T̂ = Welch S_gu,su/P_gu, μ = P_gu; GBS = sᵀH_Bs/(−gᵀs) per step (slow_sweep), median/IQR/mean over the identical mask. **No fitting, no interpretation** — the only statistic is the pre-registered secondary test (KSPEC_PREREG_ANNOTATIONS.md).

## 1. Full table (25 cells, priority order b2048 → b8)

| cell | lr | steps | κ_raw | r₁ | ω*/π | gain | **κ_spec** [CI] | drift | stat. | **GBS_med** [IQR] | GBS/κ_spec | κ_spec full / GBS full | death step (healthy %) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G_sgd_b2048_s0 | 0.01000 | 16000 | 2.007 | -1.00 | 0.99 | 1.0000 | **2.007** [2.007,2.007] | -0.016 | ✓ | **1.997** [1.973,2.023] | 0.995 | 2.007 / 1.997 | — (100%) |  |
| G_sgdm_b2048_s0 | 0.00650 | 16000 | 3.796 | -1.00 | 0.99 | 0.5264 | **1.998** [1.990,2.002] | 0.001 | ✓ | **2.000** [1.968,2.016] | 1.001 | 2.006 / 0.346 | 6517 (23%) |  |
| G_nest_b2048_s0 | 0.00650 | 16000 | 1.357 | -1.00 | 1.00 | 1.4760 | **2.003** [2.000,2.005] | -0.001 | ✓ | **2.000** [2.000,2.000] | 0.998 | 2.003 / 2.000 | — (100%) |  |
| G_adam_b2048_s0 | 0.00100 | 16000 | 34.187 | -0.96 | 0.97 | 0.0568 | **1.942** [1.936,1.945] | 0.034 | ✓ | **2.054** [1.814,2.278] | 1.058 | 1.942 / 2.054 | — (100%) |  |
| G_muon_b2048_s0 | 0.00100 | 16000 | 0.123 | -0.60 | 0.67 | 10.8055 | **1.333** [1.193,1.692] | -0.464 | ✗ | **2.002** [1.743,2.358] | 1.502 | 1.333 / 2.002 | — (100%) | nonstationary, halves-disagree |
| G_sgd_b512_s0 | 0.01000 | 16000 | 2.059 | -0.99 | 0.98 | 1.0000 | **2.059** [2.059,2.059] | -0.017 | ✓ | **1.974** [1.882,2.060] | 0.959 | 2.059 / 1.974 | — (100%) |  |
| G_sgdm_b512_s0 | 0.00800 | 16000 | 3.670 | -0.89 | 0.86 | 0.5491 | **2.015** [2.002,2.023] | 0.005 | ✓ | **1.978** [1.790,2.177] | 0.982 | 2.015 / 2.363 | 7490 (31%) |  |
| G_nest_b512_s0 | 0.00800 | 16000 | 1.392 | -0.99 | 0.98 | 1.4880 | **2.071** [2.066,2.075] | -0.002 | ✓ | **1.940** [1.810,2.053] | 0.937 | 2.071 / 1.862 | 12996 (77%) |  |
| G_adam_b512_s0 | 0.00100 | 16000 | 33.352 | -0.95 | 0.97 | 0.0587 | **1.959** [1.951,1.963] | 0.014 | ✓ | **2.022** [1.856,2.196] | 1.032 | 1.959 / 2.022 | — (100%) |  |
| G_muon_b512_s0 | 0.00100 | 16000 | 0.137 | -0.90 | 0.87 | 12.9465 | **1.779** [1.655,2.033] | -0.261 | ✗ | **1.587** [1.453,1.732] | 0.892 | 1.779 / 1.587 | — (100%) | nonstationary, halves-disagree |
| G_sgd_b128_s0 | 0.01000 | 20000 | 2.213 | -0.97 | 0.95 | 1.0000 | **2.213** [2.213,2.213] | -0.008 | ✓ | **1.915** [1.700,2.115] | 0.866 | 2.213 / 1.915 | — (100%) |  |
| G_sgdm_b128_s0 | 0.00600 | 20000 | 2.105 | -0.07 | 0.55 | 0.7160 | **1.507** [1.495,1.528] | 0.000 | ✓ | **1.473** [1.248,1.743] | 0.977 | 1.514 / 1.530 | 14548 (65%) |  |
| G_nest_b128_s0 | 0.00600 | 20000 | 1.467 | -0.95 | 0.95 | 1.5137 | **2.220** [2.218,2.226] | 0.005 | ✓ | **1.754** [1.478,2.015] | 0.790 | 2.220 / 1.722 | 18813 (93%) |  |
| G_adam_b128_s0 | 0.00100 | 20000 | 17.466 | -0.05 | 0.53 | 0.0676 | **1.181** [1.178,1.189] | -0.077 | ✓ | **1.338** [1.190,1.517] | 1.133 | 1.181 / 1.338 | — (100%) |  |
| G_muon_b128_s0 | 0.00100 | 20000 | 0.178 | -0.36 | 0.65 | 7.1201 | **1.267** [1.210,1.364] | 0.057 | ✓ | **0.563** [0.497,0.647] | 0.444 | 1.267 / 0.563 | — (100%) |  |
| G_sgd_b32_s0 | 0.01000 | 15263 | 2.766 | -0.08 | 0.92 | 1.0000 | **2.766** [2.766,2.766] | 0.012 | ✓ | **1.917** [1.589,2.317] | 0.693 | 2.766 / 1.917 | — (100%) | diverged |
| G_sgdm_b32_s0 | 0.00500 | 30000 | 0.620 | 0.28 | 0.40 | 0.8672 | **0.537** [0.537,0.550] | -0.048 | ✓ | **0.616** [0.525,0.746] | 1.147 | 0.537 / 0.616 | — (100%) |  |
| G_nest_b32_s0 | 0.00500 | 30000 | 0.870 | -0.29 | 0.63 | 1.6927 | **1.472** [1.463,1.482] | -0.020 | ✓ | **0.850** [0.726,1.011] | 0.577 | 1.472 / 0.850 | — (100%) |  |
| G_adam_b32_s0 | 0.00100 | 30000 | 8.501 | 0.17 | 0.45 | 0.0807 | **0.686** [0.669,0.699] | 0.038 | ✓ | **0.600** [0.508,0.721] | 0.875 | 0.686 / 0.600 | — (100%) |  |
| G_muon_b32_s0 | 0.00100 | 30000 | 0.184 | 0.24 | 0.40 | 2.0422 | **0.375** [0.369,0.384] | 0.130 | ✗ | **0.247** [0.227,0.269] | 0.659 | 0.375 / 0.247 | — (100%) | nonstationary |
| G_sgd_b8_s0 | 0.01000 | 30000 | 2.774 | -0.43 | 0.68 | 1.0000 | **2.774** [2.774,2.774] | 0.029 | ✓ | **1.847** [1.482,2.333] | 0.666 | 2.774 / 1.847 | — (100%) |  |
| G_sgdm_b8_s0 | 0.00200 | 30000 | 0.343 | 0.16 | 0.46 | 0.9610 | **0.329** [0.329,0.333] | -0.057 | ✓ | **0.294** [0.238,0.368] | 0.894 | 0.329 / 0.294 | — (100%) |  |
| G_nest_b8_s0 | 0.00200 | 30000 | 0.375 | 0.03 | 0.49 | 1.8164 | **0.681** [0.678,0.686] | -0.068 | ✓ | **0.455** [0.364,0.581] | 0.669 | 0.681 / 0.455 | — (100%) |  |
| G_adam_b8_s0 | 0.00100 | 30000 | 5.079 | 0.19 | 0.45 | 0.0762 | **0.387** [0.358,0.405] | 0.228 | ✗ | **0.406** [0.296,0.568] | 1.050 | 0.387 / 0.406 | — (100%) | nonstationary |
| G_muon_b8_s0 | 0.00100 | 30000 | 0.259 | 0.26 | 0.41 | 0.5077 | **0.131** [0.129,0.135] | -0.010 | ✓ | **0.132** [0.121,0.145] | 1.008 | 0.131 / 0.132 | — (100%) |  |

## 2. b2048 cells only (the disputed heavy-ball GBS-vs-κ_spec cell lives here)

| cell | lr | steps | κ_raw | r₁ | ω*/π | gain | **κ_spec** [CI] | drift | stat. | **GBS_med** [IQR] | GBS/κ_spec | κ_spec full / GBS full | death step (healthy %) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G_sgd_b2048_s0 | 0.01000 | 16000 | 2.007 | -1.00 | 0.99 | 1.0000 | **2.007** [2.007,2.007] | -0.016 | ✓ | **1.997** [1.973,2.023] | 0.995 | 2.007 / 1.997 | — (100%) |  |
| G_sgdm_b2048_s0 | 0.00650 | 16000 | 3.796 | -1.00 | 0.99 | 0.5264 | **1.998** [1.990,2.002] | 0.001 | ✓ | **2.000** [1.968,2.016] | 1.001 | 2.006 / 0.346 | 6517 (23%) |  |
| G_nest_b2048_s0 | 0.00650 | 16000 | 1.357 | -1.00 | 1.00 | 1.4760 | **2.003** [2.000,2.005] | -0.001 | ✓ | **2.000** [2.000,2.000] | 0.998 | 2.003 / 2.000 | — (100%) |  |
| G_adam_b2048_s0 | 0.00100 | 16000 | 34.187 | -0.96 | 0.97 | 0.0568 | **1.942** [1.936,1.945] | 0.034 | ✓ | **2.054** [1.814,2.278] | 1.058 | 1.942 / 2.054 | — (100%) |  |
| G_muon_b2048_s0 | 0.00100 | 16000 | 0.123 | -0.60 | 0.67 | 10.8055 | **1.333** [1.193,1.692] | -0.464 | ✗ | **2.002** [1.743,2.358] | 1.502 | 1.333 / 2.002 | — (100%) | nonstationary, halves-disagree |

## 3. Pre-registered secondary test: GBS_med vs κ_spec agreement (report only)

Registered prediction: slope ≈ 1 through the origin, both instruments = 2 at any binding edge and short by the same factor on sub-marginal cells.

| subset | n | Pearson corr | origin slope (GBS on κ_spec) |
|---|---|---|---|
| all | 25 | 0.897 | 0.886 |
| excl_heavyball | 20 | 0.878 | 0.867 |
| stationary_only | 21 | 0.906 | 0.869 |

## 4. Preflight (liveness bisect) record

| cell | accepted lr | clean | probes |
|---|---|---|---|
| sgd_b2048 | 0.01 | True | 0.01→live |
| sgdm_b2048 | 0.0065 | True | 0.0065→live |
| nest_b2048 | 0.0065 | True | 0.0065→live |
| adam_b2048 | 0.001 | True | 0.001→live |
| muon_b2048 | 0.001 | True | 0.001→live |
| sgd_b512 | 0.01 | True | 0.01→live |
| sgdm_b512 | 0.008 | True | 0.008→live |
| nest_b512 | 0.008 | True | 0.008→live |
| adam_b512 | 0.001 | True | 0.001→live |
| muon_b512 | 0.001 | True | 0.001→live |
| sgd_b128 | 0.01 | True | 0.01→live |
| sgdm_b128 | 0.006 | True | 0.006→live |
| nest_b128 | 0.006 | True | 0.006→live |
| adam_b128 | 0.001 | True | 0.001→live |
| muon_b128 | 0.001 | True | 0.001→live |
| sgd_b32 | 0.01 | True | 0.01→live |
| sgdm_b32 | 0.005 | True | 0.005→live |
| nest_b32 | 0.005 | True | 0.005→live |
| adam_b32 | 0.001 | True | 0.001→live |
| muon_b32 | 0.001 | True | 0.001→live |
| sgd_b8 | 0.01 | True | 0.01→live |
| sgdm_b8 | 0.002 | True | 0.002→live |
| nest_b8 | 0.002 | True | 0.002→live |
| adam_b8 | 0.001 | True | 0.001→live |
| muon_b8 | 0.001 | True | 0.001→live |

## 5. Files

- per-cell JSON: `kspec_results/gold/<cell>_kspec.json`; table: `kspec_results/gold/gold_table.csv`; agreement: `kspec_results/gold/agreement.json`
- raw runs: `results/kspec_gold/<cell>/dense.npz` (signed primitives gu/su/mu/dxu/gu0/su0/gbs/kappa/lam_batch/a_t …)
- driver: `experiments/gold_sweep.py`
