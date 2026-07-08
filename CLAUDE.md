# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Empirical study of the **Edge of Stochastic Stability (EoS)** — training neural nets (MLP/CNN on 8192 CIFAR-10 samples) while tracking the top Hessian eigenvalue, batch sharpness, and projections of the parameter vector onto various dynamics-relevant directions inside a tracking window near EoS.

## Running experiments

Two env vars must be set before running anything:

```bash
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
```

Entry point is `config.py`. Defaults live at the top of the file; override any of them from the CLI:

```bash
python config.py --optimizer_name Adam --lr 3e-5 --batch_size 128 \
    --optimizer_params "{'beta1': 0.9, 'beta2': 0.99}" \
    --steps 80000 --track_from 70000 --track_until 80000 \
    --fixed_u True --results_subfolder my_sweep
```

The CLI parser is homebrewed (`config.py:110-128`): any top-level variable in `config.py` is override-able via `--name value`; values are `ast.literal_eval`'d so dicts/bools/None must be quoted strings. Unknown keys raise.

### SLURM sweeps

Array jobs live in `marc_files/sweep_*.sh`, each submitting 5 batch sizes `(8 32 128 1024 8192)` for one optimizer. Submit with `sbatch marc_files/sweep_adam.sh`. Logs go to `marc_files/logs/`. Partition is `kempner_requeue` — jobs preempt and restart, so folder count can exceed run count; de-dup by picking the latest timestamp per `(optimizer, batch)`.

### Plotting

- `plot_histograms.py <run_folder>` — training curve + per-projection histograms for one run (layout auto-adjusts to 2×3 / 2×4 / 3×4 depending on how many projections are present, see `plot_histograms.py:164-171`).
- `marc_files/plot_optimizer_sweep.py`, `plot_batch_sweep.py` — cross-run summary grids.
- `visualization/plot_results.py` — legacy single-run plots.

## The three experiment families

Everything under `marc_files/` belongs to one of three experiment families. Each has its own results location, plotting script, and entry point.

### 1. Bimodality search

**Goal**: find runs where a per-step projection onto the top Hessian eigenvector (`proj_w_top5`) shows a genuine period-2 / bimodal histogram near EoS.

**Important context before running anything new**: an earlier LOBPCG ±u sign-ambiguity bug (now patched in `utils/measure.py::_run_lobpcg_with_operator` — eigenvectors are realigned to match the warm-start sign every step) made many runs *look* bimodal artificially. After the fix, a second, non-bug effect remains: when the top-2 Hessian eigenvalues are close, the top eigenvector direction itself can rotate step-to-step (`cos_sim_full_top5 < 1`), which can still make `proj_w_top5` look U-shaped/bimodal even though nothing is genuinely oscillating. **`proj_w_fixed_top5`** (projection onto the eigenvector frozen once at `track_from`, written when `--fixed_u True`) is immune to this and is the channel to trust for genuine bimodality claims. As of 2026-06-09, across many runs spanning CNN/MLP, all optimizers, multiple LRs, and both late- and early-training tracking windows, `proj_w_fixed_top5` has been unimodal in every run — see `marc_files/sweep_*.sh`, `marc_files/sweep_optimizers/`, `marc_files/sign_fix_rerun/`, and `marc_files/early_bimodality_scan*/` below for the full provenance.

- **Original sweeps** (generated the base data, before the sign fix): `marc_files/sweep_*.sh` (MLP, 5 batch sizes × optimizer, submit with `sbatch marc_files/sweep_adam.sh` etc.) and `marc_files/sweep_optimizers/sweep_cnn_*.sh` (CNN). Results land in `$RESULTS/<MLP_sweep|CNN_sweep|...>/`. Pre-rendered histogram PNGs/PDFs for these are checked into `marc_files/results_histograms{,_pdf}/` — **note these predate the sign fix**, treat any apparent bimodality in `proj_w_top5` there with the caveat above and check `proj_w_fixed_top5` / `cos_sim_full_top5` instead.
- **Sign-fix reruns**: `marc_files/sign_fix_rerun/gen_and_submit.py` regenerates sbatch scripts for the previously-"ambiguous" cells using the patched eigensolver, writing to `$RESULTS/<sweep>_signfix/`.
- **Early-training scans** (per advisor hint that bimodality, if real, should appear early in training at large LR): `marc_files/early_bimodality_scan/` (track window [300, 2300] of a 3000-step run) and `marc_files/early_bimodality_scan_8k/` (tracks the *entire* 8000-step run, `track_from=0, track_until=8000`). Each has a `gen_and_submit.py` that generates per-cell sbatch scripts (CNN: `num_data=16384`, MLP: `num_data=8192`, both `batch_size=32`, SGD at several LRs) and submits them; results land in `<dir>/results/early_scan_<model>[_8k]/`. CNN cells are expensive (~8.8s/measurement) — `track_stride` is set higher (8 vs 2) to keep wall-clock manageable; submit with `sbatch --partition=kempner_requeue` and a generous `-t` (CNN 8k cells need ~4-5h).
- **Plotting**: `python plot_histograms.py <run_folder>` for any single run (training curve + histograms for all `proj_*` channels + `cos_sim_full_top5`). `marc_files/plot_optimizer_sweep.py` / `plot_batch_sweep.py` for cross-run summary grids.

### 2. Tangent-drift tracking

**Goal**: characterize slow drift of θ along the Hessian-null/manifold-tangent (Cat 3) directions during the EoS tracking window, as a band/baseline against which the top-eigenvector projections are compared.

- Scripts: `marc_files/drift_results/scripts/{cnn,mlp}/run_*.sh` — one per optimizer per architecture, using `--cat3_m` > 1 (random directions orthogonal to the top-K subspace, see Conventions below) and `--fixed_u True`.
- Plotting: `marc_files/drift_results/plot_tangent_drift.py <run_folder>` — produces the median + IQR/decile drift-band figure, saved to `marc_files/drift_results/plots/`.
- Smoke test: `marc_files/drift_results/smoke_tangent_drift.sh`.

### 3. Curvature failure-mode (central-flow Fig. 29 reproduction)

**Goal**: test whether the loss along the top-eigenvector direction is well-approximated by its cubic Taylor expansion near EoS (Cohen et al., *Understanding Optimization in Deep Learning with Central Flows*, p.89/Fig.29) — relevant because higher-order curvature is what would produce a genuine period-2 attractor.

- Scripts: `marc_files/curvature_results/scripts/{cnn,mlp}/run_sgd.sh` (and CE-loss variants/probes under `scripts/mlp/`), enabling `--curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50`.
- **Results save locally, not to `$RESULTS`**: these scripts override `RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results` so `projections.npz`, `curvature_segment.npz`, and rendered figures land next to the script/plot code. Keep this override when adding new curvature scripts.
- Plotting: `marc_files/curvature_results/plot_curvature_failure.py <run_folder>` — Fig.29-style segment scans with Taylor overlays, along-eigvec scans, and aggregate signed-deviation panels.
- `marc_files/curvature_results/process.sh` de-dupes runs and batch-renders all three plot types into `marc_files/curvature_results/plots/`. `organize_for_drive.py` / `make_paper_bimodality_figures.py` build shareable figure trees (`bimodality_sweep/`, `drift_sweep/`, `curvature_failure_sweep/`, split by `MSE`/`CE` loss and architecture).

## Architecture

### Training loop (`training.py`)

`train()` is a single long function. Each step: forward/backward, optimizer step, then a `MeasurementRunner` dispatches optional measurements (λ_max, batch_sharpness, probe-batch GBS, distributions) gated by `utils/frequency.py` intervals. Inside the `[track_from, track_until]` window a `ProjectionTracker.pre_step()` is called **before** `optimizer.step()` and `post_step()` **after** — both hooks are required because projections compare `θ_{t+1}` against quantities computed at `θ_t`.

### Projection tracker (`utils/projection_tracker.py`)

Computes ⟨θ_t, v⟩ for up to 9 candidate directions `v`. The file's docstring enumerates the ordering; `save()` writes them to `projections.npz` in that exact order, with all-NaN arrays for slots that don't apply (e.g. preconditioned directions for SGD).

Two features worth knowing:
- **Warm-started power iteration is always on.** `EigenvectorCache(max_eigenvectors=1)` seeds each call with the previous step's top eigenvector. This preserves sign across steps (the ±u ambiguity of power iteration otherwise fakes bimodality in histograms).
- **`fixed_u=True`** additionally snapshots the first warm-started eigenvector at `track_from` and reuses it for all subsequent projections. Tests the hypothesis that the instability direction is approximately constant through the window.

Preconditioned quantities (slots 7–9) only populate when the optimizer wrapper exposes `get_preconditioner_inv_sqrt()` — currently Adam and RMSProp in `utils/optimizer.py`.

### Measurement scheduling (`utils/frequency.py`)

Centralized interval map. `config.py:157-163` sets each interval from the corresponding `*_every` config var, halving everything if `more_freq_measure=True`. Never hardcode step intervals inside the training loop — go through the frequency calculator.

### Storage (`utils/storage.py`)

`initialize_folders()` derives the run folder name from the args namespace in `config.py:199-204`. The naming convention — `YYYYMMDD_HHMM_SS_<optimizer>_lr<lr>_b<batch>[_<params>]` — is parsed back out by `plot_histograms.py::parse_run_title()` and the sweep plotters, so keep it stable.

Per-run artifacts in `$RESULTS/$results_subfolder/<run_folder>/`:
- `results.txt` — CSV of per-step scalars (step, loss, lmax, batch_sharpness, ...)
- `projections.npz` — projection arrays + `track_from`/`track_until` scalars
- `args.json` — full config snapshot

## Conventions

- **CNN needs `num_data=16384`** to stay at EoS through the standard 30k–35k tracking window. With the default `num_data=8192`, the CNN cell (SGD lr=0.02 b=32) drives loss to ~0 well before step 30k and the tracking window catches a post-convergence regime where the loss is locally quadratic and the failure-mode signal vanishes. MLP reaches EoS fine at 8192. Always set `--num_data 16384` for CNN runs that depend on hitting the tracking window at EoS.
- **Curvature failure-mode runs save under `marc_files/curvature_results/`**, NOT the default `$RESULTS` (`/n/holylabs/.../results/`). The sbatch scripts in `marc_files/curvature_results/scripts/{cnn,mlp,smoke_test}.sh` override `RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results` so per-run folders, `projections.npz`, `curvature_segment.npz`, and the rendered `curvature_failure.{pdf,png}` all land next to the script/plot code — easy to inspect, easy to commit selected artifacts. When adding new curvature-experiment scripts, keep that override.
- `top_k_track` (CLI flag, default 5) controls how many top Hessian eigvecs `ProjectionTracker` maintains per measurement step via warm-started LOBPCG. Larger values increase eigsolve cost per measurement roughly quadratically; saved-array names retain the `_top5` suffix but the second-axis size grows with `top_k_track`. Tracker also writes `proj_cat3` — projection of θ_t onto **m random directions** mutually orthogonal and each orthogonal to the top-K subspace at `track_from`. Used as guaranteed Cat 3 (manifold-tangent) samples for the slow-drift band visualization in `marc_files/drift_results/plot_tangent_drift.py`.
- `cat3_m` (CLI flag, default 1) sets m. Cost is ~m extra dot products per measurement step (microseconds) plus a one-time Gram-Schmidt at `track_from`. `proj_cat3` is saved as `(n_steps, m)`; legacy 1D arrays from older runs still plot via the back-compat path. Larger m gives the plot script median + IQR/decile shading instead of a single line.
- `curv_n_alphas` / `curv_n_betas` (CLI, default 0 = disabled) enable per-step curvature scans inside the tracking window: `u_mid^T H(α w_t + (1-α) w_{t+1}) u_mid` along the segment (Cohen et al. central flows p.89 / Fig 29) and `u_t^T H(w_t + β u_t) u_t` along the top eigvec at the iterate. `u_mid` is the top eigvec at the midpoint, recomputed via one LOBPCG warm-started from `u_t`. Same fixed subset is reused for u_mid LOBPCG and every scan HVP. Cadence: every `curv_every` tracked steps (default 50). Saved as `curvature_segment.npz` alongside `projections.npz`. Adds ~25 HVPs per measurement step.
- New plot script: `marc_files/curvature_results/plot_curvature_failure.py <run_folder>` — produces Fig 29-style segment scans with first-/second-order Taylor overlays, plus along-u scans and aggregate signed deviation panels.
- New CLI flags: add a variable to `config.py`, thread it through `config.py`'s `train(...)` call, add the param to `train()` in `training.py`, and (if it affects tracking) to `ProjectionTracker.__init__`. The CLI parser picks it up automatically.
- New projections: extend the `proj_*` lists in `ProjectionTracker`, update `save()`'s key list, then update `PROJECTIONS_BASE` / `PROJECTIONS_PRECOND` in `plot_histograms.py` so histograms render.
- New optimizer: add to `utils/optimizer.py::create_optimizer`. If it has a diagonal preconditioner, expose `get_preconditioner_inv_sqrt(params) -> Tensor` on the wrapper so the tracker computes preconditioned-Hessian projections automatically.

## Data backups (2026-06-04)
Harvard cluster access removed. All data backed up:
| What | Where |
|---|---|
| Experiment results (~7.5 GB — all `projections.npz`, `results.txt` per run) | HF dataset `marcwalden/eoss-results` |
| This code repo | GitHub `lilysun004/eoss` |
