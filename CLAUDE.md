# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Empirical study of the **Edge of Stochastic Stability (EoS)** — training neural nets (mostly an MLP on 8192 CIFAR-10 samples) while tracking the top Hessian eigenvalue, batch sharpness, and projections of the parameter vector onto various dynamics-relevant directions inside a tracking window near EoS.

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

- New CLI flags: add a variable to `config.py`, thread it through `config.py`'s `train(...)` call, add the param to `train()` in `training.py`, and (if it affects tracking) to `ProjectionTracker.__init__`. The CLI parser picks it up automatically.
- New projections: extend the `proj_*` lists in `ProjectionTracker`, update `save()`'s key list, then update `PROJECTIONS_BASE` / `PROJECTIONS_PRECOND` in `plot_histograms.py` so histograms render.
- New optimizer: add to `utils/optimizer.py::create_optimizer`. If it has a diagonal preconditioner, expose `get_preconditioner_inv_sqrt(params) -> Tensor` on the wrapper so the tracker computes preconditioned-Hessian projections automatically.
