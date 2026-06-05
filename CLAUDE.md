# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Empirical study of the **Edge of Stochastic Stability (EoS)** — training neural nets (MLP/CNN/ViT on 8192 CIFAR-10 samples, plus an `SSTTransformer` on 8192 SST-2 samples for the language extension) while tracking the top Hessian eigenvalue, batch sharpness, and projections of the parameter vector onto various dynamics-relevant directions inside a tracking window near EoS.

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

### SST-2 transformer (language task)

`SSTTransformer` (`utils/nets.py`) and `prepare_sst2` (`utils/data.py`) are copied verbatim from `marcwalden1/edge-of-stochastic-stability` to reproduce its bimodality figures inside this repo's tracking pipeline. Architecture: 2 bidirectional encoder blocks, d_model=64, n_heads=2, vocab=33278 (bert-base-uncased), seq_len=64, masked mean-pool, zero-init head. **Token embedding is frozen** (`requires_grad=False`) — sparse per-batch gradients to embeddings would invalidate the batch-sharpness estimator. With `use_bert_emb=True` (default), the frozen `tok_emb` is loaded from a precomputed BERT-SVD-projected cache at `/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/bert_emb_proj64.pt`. Driver: `marc_files/sweep_optimizers/sweep_sst.sh` (10-cell LR sweep at batch=16, SGD-Momentum β=0.5, 200k steps, tracking the final 10k).

### Qwen2.5-0.5B-Instruct (production LLM task)

`QwenClassifier` (`utils/nets.py`) wraps a HuggingFace Qwen2.5-0.5B-Instruct model with a zero-init linear classification head for SST-2 binary classification. Key design: `embed_tokens` frozen (same reason as SSTTransformer), `attn_implementation="eager"` disables flash attention so second-order autograd (HVP) works, mean pool over non-padding positions for the classification representation. Dataset: `qwen_sst2` uses Qwen's tiktoken-based tokenizer (vocab 151936), pads with EOS token (id 151643), returns int64 {0,1} labels for CE loss. **`qwen_model_path`** config var (default `…/models/Qwen2.5-0.5B-Instruct`) controls the local model dir; override with `--qwen_model_path`. Model must be downloaded first: `bash marc_files/sweep_optimizers/download_qwen_instruct.sh` (login node only). Sweep scripts: `probe_qwen_bimodality.sh` (6 LRs × 10k steps to find EoSS), `sweep_qwen_bimodality.sh` (10 LRs × 50k steps, track 40k–50k). Loss: CE + label_smoothing=0.1 (same as CE SST experiments). Results under `Qwen_SST_sweep/` in `$RESULTS`.

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

- **CNN needs `num_data=16384`** to stay at EoS through the standard 30k–35k tracking window. With the default `num_data=8192`, the CNN cell (SGD lr=0.02 b=32) drives loss to ~0 well before step 30k and the tracking window catches a post-convergence regime where the loss is locally quadratic and the failure-mode signal vanishes. Other archs (MLP, ViT, SST) reach EoS fine at 8192. Always set `--num_data 16384` for CNN runs that depend on hitting the tracking window at EoS.
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
| `bert_emb_proj64.pt` (BERT SVD cache for SSTTransformer) | HF dataset `marcwalden/eoss-results` (root) |
| This code repo | GitHub `lilysun004/eoss` |
