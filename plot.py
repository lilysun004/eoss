# %% Settings
import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.storage import parse_folder_name

RESULTS_ROOT = Path("results/0311_optimizers")
WINDOW = None  # EMA alpha for smoothing (float in (0,1]) or None to disable

# Filter runs: only plot runs matching ALL key-value pairs.
FILTER = {'optimizer_name': 'SGD'}

def _matches_filter(folder_name, filt):
    if not filt:
        return True
    parsed = parse_folder_name(folder_name)
    return all(parsed.get(k) == v for k, v in filt.items())

def _filter_title(base, filt):
    if not filt:
        return base
    parts = ', '.join(f'{k}={v}' for k, v in filt.items())
    return f'{base} ({parts})'

def _ema(series, alpha):
    """Exponential moving average with given alpha, ignoring NaNs."""
    if alpha is None:
        return series
    result = series.copy().astype(float)
    s = None
    for i, v in enumerate(series):
        if pd.isna(v):
            result.iloc[i] = float('nan') if s is None else s
            continue
        s = v if s is None else (1 - alpha) * s + alpha * v
        result.iloc[i] = s
    return result

def _smooth_probe(series):
    return _ema(series, WINDOW)

def _smooth_actual(series):
    return _ema(series, TAU)

# Load all matching runs once
_runs = []
for run_folder in sorted(RESULTS_ROOT.iterdir()):
    if not run_folder.is_dir():
        continue
    if not _matches_filter(run_folder.name, FILTER):
        continue
    results_file = run_folder / 'results.txt'
    if not results_file.exists():
        continue
    try:
        run_df = pd.read_csv(results_file, comment='#')
    except Exception:
        continue
    parsed = parse_folder_name(run_folder.name)
    _runs.append((parsed, run_df))

_runs.sort(key=lambda x: x[0].get('lr', 0))

figsize = (8, 5)

def _run_label(parsed):
    opt = parsed.get('optimizer_name', '?')
    lr = parsed.get('lr', '?')
    bs = parsed.get('batch_size', '?')
    lr_str = f"{lr:g}" if isinstance(lr, float) else str(lr)
    return f"{opt} lr={lr_str} b={bs}"

def _has_data(run_df, col):
    return col in run_df.columns and run_df[col].notna().any()

# %% Plot 1: Probe GBS (out_probe and out_probe_u)
fig, ax = plt.subplots(figsize=figsize)
for parsed, run_df in _runs:
    run_lbl = _run_label(parsed)
    for col, marker, name in [('out_probe', 'o', 'GBS'), ('out_probe_u', '^', 'GBS_u')]:
        if not _has_data(run_df, col):
            continue
        data = run_df[['step', col]].dropna()
        ax.scatter(data['step'], _smooth_probe(data[col]), marker=marker, s=4, label=f"{name} ({run_lbl})")
ax.set_xlabel('step')
ax.set_ylabel('GBS')
ax.set_ylim(-1, 5)
ax.set_title(_filter_title('Probe GBS: E[B/-A] and E[B_u/-A_u]', FILTER))
ax.legend()
plt.tight_layout()
plt.show()

# %% Plot 2: Actual-batch GBS (out_actual and out_actual_u)
TAU = 0.001  # EMA alpha for actual-batch smoothing (float in (0,1]) or None to disable
fig, ax = plt.subplots(figsize=figsize)
for parsed, run_df in _runs:
    run_lbl = _run_label(parsed)
    for col, marker, name in [('out_actual', 'o', 'GBS'), ('out_actual_u', '^', 'GBS_u')]:
        if not _has_data(run_df, col):
            continue
        data = run_df[['step', col]].dropna()
        ax.scatter(data['step'], _smooth_actual(data[col]), marker=marker, s=4, label=f"{name} ({run_lbl})")
ax.set_xlabel('step')
ax.set_ylabel('GBS')
ax.set_title(_filter_title('Actual-batch GBS: B/-A and B_u/-A_u', FILTER))
ax.legend()
ax.set_ylim(0, 10)
plt.tight_layout()
plt.show()


# %% Plot 3: Full-batch GBS (out_full and out_full_u)
fig, ax = plt.subplots(figsize=figsize)
for parsed, run_df in _runs:
    run_lbl = _run_label(parsed)
    for col, marker, name in [('out_full', 'o', 'GBS'), ('out_full_u', '^', 'GBS_u')]:
        if not _has_data(run_df, col):
            continue
        data = run_df[['step', col]].dropna()
        ax.scatter(data['step'], _smooth_probe(data[col]), marker=marker, s=4, label=f"{name} ({run_lbl})")
ax.set_xlabel('step')
ax.set_ylabel('GBS')
ax.set_ylim(-1,3)
ax.set_title(_filter_title('Full-batch GBS: B/-A and B_u/-A_u (true)', FILTER))
ax.legend()
plt.tight_layout()
plt.show()

# %% Plot 4: Batch sharpness
fig, ax = plt.subplots(figsize=figsize)
for parsed, run_df in _runs:
    if 'batch_sharpness' not in run_df.columns:
        continue
    data = run_df[['step', 'batch_sharpness']].dropna()
    if not data.empty:
        ax.scatter(data['step'], _smooth_probe(data['batch_sharpness']), marker='o', s=4, label=_run_label(parsed))
ax.set_xlabel('step')
ax.set_ylabel('batch sharpness')
ax.set_title(_filter_title('Batch Sharpness', FILTER))
ax.legend()
plt.tight_layout()
plt.show()

# %% Plot 5: Full-batch loss
fig, ax = plt.subplots(figsize=figsize)
for parsed, run_df in _runs:
    if 'full_loss' not in run_df.columns:
        continue
    data = run_df[['step', 'full_loss']].dropna()
    if not data.empty:
        ax.scatter(data['step'], _smooth_probe(data['full_loss']), marker='o', s=4, label=_run_label(parsed))
ax.set_yscale('log')
ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_title(_filter_title('Full-batch Loss', FILTER))
ax.legend()
plt.tight_layout()
plt.show()

# %%
