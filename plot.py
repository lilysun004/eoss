# %% Settings
import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.storage import parse_folder_name

RESULTS_ROOT = Path("results/0318_alloptimizers_targeted_finebatch")
WINDOW = None  # EMA alpha for smoothing (float in (0,1]) or None

# Filter runs: only plot runs matching ALL key-value pairs.
# FILTER = {'optimizer_name': 'SGD', 'lr': 0.007}
FILTER = {'optimizer_name': 'SGD-Momentum', 'lr': 0.004}
# FILTER = {'optimizer_name': 'SGD-Nesterov', 'lr': 0.004}
# FILTER = {'optimizer_name': 'Adam', 'lr': 1e-3}
# FILTER = {'optimizer_name': 'Muon', 'lr': 0.002}

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

# Load all matching runs once
_runs = []
_run_folders = []
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
    _run_folders.append(run_folder)

order = sorted(range(len(_runs)), key=lambda i: _runs[i][0].get('lr', 0))
_runs = [_runs[i] for i in order]
_run_folders = [_run_folders[i] for i in order]

figsize = (8, 5)

def _run_label(parsed):
    opt = parsed.get('optimizer_name', '?')
    lr = parsed.get('lr', '?')
    bs = parsed.get('batch_size', '?')
    lr_str = f"{lr:g}" if isinstance(lr, float) else str(lr)
    return f"{opt} lr={lr_str} b={bs}"

def _has_data(run_df, col):
    return col in run_df.columns and run_df[col].notna().any()

def _build_color_map(color_by):
    """Return {value: color} for all runs, keyed by batch_size or lr."""
    key = 'batch_size' if color_by == 'BATCHSIZE' else 'lr'
    vals = sorted(set(p.get(key) for p, _ in _runs if p.get(key) is not None))
    cmap = plt.get_cmap('tab10')
    return {v: cmap(i % 10) for i, v in enumerate(vals)}, key

def _gbs_series(run_df, gbs_col, a_col, b_col, gbs_mode):
    """Return (steps, values) for a GBS variant."""
    if gbs_mode == 'outside':
        if not _has_data(run_df, gbs_col):
            return None
        data = run_df[['step', gbs_col]].dropna()
        return data['step'], data[gbs_col]
    else:  # inside: B / -A
        needed = ['step', a_col, b_col]
        if not all(c in run_df.columns for c in needed):
            return None
        data = run_df[needed].dropna()
        if data.empty:
            return None
        values = data[b_col] / (-data[a_col])
        return data['step'], values

def _plot_gbs(gbs_col, a_col, b_col, ylabel, title, *, gbs_mode, color_by):
    color_map, color_key = _build_color_map(color_by)
    mode_label = 'outside' if gbs_mode == 'outside' else 'inside (B/-A)'
    color_label = 'batch size' if color_by == 'BATCHSIZE' else 'lr'
    fig, ax = plt.subplots(figsize=figsize)
    seen = set()
    for parsed, run_df in _runs:
        result = _gbs_series(run_df, gbs_col, a_col, b_col, gbs_mode)
        if result is None:
            continue
        steps, values = result
        val = parsed.get(color_key)
        color = color_map.get(val, 'gray')
        val_str = f"{val:g}" if isinstance(val, float) else str(val)
        lbl = f"{color_label}={val_str}" if val not in seen else None
        seen.add(val)
        ax.scatter(steps, _smooth_probe(values), marker='o', s=4, color=color, label=lbl)
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1, 5)
    ax.set_title(_filter_title(f'{title} [{mode_label}, color={color_label}]', FILTER))
    ax.legend()
    plt.tight_layout()
    return fig, ax

def _plot_generic(col, ylabel, title, *, color_by, yscale='linear'):
    color_map, color_key = _build_color_map(color_by)
    color_label = 'batch size' if color_by == 'BATCHSIZE' else 'lr'
    fig, ax = plt.subplots(figsize=figsize)
    seen = set()
    for parsed, run_df in _runs:
        if not _has_data(run_df, col):
            continue
        data = run_df[['step', col]].dropna()
        val = parsed.get(color_key)
        color = color_map.get(val, 'gray')
        val_str = f"{val:g}" if isinstance(val, float) else str(val)
        lbl = f"{color_label}={val_str}" if val not in seen else None
        seen.add(val)
        ax.scatter(data['step'], _smooth_probe(data[col]), marker='o', s=4, color=color, label=lbl)
    ax.set_yscale(yscale)
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.set_title(_filter_title(f'{title} [color={color_label}]', FILTER))
    ax.legend()
    plt.tight_layout()
    return fig, ax

def _plot_dist_mean(quantity_fn, ylabel, title, *, color_by):
    """Plot mean of a per-probe-batch quantity from distributions.npz over time."""
    color_map, color_key = _build_color_map(color_by)
    color_label = 'batch size' if color_by == 'BATCHSIZE' else 'lr'
    fig, ax = plt.subplots(figsize=figsize)
    seen = set()
    for i, (parsed, _) in enumerate(_runs):
        dfile = _run_folders[i] / 'distributions.npz'
        if not dfile.exists():
            continue
        d = np.load(dfile)
        steps = d['steps']
        vals = quantity_fn(d).mean(axis=1)
        val = parsed.get(color_key)
        color = color_map.get(val, 'gray')
        val_str = f"{val:g}" if isinstance(val, float) else str(val)
        lbl = f"{color_label}={val_str}" if val not in seen else None
        seen.add(val)
        ax.scatter(steps, _smooth_probe(pd.Series(vals)), marker='o', s=4, color=color, label=lbl)
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.set_title(_filter_title(title, FILTER))
    ax.legend()
    plt.tight_layout()
    return fig, ax

# %% Plot 1: GBS
GBS_MODE = 'outside'   # 'outside' or 'inside'
COLOR_BY = 'BATCHSIZE' # 'BATCHSIZE' or 'LEARNINGRATE'
fig, ax = _plot_gbs('GBS', 'A', 'B', 'GBS', 'GBS', gbs_mode=GBS_MODE, color_by=COLOR_BY)
plt.show()

# %% Plot 2: GBS_u (per-batch top eigenvector)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs('GBS_u', 'A_u', 'B_u', 'GBS_u', 'GBS_u', gbs_mode=GBS_MODE, color_by=COLOR_BY)
plt.show()

# %% Plot 3: GBS_ufull (full-batch eigenvector)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs('GBS_ufull', 'A_ufull', 'B_ufull', 'GBS_ufull', 'GBS_ufull', gbs_mode=GBS_MODE, color_by=COLOR_BY)
plt.show()

# %% Plot 4: GBS_g (gradient direction)
GBS_MODE = 'outside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs('GBS_g', 'A_g', 'B_g', 'GBS_g', 'GBS_g', gbs_mode=GBS_MODE, color_by=COLOR_BY)
plt.show()

# %% Plot 5: Batch sharpness
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_generic('batch_sharpness', 'batch sharpness', 'Batch Sharpness', color_by=COLOR_BY)
plt.show()

# %% Plot 6: SBS (E[sHs/||s||^2])
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_generic('SBS', 'SBS', 'SBS = E[sHs/‖s‖²]', color_by=COLOR_BY)
# ax.set_ylim(-0.5, 1)
plt.show()

# %% Plot 7: Full-batch loss
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_generic('full_loss', 'loss', 'Full-batch Loss', color_by=COLOR_BY, yscale='log')
plt.show()

# %% Plot 8: mean cos(s, g) over time
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['g_dot_s'] / (d['gnorms'] * d['snorms']), 'E[cos(s,g)]', 'Mean cos(s, g)', color_by=COLOR_BY)
plt.show()

# %% Plot 9: E[cos(s, u)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['s_dot_u'] / d['snorms'], 'E[cos(s,u)]', 'Mean cos(s, u)', color_by=COLOR_BY)
plt.show()

# %% Plot 10: E[cos(g, u)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['g_dot_u'] / d['gnorms'], 'E[cos(g,u)]', 'Mean cos(g, u)', color_by=COLOR_BY)
plt.show()

# %% Plot 11: E[cos(s, g_full)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['s_dot_gfull'] / (d['snorms'] * d['gnorm_full'][:, None]), 'E[cos(s,g_full)]', 'Mean cos(s, g_full)', color_by=COLOR_BY)
plt.show()

# %% Plot 12: E[cos(s, u_full)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['s_dot_ufull'] / d['snorms'], 'E[cos(s,u_full)]', 'Mean cos(s, u_full)', color_by=COLOR_BY)
plt.show()

# %% Plot 13: E[cos(g, g_full)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['g_dot_gfull'] / (d['gnorms'] * d['gnorm_full'][:, None]), 'E[cos(g,g_full)]', 'Mean cos(g, g_full)', color_by=COLOR_BY)
plt.show()

# %% Plot 14: E[||s||^2]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['snorms'] ** 2, 'E[‖s‖²]', 'Mean ‖s‖²', color_by=COLOR_BY)
plt.show()

# %% Plot 15: E[||g||^2]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_dist_mean(lambda d: d['gnorms'] ** 2, 'E[‖g‖²]', 'Mean ‖g‖²', color_by=COLOR_BY)
plt.show()

_GBS_AB_COLS = {
    'GBS':       ('A',       'B'),
    'GBS_u':     ('A_u',     'B_u'),
    'GBS_g':     ('A_g',     'B_g'),
    'GBS_ufull': ('A_ufull', 'B_ufull'),
}

def _plot_gbs_over_cos(gbs_col, cos_fn, cos_label, *, color_by, gbs_mode='outside'):
    """Plot {gbs_col} / -mean_cos where cos_fn(d) returns the (N,128) cosine array."""
    color_map, color_key = _build_color_map(color_by)
    color_label = 'batch size' if color_by == 'BATCHSIZE' else 'lr'
    mode_label = 'outside' if gbs_mode == 'outside' else 'inside (B/-A)'
    fig, ax = plt.subplots(figsize=figsize)
    seen = set()
    for i, (parsed, run_df) in enumerate(_runs):
        dfile = _run_folders[i] / 'distributions.npz'
        if not dfile.exists():
            continue
        d = np.load(dfile)
        mean_cos = cos_fn(d).mean(axis=1)
        if gbs_mode == 'outside':
            csv_df = run_df[['step', gbs_col]].dropna()
            merged = pd.DataFrame({'step': d['steps'], 'mean_cos': mean_cos})
            merged = merged.merge(csv_df, on='step', how='inner')
            if merged.empty:
                continue
            gbs_vals = merged[gbs_col].values
        else:
            a_col, b_col = _GBS_AB_COLS[gbs_col]
            csv_df = run_df[['step', a_col, b_col]].dropna()
            merged = pd.DataFrame({'step': d['steps'], 'mean_cos': mean_cos})
            merged = merged.merge(csv_df, on='step', how='inner')
            if merged.empty:
                continue
            gbs_vals = merged[b_col].values / (-merged[a_col].values)
        ratio = gbs_vals / (-merged['mean_cos'].values)
        val = parsed.get(color_key)
        color = color_map.get(val, 'gray')
        val_str = f"{val:g}" if isinstance(val, float) else str(val)
        lbl = f"{color_label}={val_str}" if val not in seen else None
        seen.add(val)
        ax.scatter(merged['step'], ratio, marker='o', s=4, color=color, label=lbl)
    ax.set_xlabel('step')
    ax.set_ylabel(f'{gbs_col} / -mean {cos_label}')
    ax.set_title(_filter_title(f'{gbs_col} / -mean {cos_label} [{mode_label}]', FILTER))
    ax.legend()
    plt.tight_layout()
    return fig, ax

# %% Plot 16: GBS_g / -mean cos(s_B, g_B)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs_over_cos('GBS_g',
    lambda d: d['g_dot_s'] / (d['gnorms'] * d['snorms']),
    'cos(s_B, g_B)', color_by=COLOR_BY, gbs_mode=GBS_MODE)
ax.set_ylim(-1,8)
plt.show()

# %% Plot 17: GBS_g / -mean cos(s_B, g_full)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs_over_cos('GBS_g',
    lambda d: d['s_dot_gfull'] / (d['snorms'] * d['gnorm_full'][:, None]),
    'cos(s_B, g_full)', color_by=COLOR_BY, gbs_mode=GBS_MODE)
ax.set_ylim(-1,8)
plt.show()

# %% Plot 18: GBS_g / -mean cos(g_B, g_full)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs_over_cos('GBS_g',
    lambda d: d['g_dot_gfull'] / (d['gnorms'] * d['gnorm_full'][:, None]),
    'cos(g_B, g_full)', color_by=COLOR_BY, gbs_mode=GBS_MODE)
ax.set_ylim(-5,2)
plt.show()

# %% Plot 19: GBS / -mean cos(s_B, g_B)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs_over_cos('GBS',
    lambda d: d['g_dot_s'] / (d['gnorms'] * d['snorms']),
    'cos(s_B, g_B)', color_by=COLOR_BY, gbs_mode=GBS_MODE)
ax.set_ylim(-1,5)
plt.show()

# %% Plot 20: GBS / -mean cos(s_B, g_full)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs_over_cos('GBS',
    lambda d: d['s_dot_gfull'] / (d['snorms'] * d['gnorm_full'][:, None]),
    'cos(s_B, g_full)', color_by=COLOR_BY, gbs_mode=GBS_MODE)
ax.set_ylim(-1,8)
plt.show()

# %% Plot 21: GBS / -mean cos(g_B, g_full)
GBS_MODE = 'inside'
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_gbs_over_cos('GBS',
    lambda d: d['g_dot_gfull'] / (d['gnorms'] * d['gnorm_full'][:, None]),
    'cos(g_B, g_full)', color_by=COLOR_BY, gbs_mode=GBS_MODE)
# ax.set_ylim(...)
plt.show()

# %% Plot 23: GBS_cos_sBgB = E[sHs / (-gs |cos(s_B, g_B)|)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_generic('GBS_cos_sBgB', 'GBS_cos_sBgB', 'GBS_cos_sBgB = E[sHs / (-gs|cos(s,g)|)]', color_by=COLOR_BY)
ax.set_ylim(-1, 8)
plt.show()

# %% Plot 24: GBS_cos_sBgfull = E[sHs / (-gs |cos(s_B, g_full)|)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_generic('GBS_cos_sBgfull', 'GBS_cos_sBgfull', 'GBS_cos_sBgfull = E[sHs / (-gs|cos(s,g_full)|)]', color_by=COLOR_BY)
ax.set_ylim(-1, 8)
plt.show()

# %% Plot 25: GBS_cos_gBgfull = E[sHs / (-gs |cos(g_B, g_full)|)]
COLOR_BY = 'BATCHSIZE'
fig, ax = _plot_generic('GBS_cos_gBgfull', 'GBS_cos_gBgfull', 'GBS_cos_gBgfull = E[sHs / (-gs|cos(g,g_full)|)]', color_by=COLOR_BY)
ax.set_ylim(-1,8)
plt.show()

# %% Plot 26: cos(s, g) distribution over time (Plotly animated histogram)
import plotly.graph_objects as go

DIST_FILTER = {'lr': 0.008, 'batch_size': 8}  # filter within already-loaded _runs

def _find_dist_run(filt):
    for i, (parsed, _) in enumerate(_runs):
        if all(parsed.get(k) == v for k, v in filt.items()):
            return i
    raise ValueError(f"No run matched DIST_FILTER={filt}. Available: {[p for p,_ in _runs]}")

_dist_idx  = _find_dist_run(DIST_FILTER)
_dist_file = _run_folders[_dist_idx] / 'distributions.npz'
_dist = np.load(_dist_file)

_dist_steps = _dist['steps']
_cos_sg     = _dist['g_dot_s'] / (_dist['gnorms'] * _dist['snorms'])  # (N, 128)
_mean_cos   = _cos_sg.mean(axis=1)                                     # (N,)

print(f"Run: {_run_folders[_dist_idx].name}")
print(f"Steps recorded: {_dist_steps}")
print(f"mean cos(s,g) per step:\n{_mean_cos}")

_xmin, _xmax = -1.0, 1.0
_frames = [
    go.Frame(
        data=[go.Histogram(x=_cos_sg[i].tolist(), xbins=dict(start=_xmin, end=_xmax, size=(_xmax - _xmin) / 40))],
        name=str(int(_dist_steps[i])),
    )
    for i in range(len(_dist_steps))
]

_fig26 = go.Figure(
    data=[go.Histogram(x=_cos_sg[0].tolist(), xbins=dict(start=_xmin, end=_xmax, size=(_xmax - _xmin) / 40))],
    frames=_frames,
    layout=go.Layout(
        title=f'cos(s, g) distribution over time — {_run_folders[_dist_idx].name}',
        xaxis=dict(title='cos(s, g)', range=[_xmin, _xmax]),
        yaxis_title='count',
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(int(s))], dict(mode='immediate', frame=dict(duration=0))], label=str(int(s)))
                   for s in _dist_steps],
            currentvalue=dict(prefix='step: '),
        )],
        updatemenus=[dict(
            type='buttons',
            buttons=[
                dict(label='Play',  method='animate', args=[None, dict(frame=dict(duration=300), fromcurrent=True)]),
                dict(label='Pause', method='animate', args=[[None], dict(mode='immediate')]),
            ],
        )],
    ),
)
_fig26.show()

# %%
