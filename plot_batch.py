# %% Settings
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from utils.storage import parse_folder_name

RESULTS_ROOT = Path("results/0318_alloptimizers_targeted_finebatch")
N            = 100   # number of tail (non-NaN) GBS values to average per run

# List of (optimizer_name, lr) series to plot — one line each
SERIES = [
    ('SGD',          0.007),
    ('SGD-Momentum', 0.004),
    ('SGD-Nesterov', 0.004),
    ('Adam',         1e-3),
    ('Muon',         0.002),
]

# ── Helper: load most-recent folder per batch_size for one (opt, lr) ──────────
def _load_series(optimizer, lr):
    """Return (batch_sizes, means, stderrs) for the given optimizer/lr."""
    by_bs = {}
    for folder in sorted(RESULTS_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder.name)
        if not parsed:
            continue
        if parsed.get('optimizer_name') != optimizer:
            continue
        if parsed.get('lr') != lr:
            continue
        bs = parsed.get('batch_size')
        if bs is None:
            continue
        if not (folder / 'results.txt').exists():
            continue
        by_bs[bs] = folder   # sorted() → last write wins = most recent

    batch_sizes, means, stderrs = [], [], []
    for bs in sorted(by_bs):
        try:
            df = pd.read_csv(by_bs[bs] / 'results.txt', comment='#')
        except Exception:
            continue
        if 'GBS' not in df.columns:
            continue
        vals = df['GBS'].dropna().values
        if len(vals) == 0:
            continue
        tail   = vals[-N:]
        mean   = tail.mean()
        stderr = tail.std(ddof=1) / np.sqrt(len(tail)) if len(tail) > 1 else 0.0
        batch_sizes.append(bs)
        means.append(mean)
        stderrs.append(stderr)

    return batch_sizes, means, stderrs

# %% Plot
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.get_cmap('tab10')

for i, (opt, lr) in enumerate(SERIES):
    batch_sizes, means, stderrs = _load_series(opt, lr)
    if not batch_sizes:
        print(f"  no data: {opt} lr={lr:g}")
        continue
    label = f"{opt} lr={lr:g}"
    color = cmap(i % 10)
    ax.errorbar(
        batch_sizes,
        means,
        yerr=stderrs,
        fmt='o-',
        capsize=4,
        color=color,
        ecolor=color,
        linewidth=1.5,
        markersize=6,
        label=label,
    )
    for bs, m, se in zip(batch_sizes, means, stderrs):
        print(f"  {opt} lr={lr:g}  bs={bs:5d}  mean={m:.4f}  se={se:.4f}")

ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('batch size (log₂ scale)')
ax.set_ylabel('GBS (tail mean)')
ax.set_title(f'GBS vs batch size  (last {N} values per run)')
ax.legend()
ax.set_ylim(-0.5, 3)
plt.tight_layout()
plt.show()

# %%
