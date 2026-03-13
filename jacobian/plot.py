# %% Settings
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

RESULTS_ROOT = Path("results")
TAU = 500    # rolling average window (int or None)
RHO_COL = 'rho_10'   # which rho column to plot


def _smooth(series, w):
    if w is not None:
        return series.rolling(w, min_periods=1).mean()
    return series


# Load all runs
_runs = []
for run_folder in sorted(RESULTS_ROOT.iterdir()):
    if not run_folder.is_dir():
        continue
    results_file = run_folder / 'results.txt'
    if not results_file.exists():
        continue
    try:
        run_df = pd.read_csv(results_file, comment='#')
    except Exception:
        continue
    _runs.append((run_folder.name, run_df))

print(f"Loaded {len(_runs)} run(s) from {RESULTS_ROOT}")

# %% Plot: rho vs step
fig, ax = plt.subplots(figsize=(8, 5))
for name, df in _runs:
    if RHO_COL not in df.columns:
        continue
    data = df[['step', RHO_COL]].dropna()
    if data.empty:
        continue
    ax.scatter(data['step'], _smooth(data[RHO_COL], TAU),
               marker='o', s=4, label=name)
ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='ρ = 1')
ax.set_xlabel('step')
ax.set_ylabel(f'g_τ  ({RHO_COL})')
ax.set_title(f'Lyapunov growth  (smoothing τ = {TAU})')
ax.legend(fontsize=7, loc='upper right')
plt.tight_layout()
plt.show()

# %%
