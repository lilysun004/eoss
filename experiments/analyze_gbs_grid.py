"""
Analyze the runs produced by run_gbs_grid.py (results/gbs_grid_v3/<tag>/<run>/results.txt).
Produces two tables:
  (C) GBS_t distribution: mean / median / geomean / mu=E[log GBS] / sigma2=Var[log GBS]
      over the plateau window, per (optimizer,batch). Screens Mechanism C.
  (A) beta-sweep: GBS-plateau mean and deficit (2 - mean) vs beta for SGD-Momentum b8,
      and vs beta1 for Adam b8. Screens Mechanism A.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(_REPO, 'results', 'gbs_grid_v3')
OUT_DIR = os.path.join(_REPO, 'results', 'gbs_grid_v3', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# tag -> (optimizer label, batch, beta-or-None, is_base_grid)
TAGS = {
    "SGD_b8": ("SGD", 8, None, True), "SGD_b128": ("SGD", 128, None, True), "SGD_b2048": ("SGD", 2048, None, True),
    "SGDM09_b8": ("SGD-Momentum", 8, 0.9, True), "SGDM09_b128": ("SGD-Momentum", 128, 0.9, True), "SGDM09_b2048": ("SGD-Momentum", 2048, 0.9, True),
    "Adam_b8": ("Adam", 8, 0.9, True), "Adam_b128": ("Adam", 128, 0.9, True), "Adam_b2048": ("Adam", 2048, 0.9, True),
    "Muon_b8": ("Muon", 8, 0.9, True), "Muon_b128": ("Muon", 128, 0.9, True), "Muon_b2048": ("Muon", 2048, 0.9, True),
    "SGDM_b0_b8": ("SGD-Momentum", 8, 0.0, False), "SGDM_b03_b8": ("SGD-Momentum", 8, 0.3, False),
    "SGDM_b06_b8": ("SGD-Momentum", 8, 0.6, False), "SGDM_b099_b8": ("SGD-Momentum", 8, 0.99, False),
    "AdamB1_0_b8": ("Adam-b1", 8, 0.0, False), "AdamB1_05_b8": ("Adam-b1", 8, 0.5, False), "AdamB1_099_b8": ("Adam-b1", 8, 0.99, False),
}


def load_gbs(tag):
    runs = sorted(glob.glob(os.path.join(GRID_DIR, tag, '*', 'results.txt')))
    if not runs:
        return None
    df = pd.read_csv(runs[-1], comment='#')
    if 'GBS' not in df.columns:
        return None
    g = df[['step', 'GBS']].dropna()
    return g


def plateau_stats(g, back_frac=0.5):
    """Stats over the back `back_frac` of the GBS samples."""
    if g is None or len(g) < 6:
        return None
    n = len(g)
    vals = g['GBS'].values[n // 2:] if back_frac == 0.5 else g['GBS'].values[int(n * (1 - back_frac)):]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return None
    pos = vals[vals > 0]
    logv = np.log(pos) if len(pos) else np.array([])
    return dict(
        n=len(vals), n_pos=len(pos), n_nonpos=int((vals <= 0).sum()),
        mean=float(np.mean(vals)), median=float(np.median(vals)),
        geomean=float(np.exp(np.mean(logv))) if len(logv) else float('nan'),
        mu=float(np.mean(logv)) if len(logv) else float('nan'),
        sigma2=float(np.var(logv, ddof=1)) if len(logv) > 1 else float('nan'),
    )


def main():
    rows = []
    for tag, (opt, batch, beta, is_base) in TAGS.items():
        g = load_gbs(tag)
        st = plateau_stats(g)
        rows.append(dict(tag=tag, optimizer=opt, batch=batch, beta=beta, is_base=is_base,
                         **(st or dict(n=0))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'gbs_grid_analysis.csv'), index=False)

    # ---- Table C: distribution across base grid ----
    print("\n=== (C) GBS_t distribution over plateau, base grid (Mechanism C) ===")
    print(f"{'tag':14s} {'opt':13s} {'batch':>5s} {'n':>4s} {'mean':>7s} {'median':>7s} {'geomean':>7s} {'mu':>7s} {'sigma2':>7s} {'nonpos':>6s}")
    for _, r in df[df.is_base].iterrows():
        if r['n'] == 0:
            print(f"{r['tag']:14s} {r['optimizer']:13s} {r['batch']:5d}  NO DATA"); continue
        print(f"{r['tag']:14s} {r['optimizer']:13s} {int(r['batch']):5d} {int(r['n']):4d} "
              f"{r['mean']:7.3f} {r['median']:7.3f} {r['geomean']:7.3f} {r['mu']:7.3f} {r['sigma2']:7.3f} {int(r['n_nonpos']):6d}")

    # ---- Table A: beta sweep ----
    print("\n=== (A) beta-sweep GBS-plateau vs beta, SGD-Momentum b8 (Mechanism A) ===")
    sgdm = df[(df.optimizer == 'SGD-Momentum') & (df.batch == 8)].sort_values('beta')
    print(f"{'beta':>5s} {'mean_GBS':>9s} {'deficit(2-)':>11s} {'median':>7s} {'n':>4s}")
    for _, r in sgdm.iterrows():
        if r['n'] == 0:
            print(f"{r['beta']:5.2f}  NO DATA"); continue
        print(f"{r['beta']:5.2f} {r['mean']:9.3f} {2 - r['mean']:11.3f} {r['median']:7.3f} {int(r['n']):4d}")

    print("\n=== (A bonus) Adam beta1-sweep GBS-plateau vs beta1, b8 ===")
    adamb1 = df[(df.optimizer == 'Adam-b1') & (df.batch == 8)].sort_values('beta')
    # include base Adam_b8 as the beta1=0.9 point
    base_adam = df[(df.tag == 'Adam_b8')]
    print(f"{'beta1':>5s} {'mean_GBS':>9s} {'deficit':>9s} {'n':>4s}")
    seen = set()
    for _, r in pd.concat([adamb1, base_adam.assign(beta=0.9)]).sort_values('beta').iterrows():
        if r['beta'] in seen:
            continue
        seen.add(r['beta'])
        if r['n'] == 0:
            print(f"{r['beta']:5.2f}  NO DATA"); continue
        print(f"{r['beta']:5.2f} {r['mean']:9.3f} {2 - r['mean']:9.3f} {int(r['n']):4d}")

    print(f"\nwrote {OUT_DIR}/gbs_grid_analysis.csv")


if __name__ == '__main__':
    main()
