"""
Close the "one lr" hole: does canonical (mlp/8192) SGD-Momentum b8 reach a STABLE
GBS~2 marginal shelf at ANY lr, or go sub-edge -> divergence with no shelf (as lean
Exp-2 showed)? A referee's obvious objection is "your lr was too low"; this brackets it.

Reuses edge_reachability.run (trains to ~stationarity, reports stabilized GBS/kappa +
divergence) at CANONICAL scale via EOSS_MODEL/EOSS_NUM_DATA env overrides that
long_train_grid.build/get_data respect. SGD b8 included as the in-run control that DOES
have a stable edge shelf.
"""
import os, sys, json
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
os.environ['EOSS_MODEL'] = 'mlp'        # canonical model (bigger than mlp_s)
os.environ['EOSS_NUM_DATA'] = '8192'    # canonical size

import experiments.edge_reachability as E   # its run() uses long_train_grid.build/get_data

OUT_DIR = os.path.join(_REPO, 'results', 'canonical_lr_sweep')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    steps = 18000
    sweeps = {
        'SGD-Momentum': ({'beta': 0.9}, [0.001, 0.002, 0.004, 0.006, 0.008]),
        'SGD':          ({}, [0.01, 0.02, 0.03]),   # control: should have a stable GBS~2 shelf
    }
    results = {}
    for optn, (params, lrs) in sweeps.items():
        print(f"\n=== canonical {optn} b8 sweep lrs={lrs} (mlp/8192, {steps} steps) ===", flush=True)
        cell = []
        for lr in lrs:
            r = E.run(optn, params, 8, lr, steps, seed=0, measure_every=3000)
            edge = 2 * (1 + params.get('beta', 0.0))
            koe = r['kappa'] / edge if np.isfinite(r['kappa']) else float('nan')
            rec = dict(lr=lr, gbs=r['gbs'], kappa=r['kappa'], kappa_over_edge=koe, diverged=r['diverged'])
            cell.append(rec)
            print(f"  lr={lr:.4f}: GBS={r['gbs']:.3f}  eta*lam={r['kappa']:.3f}  k/edge={koe:.1%}  diverged={r['diverged']}", flush=True)
        results[optn] = cell
        with open(os.path.join(OUT_DIR, 'canonical_lr_sweep.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n===== VERDICT =====")
    for optn, cell in results.items():
        maxg = max((c['gbs'] for c in cell if np.isfinite(c['gbs'])), default=float('nan'))
        shelf = any(np.isfinite(c['gbs']) and c['gbs'] > 1.7 and not c['diverged'] for c in cell)
        print(f"  {optn:13s}: max stable GBS={maxg:.3f}  has stable GBS>1.7 shelf? {shelf}")
    print("  SGD should have a shelf (GBS~2); if SGD-Momentum has NONE (sub-edge->diverge), the")
    print("  'no marginal shelf at any lr' claim holds at canonical scale, closing the one-lr hole.")


if __name__ == '__main__':
    main()
