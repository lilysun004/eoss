"""
CAUSAL test of the control parameter R (gates the whole R story).

The batch-size mechanism table confounds R with everything (noise, curvature, alignment all
change with batch). The clean causal test: FIX batch=8, sweep beta in {0,0.3,0.6,0.9,0.99}.
This varies R = (buffer memory 1/(1-beta)) / (u_B rotation timescale tau_rot) PURELY through
the buffer memory, holding the landscape and (approximately) tau_rot fixed. beta=0 is SGD
(buffer contributes nothing) and must recover edge-reaching behavior continuously.

Prediction if R is CAUSAL (not just correlated with batch): edge-reachability (kappa/edge, GBS)
and buffer-u_B alignment decline MONOTONICALLY with beta at fixed batch. If reachability does
NOT track beta at fixed batch, R is a correlate, not the control parameter, and the R story
is wrong -- do not write it up.

Also reports the random-alignment baseline E|cos| ~ sqrt(2/(pi*d)) so cos(buffer,u_B) is read
against chance (~0.001 for d~8e5), not in a vacuum.
lr per beta from FINAL_GRID's validated beta-sweep (keeps each cell near its own edge).
"""
import os, sys, json
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')

import experiments.mechanism_buffer_rotation as M
import torch as T

OUT_DIR = os.path.join(_REPO, 'results', 'beta_sweep_mechanism')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # (beta, lr, train_steps) -- lr from FINAL_GRID beta_sweep_SGDMomentum_b8 (each near its edge)
    grid = [
        (0.0,  0.006, 3500),
        (0.3,  0.005, 3500),
        (0.6,  0.004, 3500),
        (0.9,  0.002, 4000),
        (0.99, 0.0001, 6000),
    ]
    batch = 8
    # random |cos| baseline in param space
    d = M.param_vector(M.L.build()[0]).numel()
    chance = float(np.sqrt(2.0 / (np.pi * d)))
    print(f"param dim d={d}  random |cos| chance level ~ {chance:.5f}\n", flush=True)

    results = []
    for beta, lr, ts in grid:
        tag = f"b8_beta{beta}"
        print(f"=== {tag}: SGD-Momentum b8 beta={beta} lr={lr} steps={ts} ===", flush=True)
        r = M.run_cell(tag, "SGD-Momentum", {"beta": beta}, batch, lr, ts)
        if r.get('diverged'):
            print(f"  {tag}: DIVERGED"); results.append(r); continue
        r['chance_cos'] = chance
        r['cos_buf_over_chance'] = r['cos_buf_uB'] / chance
        results.append(r)
        print(f"  beta={beta}: R={r['R']:.2f}  tau_rot={r['tau_rot']:.2f}  "
              f"cos(buf,uB)={r['cos_buf_uB']:.3f} ({r['cos_buf_uB']/chance:.0f}x chance)  "
              f"cos(step,uB)={r['cos_step_uB']:.3f}  GBS={r['gbs']:.3f}  "
              f"eta*lam/edge={r['eta_lam_over_edge']:.1%}", flush=True)
        with open(os.path.join(OUT_DIR, 'beta_sweep_mechanism.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n===== CAUSAL VERDICT: does reachability track beta at FIXED batch=8? =====")
    print(f"{'beta':>5s} {'R':>7s} {'tau_rot':>7s} {'cos(buf,uB)':>11s} {'cos(step,uB)':>12s} {'GBS':>6s} {'lam/edge':>9s}")
    for r in results:
        if r.get('diverged'):
            print(f"{r['beta']:5.2f}  diverged"); continue
        print(f"{r['beta']:5.2f} {r['R']:7.2f} {r['tau_rot']:7.2f} {r['cos_buf_uB']:11.3f} "
              f"{r['cos_step_uB']:12.3f} {r['gbs']:6.2f} {r['eta_lam_over_edge']:9.1%}")
    print(f"\n (random |cos| chance ~ {chance:.5f})")
    print(" MONOTONIC decline in GBS/alignment with beta, beta=0 recovering SGD => R is CAUSAL.")
    print(" If flat/non-monotonic => R is a batch-size correlate, not the control parameter.")


if __name__ == '__main__':
    main()
