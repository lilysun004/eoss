"""
Criticality sweep: for a given (optimizer, batch, params) cell, sweep lr upward
and report the plateau GBS, batch_sharpness, eta*batch_sharpness, and lmax --
NO pass/fail gating (catapults near the edge are expected). Goal: locate the lr
at which the run is genuinely AT the edge (GBS -> 2 for SGD / eta*BS -> ~2),
not merely stably sub-critical.

Usage:
    python experiments/criticality_sweep.py SGD 8 '{}' 0.01,0.014,0.018,0.022,0.026,0.03
    python experiments/criticality_sweep.py SGD-Momentum 8 "{'beta':0.9}" 0.002,0.003,0.004,0.005
"""
import os, sys, json, ast
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')

import experiments.calibrate_grid as cg


def main():
    optn = sys.argv[1]
    batch = int(sys.argv[2])
    params = ast.literal_eval(sys.argv[3])
    lrs = [float(x) for x in sys.argv[4].split(',')]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    print(f"# criticality sweep {optn} b={batch} params={params} steps={steps}")
    print(f"{'lr':>8s} {'GBS':>7s} {'BS':>9s} {'eta*BS':>7s} {'lmax':>9s} {'eta*lmax':>8s} {'floss':>8s} {'div':>4s}")
    rows = []
    for lr in lrs:
        r = cg.run_cell(optn, params, batch, lr, steps,
                        measure_every=max(40, steps // 30), n_probe=6, bs_estimates=8,
                        tag=f"crit_{optn}_b{batch}_lr{lr}")
        v = r['verdict']
        gbs = v.get('gbs_mean'); bs = v.get('bs_mean'); lmax = v.get('lmax_mean')
        fl = v.get('final_loss'); div = r['diverged']
        etabs = lr * bs if (bs is not None and np.isfinite(bs)) else float('nan')
        etalm = lr * lmax if (lmax is not None and np.isfinite(lmax)) else float('nan')
        def f(x, w=7, p=3):
            return (f"{x:{w}.{p}f}" if x is not None and np.isfinite(x) else f"{'--':>{w}s}")
        print(f"{lr:8.4f} {f(gbs)} {f(bs,9,2)} {f(etabs)} {f(lmax,9,2)} {f(etalm,8)} {f(fl,8,4)} {str(div):>4s}", flush=True)
        rows.append(dict(lr=lr, gbs=gbs, bs=bs, eta_bs=etabs, lmax=lmax, diverged=div, final_loss=fl))
    out = os.path.join(_REPO, 'results', 'criticality')
    os.makedirs(out, exist_ok=True)
    tag = f"{optn}_b{batch}_{'_'.join(f'{k}{v}' for k,v in params.items())}"
    with open(os.path.join(out, f"{tag}.json"), 'w') as fh:
        json.dump(rows, fh, indent=2)


if __name__ == '__main__':
    main()
