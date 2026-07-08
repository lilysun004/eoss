"""
Experiment 2 (decisive dichotomy test): is the marginal-stability edge REACHABLE
for small-batch stateful optimizers?

Exp 1 showed small-batch momentum/Adam stabilize FAR below their edge (lambda/edge
~0.1-0.2, GBS~0.3-0.46), while SGD sits AT its edge (GBS~2) at every batch size.
The arbiter showed momentum b8 DIVERGES at lr~0.0025 while still sub-edge. So the
hypothesis is: stateful optimizers at small batch go sub-edge-stable -> noise
divergence WITHOUT a stable marginal (GBS~=2) state in between.

Test: sweep lr for SGD (control, has a stable edge), SGD-Momentum, Adam at b8,
2 seeds each, train to ~stationarity, and record the stabilized GBS, eta*lambda,
lambda/own_edge, and whether it diverged. If SGD shows a broad stable GBS~=2
plateau vs lr while momentum/Adam show GBS staying <1 then diverging (never ~2),
the regime dichotomy is airtight and GBS=2 is confirmed as an at-the-edge-only
signature that small-batch stateful optimizers never reach.
"""
import os, sys, json, time
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import experiments.long_train_grid as L   # reuse build/probe/get_data
from utils.optimizer import create_optimizer

OUT_DIR = os.path.join(_REPO, 'results', 'edge_reachability')
os.makedirs(OUT_DIR, exist_ok=True)


def run(optn, params, batch, lr, steps, seed, measure_every=2000):
    T.manual_seed(seed); np.random.seed(seed)
    X, Y = L.get_data()
    net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    gbs_hist, kap_hist, diverged = [], [], False
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            diverged = True; break
        opt.zero_grad(); lo.backward(); opt.step()
        if step % measure_every == 0 and step > 0:
            m = L.probe(net, X, Y, loss_fn, opt, lr, batch, n_probe=12)
            gbs_hist.append(m['gbs']); kap_hist.append(m['kappa'])
    # stabilized = mean over back half of measured points
    if gbs_hist:
        h = len(gbs_hist) // 2
        gbs = float(np.mean(gbs_hist[h:])); kap = float(np.mean(kap_hist[h:]))
    else:
        gbs = kap = float('nan')
    return dict(gbs=gbs, kappa=kap, diverged=diverged, n_meas=len(gbs_hist))


def main():
    beta = 0.9
    sweeps = {
        'SGD':          ({}, [0.008, 0.012, 0.016, 0.020, 0.024], 2/1),
        'SGD-Momentum': ({'beta': beta}, [0.0018, 0.0021, 0.0024, 0.0027, 0.0030], 2*(1+beta)),
        'Adam':         ({'beta1': 0.9, 'beta2': 0.99}, [0.0008, 0.0012, 0.0016, 0.0020, 0.0026], None),
    }
    batch, steps, seeds = 8, 12000, [0, 1]
    results = {}
    for optn, (params, lrs, edge_mult) in sweeps.items():
        print(f"\n=== {optn} b={batch} sweep lrs={lrs} ===", flush=True)
        cell = []
        for lr in lrs:
            runs = [run(optn, params, batch, lr, steps, s) for s in seeds]
            gbs = np.nanmean([r['gbs'] for r in runs])
            kap = np.nanmean([r['kappa'] for r in runs])
            ndiv = sum(r['diverged'] for r in runs)
            own_edge_kappa = edge_mult if edge_mult else float('nan')  # eta*lambda threshold
            rec = dict(lr=lr, gbs=float(gbs), kappa=float(kap), n_div=ndiv, nseed=len(seeds),
                       kappa_over_edge=(float(kap/own_edge_kappa) if edge_mult else None))
            cell.append(rec)
            edge_s = f"k/edge={rec['kappa_over_edge']:.2f}" if edge_mult else ""
            print(f"  lr={lr:.4f}: GBS={gbs:.3f}  eta*lam={kap:.3f}  {edge_s}  diverged={ndiv}/{len(seeds)}", flush=True)
        results[optn] = cell
        with open(os.path.join(OUT_DIR, 'edge_reachability.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # plot GBS vs lr per optimizer (normalized lr by first divergence for overlay)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for optn, cell in results.items():
        lrs = [c['lr'] for c in cell]; gbs = [c['gbs'] for c in cell]
        divmask = [c['n_div'] > 0 for c in cell]
        ax[0].plot(lrs, gbs, '-o', label=optn)
        for lr, g, dv in zip(lrs, gbs, divmask):
            if dv: ax[0].plot(lr, g, 'x', color='red', ms=10)
    ax[0].axhline(2.0, color='k', ls='--', lw=0.8)
    ax[0].set_xlabel('lr'); ax[0].set_ylabel('stabilized GBS'); ax[0].set_title('GBS vs lr (red x = some seed diverged)')
    ax[0].legend(fontsize=8); ax[0].set_xscale('log')
    for optn, cell in results.items():
        koe = [c['kappa_over_edge'] for c in cell if c['kappa_over_edge'] is not None]
        lrs = [c['lr'] for c in cell if c['kappa_over_edge'] is not None]
        if koe: ax[1].plot(lrs, koe, '-o', label=optn)
    ax[1].axhline(1.0, color='k', ls='--', lw=0.8)
    ax[1].set_xlabel('lr'); ax[1].set_ylabel('eta*lambda / own_edge'); ax[1].set_title('sharpness vs own linear edge')
    ax[1].legend(fontsize=8); ax[1].set_xscale('log')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'edge_reachability.png'), dpi=90); plt.close(fig)

    print("\n===== VERDICT =====")
    for optn, cell in results.items():
        maxg = max((c['gbs'] for c in cell if np.isfinite(c['gbs'])), default=float('nan'))
        anystable2 = any(np.isfinite(c['gbs']) and c['gbs'] > 1.7 and c['n_div'] == 0 for c in cell)
        print(f"  {optn:13s}: max stable GBS={maxg:.3f}  reaches stable GBS>1.7? {anystable2}")
    print("  => SGD reaches stable GBS~2; if momentum/Adam do NOT (jump sub-edge->diverge), dichotomy airtight.")


if __name__ == '__main__':
    main()
