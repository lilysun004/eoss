"""
The other two causal, yardstick-free regime instruments (companions to perturb_relax),
across the beta-sweep at fixed b8 for SGD-Momentum. No beta, no edge formula in any metric.

TEST 1 -- SHARPENING-SUPPRESSION ("is the boundary binding?"): at the plateau, DROP lr 10x
for a stretch and watch top Hessian sharpness lambda. MARGINAL: lambda was PINNED at the edge
by feedback; dropping lr raises the edge (2/eta' = 20/eta), releasing the constraint, so
lambda RESUMES CLIMBING (Cohen-et-al. lr-drop signature). METASTABLE: lambda was sub-edge in a
basin, not pinned, so it stays flat. Report the relative rise (lambda_end-lambda_0)/lambda_0
over the low-lr stretch, RELATIVE to a same-length baseline at the original lr (controls for
drift). Marginal => large positive rise; metastable => ~0.

TEST 2 -- CATAPULT-SIDE EXCURSION DISTRIBUTION (renewal fingerprint): log x_t = u^T(theta_t -
EMA(theta_t)) over a long plateau window. MARGINAL: unimodal, continuous, light-tailed (modest
kurtosis, small p99/p50). METASTABLE: two-regime -- small damped bulk + rare large spikes;
heavy-tailed (high kurtosis, large p99/p50), finite catapult rate. Report kurtosis, p99/p50 of
|x|, and catapult rate, per beta.

Both should flip character near the R~1 crossover (beta~0.3) where GBS declines and where
perturb_relax's gamma flips sign -- three independent causal instruments agreeing.
"""
import os, sys, json
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True
from scipy import stats as _stats

import experiments.perturb_relax as PR
import experiments.long_train_grid as L
from utils.measure import compute_eigenvalues, EigenvectorCache, param_vector

T.set_num_threads(4)
OUT_DIR = os.path.join(_REPO, 'results', 'causal_regime_tests')
os.makedirs(OUT_DIR, exist_ok=True)


def lam_subset(net, loss_fn, X, Y, cache, cap=2048):
    Xs, Ys = X[:cap], Y[:cap]
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    return float(compute_eigenvalues(lo, net, k=1, max_iterations=40, reltol=0.01,
                                     eigenvector_cache=cache, return_eigenvectors=False,
                                     use_power_iteration=False))


def sharpening_suppression(state, lr, batch, stretch=1000, every=50):
    """Run `stretch` steps at lr/10 and at lr (baseline); report lambda rise in each."""
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']
    import copy
    def run_at(mult, seed):
        net.load_state_dict(copy.deepcopy(state['net_sd']))
        PR.set_optimizer_state(opt, state['opt_sd'])
        opt.inner.param_groups[0]['lr'] = lr * mult
        cache = EigenvectorCache(1); lams = []
        g = T.Generator().manual_seed(seed)
        for t in range(stretch):
            if t % every == 0:
                lams.append(lam_subset(net, loss_fn, X, Y, cache))
            idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            if not np.isfinite(lo.item()) or lo.item() > 1e6:
                break
            opt.zero_grad(); lo.backward(); opt.step()
        opt.inner.param_groups[0]['lr'] = lr   # restore
        return np.array(lams)
    drop = run_at(0.1, seed=11)
    base = run_at(1.0, seed=11)
    def rel_rise(a):
        return float((a[-1] - a[0]) / (a[0] + 1e-12)) if len(a) > 1 else float('nan')
    return dict(lr_drop_rel_rise=rel_rise(drop), baseline_rel_rise=rel_rise(base),
                lam0=float(drop[0]) if len(drop) else float('nan'),
                drop_series=drop.tolist(), base_series=base.tolist())


def excursion_stats(state, u, batch, window=3000, ema_hl=100):
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']
    import copy
    net.load_state_dict(copy.deepcopy(state['net_sd']))
    PR.set_optimizer_state(opt, state['opt_sd'])
    alpha = 1 - 0.5 ** (1.0 / ema_hl)
    theta = param_vector(net).clone()
    ema = float(T.dot(theta, u))
    xs = []
    g = T.Generator().manual_seed(23)
    for t in range(window):
        x = float(T.dot(param_vector(net), u))
        ema = (1 - alpha) * ema + alpha * x
        xs.append(x - ema)
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            break
        opt.zero_grad(); lo.backward(); opt.step()
    xs = np.array(xs); ax = np.abs(xs)
    ax = ax[np.isfinite(ax)]
    if len(ax) < 50:
        return dict(kurtosis=float('nan'), p99_over_p50=float('nan'), cat_rate=float('nan'), n=len(ax))
    med = np.median(ax) + 1e-30
    cat = int(np.sum(ax > 5 * med))
    return dict(kurtosis=float(_stats.kurtosis(xs[np.isfinite(xs)])),
                p99_over_p50=float(np.quantile(ax, 0.99) / med),
                cat_rate=float(cat / len(ax)), n=len(ax))


def main():
    cells = [
        (0.0,  0.006, 12000),
        (0.3,  0.005, 12000),
        (0.6,  0.004, 12000),
        (0.9,  0.002, 14000),
        (0.99, 0.0001, 16000),
    ]
    batch = 8
    results = []
    for beta, lr, steps in cells:
        print(f"\n=== beta={beta} lr={lr} steps={steps} ===", flush=True)
        st = PR.train_plateau("SGD-Momentum", {"beta": beta}, batch, lr, steps)
        if st is None:
            print(f"  beta={beta}: diverged"); results.append(dict(beta=beta, diverged=True)); continue
        sh = sharpening_suppression(st, lr, batch)
        ex = excursion_stats(st, st['u_hess'], batch)
        rec = dict(beta=beta, lr=lr, diverged=False,
                   sharpening_lr_drop_rise=sh['lr_drop_rel_rise'],
                   sharpening_baseline_rise=sh['baseline_rel_rise'], lam0=sh['lam0'],
                   excursion_kurtosis=ex['kurtosis'], excursion_p99_p50=ex['p99_over_p50'],
                   excursion_cat_rate=ex['cat_rate'])
        results.append(rec)
        print(f"  SHARPEN: lr-drop rise={sh['lr_drop_rel_rise']:+.2f} (baseline {sh['baseline_rel_rise']:+.2f})  "
              f"EXCURSION: kurt={ex['kurtosis']:.2f} p99/p50={ex['p99_over_p50']:.2f} cat_rate={ex['cat_rate']:.4f}",
              flush=True)
        with open(os.path.join(OUT_DIR, 'causal_regime_tests.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n===== VERDICT (both instruments vs beta / R) =====")
    print(f"{'beta':>5s} {'sharpen(lr-drop-baseline)':>26s} {'exc_kurtosis':>13s} {'p99/p50':>8s} {'cat_rate':>9s}")
    for r in results:
        if r.get('diverged'): print(f"{r['beta']:5.2f}  diverged"); continue
        net_rise = r['sharpening_lr_drop_rise'] - r['sharpening_baseline_rise']
        print(f"{r['beta']:5.2f} {net_rise:26.2f} {r['excursion_kurtosis']:13.2f} "
              f"{r['excursion_p99_p50']:8.2f} {r['excursion_cat_rate']:9.4f}")
    print("\n MARGINAL (small beta): large sharpening-resumption, low kurtosis/light tails.")
    print(" METASTABLE (large beta): ~0 sharpening-resumption, high kurtosis/heavy tails.")
    print(" Transition near beta~0.3 (R~1) => third+fourth causal instruments agree with perturb-relax gamma.")


if __name__ == '__main__':
    main()
