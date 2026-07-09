"""
TASK 1 robustness: is lean Adam b8's preconditioned sharpness (~6.7 >> 2) a real
super-edge, or an eps-floor artifact?

Near interpolation the batch gradients shrink, so v_hat shrinks, so D = sqrt(v_hat)+eps
approaches the eps=1e-8 floor. D^{-1/2}=1/sqrt(D) then over-amplifies "dead" (low-v)
coordinates. If the top eigenvector of D^{-1/2} H_B D^{-1/2} lives in those coordinates,
the ~6.7 is spurious.

Checks at the 20k plateau (retrains, ~1 min):
  1. eps-sweep: recompute D with eps = c * median(sqrt(v_hat)) for c in a range, and
     also the raw Adam eps=1e-8. If kappa_precond is roughly flat across c, it's the
     real preconditioned sharpness, not the floor.
  2. eigenvector localization: for the top preconditioned eigenvector w, report the
     participation ratio and the correlation of |w| with sqrt(v_hat). If w concentrates
     on low-v coords, the number is a dead-coordinate artifact.
  3. Adam-step-aligned curvature (floor-immune sanity): kappa_step = eta*lam scaled by
     the curvature actually seen along Adam's normalized step ŝ:  s^T H_B s / (s^T D s)
     * lr  -- a scalar that governs the loss change along Adam's real motion, no LOBPCG,
     no floor amplification of unvisited directions.
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

import experiments.precond_edge_task1 as P
from utils.optimizer import create_optimizer
from utils.measure import (
    compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
    _run_lobpcg_with_operator, flatt,
)
import torch as T

OUT_DIR = P.OUT_DIR


def v_hat_flat(opt):
    """Bias-corrected sqrt(v_hat) flat vector from Adam state (matches get_preconditioner_inv_sqrt)."""
    beta2 = opt.beta2
    pieces = []
    for p in opt.inner.param_groups[0]['params']:
        state = opt.inner.state.get(p)
        v = state['exp_avg_sq']
        t = state['step'].item() if T.is_tensor(state['step']) else state['step']
        v_hat = v / (1 - beta2 ** t)
        pieces.append(v_hat.sqrt().view(-1))
    return T.cat(pieces).detach()


def lam_precond_with_D(loss, net, D_inv_sqrt, cache, want_vec=False):
    hvp = create_hessian_vector_product(loss, net, retain_graph=True)
    d = D_inv_sqrt

    def op(v):
        if v.ndim == 1:
            return d * hvp(d * v).detach()
        out = T.empty_like(v)
        for j in range(v.shape[1]):
            out[:, j] = d * hvp(d * v[:, j]).detach()
        return out
    try:
        eig = _run_lobpcg_with_operator(op, net, k=1, max_iterations=80, reltol=1e-2,
                                        init_vectors=None, eigenvector_cache=cache,
                                        return_eigenvectors=True)
        eigvals, eigvecs = eig
    finally:
        hvp.free_memory()
    idx = int(T.argmax(eigvals))
    return float(eigvals[idx]), eigvecs[:, idx].detach()


def main():
    lr = 0.001
    X, Y = P.get_data()
    net, loss_fn = P.build()
    opt = create_optimizer('Adam', net, lr, {'beta1': 0.9, 'beta2': 0.99})
    t0 = time.time()
    for step in range(1, 20001):
        idx = T.randperm(len(X))[:8]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        opt.zero_grad(); lo.backward(); opt.step()
    print(f"trained 20k in {time.time()-t0:.0f}s, loss={lo.item():.4f}", flush=True)

    sqv = v_hat_flat(opt)                          # sqrt(v_hat)
    med_sqv = float(sqv.median())
    print(f"sqrt(v_hat): median={med_sqv:.3e} q10={float(sqv.quantile(0.1)):.3e} "
          f"q90={float(sqv.quantile(0.9)):.3e} min={float(sqv.min()):.3e}", flush=True)
    frac_at_floor = float((sqv < 1e-6).float().mean())
    print(f"frac coords with sqrt(v_hat) < 1e-6 (near eps floor): {frac_at_floor:.3f}", flush=True)

    # probe batches: average over several
    eps_list = [('adam_1e-8', 1e-8),
                ('rel_0.01', 0.01 * med_sqv),
                ('rel_0.1', 0.1 * med_sqv),
                ('rel_1.0', 1.0 * med_sqv),
                ('rel_10', 10.0 * med_sqv)]
    n_probe = 8
    results = {name: [] for name, _ in eps_list}
    step_kappas = []
    locz = {name: [] for name, _ in eps_list}
    for pi in range(n_probe):
        idx = T.randperm(len(X))[:8]
        Xb, Yb = X[idx], Y[idx]
        for name, eps in eps_list:
            D = sqv + eps
            D_inv_sqrt = (1.0 / D.sqrt()).detach()
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            cache = EigenvectorCache(1)
            try:
                lam, w = lam_precond_with_D(lo, net, D_inv_sqrt, cache, want_vec=True)
                results[name].append(lr * lam)
                # localization: participation ratio & corr(|w|, sqrt(v_hat))
                wa = w.abs()
                pr_ratio = float((wa.sum() ** 2) / (wa.numel() * (wa ** 2).sum()))
                # correlation of squared weight with sqrt(v_hat) rank
                w2 = (w ** 2)
                # mass-weighted median sqrt(v_hat) at the eigenvector's support
                order = T.argsort(w2, descending=True)
                top = order[:max(1, w2.numel() // 100)]  # top 1% coords by weight
                locz[name].append(float(sqv[top].median()))
            except Exception as e:
                pass
        # step-aligned curvature (floor-immune): use raw Adam step direction
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        params = [p for p in net.parameters() if p.requires_grad]
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads)
        s = opt.compute_step_direction(g, params).detach()   # = -lr * mhat/denom
        hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
        try:
            Hs = hvp(s, retain_graph_override=False).detach()
        finally:
            hvp.free_memory()
        # along Adam's actual motion, the effective 1D curvature ratio governing
        # loss change is (s^T H s)/(-g^T s) -- this is exactly GBS; and the step-map
        # gain is (s^T H s)/(s^T D s)*? Simpler diagnostic: normalized curvature along s
        sHs = float(T.dot(s, Hs))
        sDs = float(T.dot(s, (sqv + 1e-8) * s))   # s^T D s
        # kappa along the actual step (the linear map gain along s): |1 - eta * (Hs)/(D s)| proxy
        step_kappas.append(dict(sHs=sHs, sDs=sDs,
                                ratio_HD=float(sHs / (sDs + 1e-30))))

    print("\n eps variant       kappa_precond(mean+/-std over probes)   top1%-coord median sqrt(v_hat)")
    for name, _ in eps_list:
        arr = np.array(results[name])
        lz = np.array(locz[name])
        if len(arr):
            print(f"  {name:12s}  {arr.mean():8.3f} +/- {arr.std():.3f}   "
                  f"loc_sqv={np.median(lz):.3e}", flush=True)
    sk = step_kappas
    ratio = np.array([d['ratio_HD'] for d in sk])
    print(f"\n step-aligned s^T H s / s^T D s (mean over probes) = {ratio.mean():.3f} +/- {ratio.std():.3f}")
    print(f"  (this * lr would be the raw gain; interpret vs 2/lr)")

    out = dict(lr=lr, med_sqrt_vhat=med_sqv, frac_at_floor=frac_at_floor,
               kappa_precond={name: float(np.mean(results[name])) if results[name] else None
                              for name, _ in eps_list},
               loc_top1pct_sqv={name: float(np.median(locz[name])) if locz[name] else None
                                for name, _ in eps_list},
               step_ratio_HD_mean=float(ratio.mean()))
    with open(os.path.join(OUT_DIR, 'task1_robust.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nInterpretation:")
    print("  kappa_precond roughly flat across eps variants => real preconditioned sharpness (not floor).")
    print("  kappa_precond collapses as eps grows / top-coords have tiny sqrt(v_hat) => dead-coord artifact.")


if __name__ == '__main__':
    main()
