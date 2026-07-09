"""
TASK 1 (pivotal): re-measure lean Adam b8 at its PRECONDITIONED edge.

Our lean "Adam sub-edge" finding measured eta*lambda of the RAW batch Hessian H_B
(~0.40, ~20% of the edge 2). But Adam's relevant edge is on the PRECONDITIONED
Hessian  P^{-1/2} H_B P^{-1/2}  with P = diag(sqrt(v_hat) + eps) from Adam's state
(Cohen et al. 2022, "Adaptive gradient methods at the edge of stability", threshold
eta*lambda_precond ~ 2, or 2(1+beta1) ~ 3.8 for the momentum-corrected bound).

This trains lean Adam b8 (mlp_s, num_data=2048, MSE, lr=0.001) to stationarity, then
at the plateau, over several probe batches, computes BOTH:
    kappa_raw     = lr * lambda_max(H_B)
    kappa_precond = lr * lambda_max(D^{-1/2} H_B D^{-1/2})   (D^{-1/2}=get_preconditioner_inv_sqrt)
Preconditioned operator: v -> D^{-1/2} . H_B . (D^{-1/2} . v), top eigenvalue via LOBPCG
(same pattern as utils/projection_tracker.py::_top5_precond_eigenvectors).

DECISION:
  kappa_precond ~ 2   -> Adam IS at its preconditioned edge; lean "sub-edge" was a
                         wrong-Hessian measurement error (Cohen collision resolved).
  kappa_precond << 2   -> Adam genuinely sub-edge on the correct Hessian; metastability
                         survives the measurement fix.
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

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.measure import (
    compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
    _run_lobpcg_with_operator, flatt,
)

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = int(os.environ.get('EOSS_NUM_DATA', '2048'))
MODEL = os.environ.get('EOSS_MODEL', 'mlp_s')
OUT_DIR = os.path.join(_REPO, 'results', 'precond_edge')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


def get_data():
    if 'xy' not in _DATA:
        X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], 888, loss_type='mse')
        _DATA['xy'] = (X, Y)
    return _DATA['xy']


def build():
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets[MODEL]['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets[MODEL]['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    return net, SquaredLoss()


def lambda_raw(loss, net, cache):
    """Top eigenvalue of H_B (raw batch Hessian)."""
    lam = compute_eigenvalues(loss, net, k=1, max_iterations=60, reltol=1e-2,
                              eigenvector_cache=cache, return_eigenvectors=False,
                              use_power_iteration=False)
    return float(lam)


def lambda_precond(loss, net, D_inv_sqrt, cache):
    """Top eigenvalue of D^{-1/2} H_B D^{-1/2} via LOBPCG with a custom operator."""
    hvp = create_hessian_vector_product(loss, net, retain_graph=True)
    d = D_inv_sqrt

    def precond_operator(v):
        if v.ndim == 1:
            return d * hvp(d * v).detach()
        out = T.empty_like(v)
        for j in range(v.shape[1]):
            out[:, j] = d * hvp(d * v[:, j]).detach()
        return out

    try:
        eigvals = _run_lobpcg_with_operator(
            precond_operator, net, k=1, max_iterations=60, reltol=1e-2,
            init_vectors=None, eigenvector_cache=cache, return_eigenvectors=False)
    finally:
        hvp.free_memory()
    return float(eigvals.max())


def measure_plateau(net, X, Y, loss_fn, opt, lr, batch, n_probe=12):
    """Over n_probe batches, measure raw and preconditioned top eigenvalue of H_B."""
    D_inv_sqrt = opt.get_preconditioner_inv_sqrt()
    if D_inv_sqrt is None:
        raise RuntimeError("optimizer has no preconditioner state yet")
    cache_raw = EigenvectorCache(1)
    cache_pre = EigenvectorCache(1)
    N = len(X)
    lam_raw_list, lam_pre_list = [], []
    for _ in range(n_probe):
        idx = T.randperm(N)[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        try:
            lr_ = lambda_raw(lo, net, cache_raw)
        except Exception as e:
            lr_ = float('nan')
        pr2 = net(Xb).squeeze(-1); lo2 = loss_fn(pr2, Yb)
        try:
            lp_ = lambda_precond(lo2, net, D_inv_sqrt, cache_pre)
        except Exception as e:
            lp_ = float('nan')
        if np.isfinite(lr_) and lr_ > 0:
            lam_raw_list.append(lr_)
        if np.isfinite(lp_) and lp_ > 0:
            lam_pre_list.append(lp_)
    lam_raw_arr = np.array(lam_raw_list); lam_pre_arr = np.array(lam_pre_list)
    return dict(
        lam_raw_mean=float(lam_raw_arr.mean()) if len(lam_raw_arr) else float('nan'),
        lam_raw_med=float(np.median(lam_raw_arr)) if len(lam_raw_arr) else float('nan'),
        lam_raw_std=float(lam_raw_arr.std()) if len(lam_raw_arr) else float('nan'),
        lam_pre_mean=float(lam_pre_arr.mean()) if len(lam_pre_arr) else float('nan'),
        lam_pre_med=float(np.median(lam_pre_arr)) if len(lam_pre_arr) else float('nan'),
        lam_pre_std=float(lam_pre_arr.std()) if len(lam_pre_arr) else float('nan'),
        kappa_raw=float(lr * lam_raw_arr.mean()) if len(lam_raw_arr) else float('nan'),
        kappa_precond=float(lr * lam_pre_arr.mean()) if len(lam_pre_arr) else float('nan'),
        kappa_precond_med=float(lr * np.median(lam_pre_arr)) if len(lam_pre_arr) else float('nan'),
        n_raw=len(lam_raw_arr), n_pre=len(lam_pre_arr),
        lam_raw_all=lam_raw_list, lam_pre_all=lam_pre_list,
    )


def run(tag='Adam_b8_lean', optn='Adam', params=None, batch=8, lr=0.001,
        train_steps=20000, measure_at=(8000, 12000, 16000, 20000)):
    if params is None:
        params = {'beta1': 0.9, 'beta2': 0.99}
    X, Y = get_data()
    net, loss_fn = build()
    opt = create_optimizer(optn, net, lr, params)
    beta1 = params.get('beta1', 0.9)
    t0 = time.time()
    measure_at = sorted(set(measure_at))
    checkpoints = []
    diverged = False
    mi = 0
    for step in range(1, train_steps + 1):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item()
        if not np.isfinite(lv) or lv > 1e6:
            diverged = True; print(f"  [{tag}] DIVERGED @ {step}", flush=True); break
        opt.zero_grad(); lo.backward(); opt.step()
        if mi < len(measure_at) and step == measure_at[mi]:
            m = measure_plateau(net, X, Y, loss_fn, opt, lr, batch, n_probe=12)
            m['step'] = step; m['loss'] = lv
            checkpoints.append(m)
            print(f"  [{tag}] {step:6d} loss={lv:.4f}  "
                  f"kappa_raw={m['kappa_raw']:.3f} (lam_raw={m['lam_raw_mean']:.1f})  "
                  f"kappa_precond={m['kappa_precond']:.3f} (lam_pre={m['lam_pre_mean']:.2f}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            mi += 1

    # stationarity check on kappa over last 2 checkpoints
    result = dict(tag=tag, optimizer=optn, params=params, batch=batch, lr=lr,
                  beta1=beta1, train_steps=train_steps, diverged=diverged,
                  edge_raw=2.0, edge_precond=2.0, edge_precond_mom=2 * (1 + beta1),
                  checkpoints=checkpoints, wall=time.time() - t0)
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    r = run()
    print("\n===== TASK 1 VERDICT (lean Adam b8) =====")
    if r['diverged'] or not r['checkpoints']:
        print("  diverged / no data"); return
    final = r['checkpoints'][-1]
    print(f"  trained {r['train_steps']} steps, lr={r['lr']}, beta1={r['beta1']}")
    print(f"  RAW      : kappa_raw     = eta*lam(H_B)              = {final['kappa_raw']:.3f}  (edge 2)")
    print(f"  PRECOND  : kappa_precond = eta*lam(D^-1/2 H_B D^-1/2) = {final['kappa_precond']:.3f}  (edge 2; mom-edge {r['edge_precond_mom']:.2f})")
    print(f"  ratio precond/raw = {final['kappa_precond']/max(final['kappa_raw'],1e-9):.2f}x")
    print(f"  precond/edge2 = {final['kappa_precond']/2.0:.2%}   precond/edge{r['edge_precond_mom']:.1f} = {final['kappa_precond']/r['edge_precond_mom']:.2%}")
    if final['kappa_precond'] > 1.4:
        print("  => Adam sits at (or near) its PRECONDITIONED edge. Cohen collision RESOLVED; lean sub-edge was wrong-Hessian.")
    elif final['kappa_precond'] < 0.7:
        print("  => Adam still << preconditioned edge. Metastability SURVIVES the measurement fix.")
    else:
        print("  => intermediate: partially closes the gap; report the number.")


if __name__ == '__main__':
    main()
