"""
TASK 2: canonical-scale replication of the sub-edge-metastability dichotomy.

Lean setup (mlp_s / num_data=2048 / MSE) concluded small-batch stateful optimizers are
METASTABLE (sub-edge). Does that survive at a realistic scale?

Canonical: mlp (hidden 512 x 2 layers) / num_data=8192 / MSE and CE, small batch (b8).
Optimizers: Adam and SGD-Momentum at standard lrs. Enough steps (default 30k) to reach EoS
(progressive sharpening is slower at 8192).

Per cell, at stationarity, the SAME three signatures:
  (1) GBS = E_B[s^T H_B s / (-g^T s)]                       (edge value 2)
  (2) kappa vs the CORRECT edge:
        SGD-Momentum: kappa_raw = lr*lam(H_B)   vs 2(1+beta)
        Adam:         kappa_precond = lr*lam(D^-1/2 H_B D^-1/2) vs 2   (Task-1 measurement)
                      (also kappa_raw vs 2 for reference)
  (3) AR-pole rho of the unstable coordinate x_t = u^T(theta - EMA), u=top Hessian eigvec.
        marginal edge => rho ~ 1; damped/metastable => rho < 1 (~beta).

Outcomes: (a) metastability survives (dichotomy real); (b) disappears (artifact);
(c) splits by optimizer.

Usage:
  python experiments/canonical_replication.py <cell_tag>
    cells: adam_mse_b8, sgdm_mse_b8, adam_ce_b8, sgdm_ce_b8  (or 'all')
  Env overrides: EOSS_STEPS, EOSS_LR to change per-run.
"""
import os, sys, json, time
import numpy as np
import torch as T
import torch.nn as nn

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
    _run_lobpcg_with_operator, flatt, param_vector,
)

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = 8192
MODEL = 'mlp'          # canonical: hidden 512 x 2 layers
OUT_DIR = os.path.join(_REPO, 'results', 'canonical')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


def get_data(loss_type):
    key = f'xy_{loss_type}'
    if key not in _DATA:
        X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], 888, loss_type=loss_type)
        _DATA[key] = (X, Y)
    return _DATA[key]


def build(loss_type, label_smoothing=0.05):
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets[MODEL]['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets[MODEL]['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    if loss_type == 'mse':
        loss_fn = SquaredLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return net, loss_fn


# --------------------------- measurements ---------------------------

def lam_raw(loss, net, cache):
    return float(compute_eigenvalues(loss, net, k=1, max_iterations=60, reltol=1e-2,
                                     eigenvector_cache=cache, return_eigenvectors=False,
                                     use_power_iteration=False))


def lam_precond(loss, net, D_inv_sqrt, cache):
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
        eig = _run_lobpcg_with_operator(op, net, k=1, max_iterations=60, reltol=1e-2,
                                        init_vectors=None, eigenvector_cache=cache,
                                        return_eigenvectors=False)
    finally:
        hvp.free_memory()
    return float(eig.max())


def probe(net, X, Y, loss_fn, opt, lr, batch, is_adam, n_probe=8):
    """GBS, kappa_raw, kappa_precond over n_probe batches."""
    params = [p for p in net.parameters() if p.requires_grad]
    cache_r = EigenvectorCache(1); cache_p = EigenvectorCache(1)
    D_inv_sqrt = opt.get_preconditioner_inv_sqrt() if is_adam else None
    N = len(X)
    Gbs, LamR, LamP = [], [], []
    for _ in range(n_probe):
        idx = T.randperm(N)[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads); gd = g.detach()
        s = opt.compute_step_direction(g, params).detach()
        hvp_b = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
        try:
            Hs = hvp_b(s, retain_graph_override=False)
            gbs = (T.dot(s, Hs) / (-T.dot(gd, s) + 1e-30)).item()
        finally:
            hvp_b.free_memory()
        if np.isfinite(gbs):
            Gbs.append(gbs)
        pr2 = net(Xb).squeeze(-1); lo2 = loss_fn(pr2, Yb)
        try:
            lr_ = lam_raw(lo2, net, cache_r)
            if np.isfinite(lr_) and lr_ > 0:
                LamR.append(lr_)
        except Exception:
            pass
        if D_inv_sqrt is not None:
            pr3 = net(Xb).squeeze(-1); lo3 = loss_fn(pr3, Yb)
            try:
                lp_ = lam_precond(lo3, net, D_inv_sqrt, cache_p)
                if np.isfinite(lp_) and lp_ > 0:
                    LamP.append(lp_)
            except Exception:
                pass
    Gbs, LamR, LamP = np.array(Gbs), np.array(LamR), np.array(LamP)
    return dict(
        gbs=float(Gbs.mean()) if len(Gbs) else float('nan'),
        gbs_med=float(np.median(Gbs)) if len(Gbs) else float('nan'),
        lam_raw=float(LamR.mean()) if len(LamR) else float('nan'),
        kappa_raw=float(lr * LamR.mean()) if len(LamR) else float('nan'),
        lam_precond=float(LamP.mean()) if len(LamP) else float('nan'),
        kappa_precond=float(lr * LamP.mean()) if len(LamP) else float('nan'),
    )


# --------------------------- AR-pole ---------------------------

def ema_detrend(x, halflife=100):
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    ema = np.empty_like(x); m = x[0]
    for t in range(len(x)):
        m = alpha * x[t] + (1 - alpha) * m; ema[t] = m
    return x - ema


def fit_ar_top_pole(x, k):
    x = np.asarray(x, float); N = len(x)
    if N < k + 40:
        return None
    Y = x[k:N]
    cols = [x[k - 1 - i: N - 1 - i] for i in range(k)]
    Xd = np.column_stack(cols)
    a, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    C = np.zeros((k, k)); C[0, :] = a
    if k > 1:
        C[1:, :-1] = np.eye(k - 1)
    poles = np.linalg.eigvals(C)
    idx = int(np.argmax(np.abs(poles)))
    return float(np.abs(poles[idx])), float(np.angle(poles[idx]))


def ar_pole(net, X, Y, loss_fn, opt, batch, log_steps=4000):
    """Fix u = top eigvec of full-data Hessian; log x_t=u^T theta; fit AR; return rho,phase."""
    Xf, Yf = X, Y
    pf = net(Xf).squeeze(-1); lf = loss_fn(pf, Yf)
    lam, u = compute_eigenvalues(lf, net, k=1, max_iterations=100, reltol=1e-2,
                                 return_eigenvectors=True, use_power_iteration=False)
    u = u.detach().reshape(-1); u = u / (u.norm() + 1e-30)
    xs, losses = [], []
    for _ in range(log_steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item(); losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            break
        opt.zero_grad(); lo.backward(); opt.step()
        xs.append(float(T.dot(u, param_vector(net))))
    xs = np.asarray(xs)
    x_detr = ema_detrend(xs)
    x_detr = x_detr - x_detr.mean()
    rhos, phases = [], []
    for k in range(3, 9):
        r = fit_ar_top_pole(x_detr, k)
        if r:
            rhos.append(r[0]); phases.append(r[1])
    return dict(lam_full=float(lam), rho=float(np.median(rhos)) if rhos else float('nan'),
                rho_spread=float(np.std(rhos)) if rhos else float('nan'),
                phase=float(np.median(phases)) if phases else float('nan'),
                n=len(xs))


# --------------------------- run ---------------------------

CELLS = {
    'adam_mse_b8': dict(optn='Adam', params={'beta1': 0.9, 'beta2': 0.99}, batch=8, lr=0.001, loss='mse'),
    'sgdm_mse_b8': dict(optn='SGD-Momentum', params={'beta': 0.9}, batch=8, lr=0.002, loss='mse'),
    'adam_ce_b8':  dict(optn='Adam', params={'beta1': 0.9, 'beta2': 0.99}, batch=8, lr=0.001, loss='ce'),
    'sgdm_ce_b8':  dict(optn='SGD-Momentum', params={'beta': 0.9}, batch=8, lr=0.01, loss='ce'),
    'sgd_mse_b8':  dict(optn='SGD', params={}, batch=8, lr=0.01, loss='mse'),
    'sgd_ce_b8':   dict(optn='SGD', params={}, batch=8, lr=0.02, loss='ce'),
}


def run_cell(tag, steps=30000, measure_every=6000):
    c = CELLS[tag]
    lr = float(os.environ.get('EOSS_LR', c['lr']))
    steps = int(os.environ.get('EOSS_STEPS', steps))
    is_adam = c['optn'] == 'Adam'
    beta = 0.0 if c['optn'] == 'SGD' else c['params'].get('beta', 0.9)
    edge = 2.0 if is_adam else 2 * (1 + beta)   # correct edge for the raw-H comparison
    X, Y = get_data(c['loss'])
    net, loss_fn = build(c['loss'])
    opt = create_optimizer(c['optn'], net, lr, c['params'])
    print(f"\n=== {tag}: {c['optn']} b={c['batch']} lr={lr} loss={c['loss']} steps={steps} "
          f"(edge={edge:.2f}) ===", flush=True)
    t0 = time.time()
    traj = []
    diverged = False
    for step in range(1, steps + 1):
        idx = T.randperm(len(X))[:c['batch']]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item()
        if not np.isfinite(lv) or lv > 1e6:
            diverged = True; print(f"  DIVERGED @ {step}", flush=True); break
        opt.zero_grad(); lo.backward(); opt.step()
        if step % measure_every == 0:
            m = probe(net, X, Y, loss_fn, opt, lr, c['batch'], is_adam)
            m['step'] = step; m['loss'] = lv; traj.append(m)
            kp = f"kappa_precond={m['kappa_precond']:.3f}" if is_adam else ""
            print(f"  {step:6d} loss={lv:.4f} GBS={m['gbs']:.3f} "
                  f"kappa_raw={m['kappa_raw']:.3f}/edge{edge:.1f} {kp} [{time.time()-t0:.0f}s]", flush=True)
    ar = dict(rho=float('nan'))
    if not diverged:
        ar = ar_pole(net, X, Y, loss_fn, opt, c['batch'])
        print(f"  AR-pole: rho={ar['rho']:.3f}+/-{ar['rho_spread']:.3f} phase/pi={ar['phase']/np.pi:+.2f} "
              f"lam_full={ar['lam_full']:.1f}", flush=True)
    result = dict(tag=tag, **{k:v for k,v in c.items() if k!='lr'}, lr=lr, steps=steps, edge=edge, is_adam=is_adam,
                  diverged=diverged, traj=traj, ar_pole=ar, wall=time.time() - t0)
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    # summary line
    if traj and not diverged:
        f = traj[-1]
        kk = f['kappa_precond'] if is_adam else f['kappa_raw']
        edgek = 2.0 if is_adam else edge
        print(f"  SUMMARY {tag}: GBS={f['gbs']:.2f}  kappa={kk:.2f}/edge{edgek:.1f}={kk/edgek:.0%}  "
              f"rho={ar['rho']:.3f}", flush=True)
    return result


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    tags = list(CELLS) if which == 'all' else [which]
    for tag in tags:
        run_cell(tag)


if __name__ == '__main__':
    main()
