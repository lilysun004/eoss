"""
Synthetic arbiter v2 -- the CERTIFICATE test (addresses the tautology objection).

v1 flaw: harvested curvature along the FULL-Hessian top eigenvector u, wrong
direction at small batch. v2 harvests h = lambda_max(H_B) along the PER-BATCH
top eigenvector (= batch sharpness, the EoSS order parameter) over many batches
at a near-edge but stable checkpoint.

Then, purely in numpy, for the harvested h-distribution we compute the PREDICTED
divergence threshold eta* under each stability notion:
  SGD scalar  x_{t+1}=(1-eta h_t)x_t :
     a.s./log : E[log|1-eta h|] = 0
     mean     : E[|1-eta h|]    = 1
     mean-sq  : E[(1-eta h)^2]  = 1   (eta_L2 = 2 E[h]/E[h^2], closed form)
  heavy-ball companion M_t=[[1+b-eta h_t,-b],[1,0]] :
     a.s.     : top Lyapunov exponent (simulated) = 0
     mean-sq  : top eigenvalue of E[M⊗M] = 1

Non-circular part: we ALSO empirically bracket where the REAL optimizer diverges
(coarse lr scan, harness = loss blows past 1e6). The stability notion whose
predicted eta* matches the empirical divergence lr is the operative CERTIFICATE
-- the thing self-stabilization regulates, whose violation predicts blow-up.
That is the claim v1 could not make.

Lean: num_data=2048, mlp_s, CPU, a few cells.
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
from utils.measure import compute_eigenvalues, EigenvectorCache

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = int(os.environ.get('EOSS_NUM_DATA', '2048'))
OUT_DIR = os.path.join(_REPO, 'results', 'synthetic_arbiter_v2')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


def get_data():
    if 'xy' not in _DATA:
        X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], 888, loss_type='mse')
        _DATA['xy'] = (X, Y)
    return _DATA['xy']


def build():
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets['mlp_s']['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets['mlp_s']['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    return net, SquaredLoss()


def train(net, opt, X, Y, loss_fn, batch, steps):
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return False
        opt.zero_grad(); lo.backward(); opt.step()
    return True


def harvest_batch_lambda(net, X, Y, loss_fn, batch, n_batches=150):
    """h = lambda_max(H_B) per batch, warm-started LOBPCG."""
    cache = EigenvectorCache(1)
    hs = []
    for _ in range(n_batches):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        try:
            lam = compute_eigenvalues(lo, net, k=1, max_iterations=40, reltol=0.02,
                                      eigenvector_cache=cache, return_eigenvectors=False,
                                      use_power_iteration=False).item()
        except Exception:
            continue
        if np.isfinite(lam):
            hs.append(lam)
    return np.array(hs, dtype=np.float64)


# ---------- numpy stability thresholds from harvested h ----------
def _root(f, lo, hi, iters=60):
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (flo <= 0) == (fm <= 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sgd_thresholds(h):
    Eh, Eh2 = np.mean(h), np.mean(h * h)
    eta_L2 = 2 * Eh / Eh2
    hi = 4.0 / (np.median(h) + 1e-12)
    eta_mean = _root(lambda e: np.mean(np.abs(1 - e * h)) - 1.0, 1e-6, hi)
    eta_as = _root(lambda e: np.mean(np.log(np.abs(1 - e * h) + 1e-300)), 1e-6, hi)
    return dict(eta_as=float(eta_as), eta_mean=float(eta_mean), eta_L2=float(eta_L2))


def _companion(hv, eta, b):
    return np.array([[1.0 + b - eta * hv, -b], [1.0, 0.0]])


def mom_lyap(h, eta, b, T_sim=8000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(2); v /= np.linalg.norm(v)
    acc = 0.0
    for i in rng.integers(0, len(h), size=T_sim):
        v = _companion(h[i], eta, b) @ v
        n = np.linalg.norm(v); acc += np.log(n + 1e-300); v /= (n + 1e-300)
    return acc / T_sim


def mom_msq_top(h, eta, b):
    EM2 = np.mean([np.kron(_companion(hv, eta, b), _companion(hv, eta, b)) for hv in h], axis=0)
    return float(np.max(np.abs(np.linalg.eigvals(EM2))))


def mom_thresholds(h, b):
    hi = 5.0 / (np.median(h) + 1e-12)
    eta_as = _root(lambda e: mom_lyap(h, e, b), 1e-6, hi, iters=45)
    eta_L2 = _root(lambda e: mom_msq_top(h, e, b) - 1.0, 1e-6, hi, iters=45)
    return dict(eta_as=float(eta_as), eta_L2=float(eta_L2))


def empirical_divergence_lr(optn, params, batch, lr_grid, steps=1500):
    """Coarse: smallest lr in grid at which the real run blows up (loss>1e6)."""
    X, Y = get_data()
    div_lr = None
    last_ok = None
    for lr in lr_grid:
        net, loss_fn = build()
        opt = create_optimizer(optn, net, lr, params)
        ok = train(net, opt, X, Y, loss_fn, batch, steps)
        if ok:
            last_ok = lr
        else:
            div_lr = lr
            break
    return dict(div_lr=div_lr, last_stable_lr=last_ok)


def main():
    # (tag, opt, params, batch, lr_harvest[near-edge stable], steps, beta, div_lr_grid)
    cells = [
        ("SGD_b8",     "SGD",          {},            8,    0.015, 3000, None, [0.015, 0.017, 0.019, 0.021, 0.024]),
        ("SGD_b2048",  "SGD",          {},            2048, 0.02,  1500, None, [0.02, 0.024, 0.028, 0.032]),
        ("SGDM09_b8",  "SGD-Momentum", {"beta": 0.9}, 8,    0.003, 3000, 0.9,  [0.003, 0.0035, 0.004, 0.0045, 0.005]),
    ]
    out = []
    for tag, optn, params, batch, lrh, steps, beta, div_grid in cells:
        print(f"\n=== {tag}: harvest at lr={lrh} ===", flush=True)
        X, Y = get_data()
        net, loss_fn = build()
        opt = create_optimizer(optn, net, lrh, params)
        if not train(net, opt, X, Y, loss_fn, batch, steps):
            print(f"  {tag}: diverged during harvest-train at lr={lrh}, skip"); out.append(dict(tag=tag, diverged=True)); continue
        h = harvest_batch_lambda(net, X, Y, loss_fn, batch)
        rec = dict(tag=tag, optimizer=optn, batch=batch, beta=beta, lr_harvest=lrh,
                   h_mean=float(h.mean()), h_std=float(h.std()), h_cv=float(h.std() / (h.mean() + 1e-30)),
                   n_h=len(h))
        rec['sgd_thresh'] = sgd_thresholds(h)
        if beta is not None:
            rec['mom_thresh'] = mom_thresholds(h, beta)
        print(f"  harvesting divergence lr (real run)...", flush=True)
        rec['empirical'] = empirical_divergence_lr(optn, params, batch, div_grid)
        out.append(rec)
        st = rec['sgd_thresh']; em = rec['empirical']
        print(f"  {tag}: h_mean={h.mean():.1f} cv={rec['h_cv']:.3f}  "
              f"SGD-pred eta*: as={st['eta_as']:.4f} mean={st['eta_mean']:.4f} L2={st['eta_L2']:.4f}  "
              f"| EMPIRICAL div_lr={em['div_lr']} last_stable={em['last_stable_lr']}", flush=True)
        if beta is not None:
            mt = rec['mom_thresh']
            print(f"        momentum-pred eta*: as={mt['eta_as']:.4f} L2={mt['eta_L2']:.4f}", flush=True)
        with open(os.path.join(OUT_DIR, 'synthetic_arbiter_v2.json'), 'w') as f:
            json.dump(out, f, indent=2)

    print("\n===== CERTIFICATE VERDICT (which predicted eta* matches empirical divergence lr?) =====")
    for r in out:
        if r.get('diverged'):
            print(f"  {r['tag']}: harvest diverged"); continue
        em = r['empirical']; st = r['sgd_thresh']
        print(f"  {r['tag']:12s} empirical div_lr≈{em['div_lr']} (last stable {em['last_stable_lr']})  "
              f"SGD pred: as={st['eta_as']:.4f} mean={st['eta_mean']:.4f} L2={st['eta_L2']:.4f}")
        if 'mom_thresh' in r:
            mt = r['mom_thresh']
            print(f"               momentum pred: as={mt['eta_as']:.4f} L2={mt['eta_L2']:.4f}")
    print(f"\nwrote {OUT_DIR}/synthetic_arbiter_v2.json")


if __name__ == '__main__':
    main()
