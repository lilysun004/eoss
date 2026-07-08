"""
Experiment (2): synthetic arbiter (suggested.txt ranked experiment #8).

Harvest the empirical distribution of directional batch curvature
    h_t = u^T H_{B} u ,   u = top eigenvector of the full (subset) Hessian at a
plateau checkpoint, B ranging over many random mini-batches.
Then, with NO deep-learning confounders, simulate the linear recursion in the
unstable mode and ask which member of the Lyapunov hierarchy is exactly marginal
at the learning rate the real optimizer actually uses (eta_real).

SGD scalar mode:   x_{t+1} = (1 - eta h_t) x_t
  a.s./log:      E[log|1 - eta h|]        == 0 ?
  mean:          E[|1 - eta h|]           == 1 ?
  mean-square:   E[(1 - eta h)^2]         == 1 ?   (<=> eta = 2 E[h]/E[h^2])

Heavy-ball companion mode:  v_{t+1} = M_t v_t,  M_t = [[1+b-eta h_t, -b],[1,0]]
  a.s.:          top Lyapunov exponent of the random product (simulated) == 0 ?
  mean-square:   top eigenvalue of E[M ⊗ M] == 1 ?
  (annealed):    spectral radius of E[M] == 1 ?

For each (optimizer,batch) cell we print all three condition-values at eta_real;
whichever is ~0 (SGD) / ~1 (momentum) is the operative invariant. Large batch =>
low Var(h) => all three coincide (the known large-batch collapse). Small batch =>
they fan out, and the one sitting at marginal is the answer.
"""
import os, sys, time, json
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
from utils.measure import (compute_eigenvalues, EigenvectorCache,
                           create_hessian_vector_product, flatt)

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
OUT_DIR = os.path.join(_REPO, 'results', 'synthetic_arbiter')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


def get_data():
    if 'xy' not in _DATA:
        X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, 2048, [], 888, loss_type='mse')
        _DATA['xy'] = (X, Y)
    return _DATA['xy']


def build():
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets['mlp_s']['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets['mlp_s']['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    return net, SquaredLoss()


def harvest_h(net, X, Y, loss_fn, batch_size, n_batches=200, subset_cap=2048):
    """Return array of h = u^T H_B u over random batches, u = full-H top eigvec."""
    params = [p for p in net.parameters() if p.requires_grad]
    # top eigvec of the full (subset) Hessian at the current iterate
    Xs, Ys = (X, Y) if len(X) <= subset_cap else (X[:subset_cap], Y[:subset_cap])
    preds = net(Xs).squeeze(-1); loss = loss_fn(preds, Ys)
    _, u = compute_eigenvalues(loss, net, k=1, max_iterations=60, reltol=0.01,
                               eigenvector_cache=EigenvectorCache(1),
                               return_eigenvectors=True, use_power_iteration=False)
    u = u.detach(); u = u / u.norm()
    hs = []
    N = len(X)
    for _ in range(n_batches):
        idx = T.randperm(N)[:batch_size]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        grads = T.autograd.grad(lo, params, create_graph=True)
        hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=flatt(grads))
        try:
            Hu = hvp(u, retain_graph_override=False)
        finally:
            hvp.free_memory()
        hs.append(T.dot(u, Hu).item())
    return np.array(hs, dtype=np.float64)


def train_to_plateau(optname, params, batch_size, lr, steps):
    X, Y = get_data()
    net, loss_fn = build()
    opt = create_optimizer(optname, net, lr, params)
    for step in range(steps):
        idx = T.randperm(len(X))[:batch_size]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return None, None, None
        opt.zero_grad(); lo.backward(); opt.step()
        if step % max(1, steps // 4) == 0:
            print(f"    train step {step}/{steps} loss={lo.item():.4f}", flush=True)
    return net, loss_fn, (X, Y)


# ---- numpy stability conditions on the harvested h-distribution ----
def sgd_conditions(h, eta):
    a = 1.0 - eta * h
    return dict(
        log=float(np.mean(np.log(np.abs(a) + 1e-300))),        # ==0 at a.s. margin
        mean=float(np.mean(np.abs(a)) - 1.0),                   # ==0 at mean margin
        msq=float(np.mean(a * a) - 1.0),                        # ==0 at L2 margin
        eta_msq_thresh=float(2 * np.mean(h) / np.mean(h * h)),  # eta making E[a^2]=1
    )


def _companion(hval, eta, b):
    return np.array([[1.0 + b - eta * hval, -b], [1.0, 0.0]])


def momentum_conditions(h, eta, b, T_sim=20000, seed=0):
    rng = np.random.default_rng(seed)
    # a.s. Lyapunov via random matrix product
    v = rng.standard_normal(2); v /= np.linalg.norm(v)
    acc = 0.0
    idxs = rng.integers(0, len(h), size=T_sim)
    for i in idxs:
        v = _companion(h[i], eta, b) @ v
        nrm = np.linalg.norm(v)
        acc += np.log(nrm + 1e-300)
        v /= (nrm + 1e-300)
    gamma = acc / T_sim
    # mean-square: top eigenvalue of E[M ⊗ M]
    EM2 = np.zeros((4, 4))
    for hv in h:
        M = _companion(hv, eta, b)
        EM2 += np.kron(M, M)
    EM2 /= len(h)
    msq_top = float(np.max(np.abs(np.linalg.eigvals(EM2))))
    # annealed: spectral radius of mean matrix
    Mbar = _companion(np.mean(h), eta, b)
    rho_mean = float(np.max(np.abs(np.linalg.eigvals(Mbar))))
    return dict(log_gamma=float(gamma), msq_top=msq_top - 1.0, rho_mean=rho_mean - 1.0)


def main():
    # (tag, optimizer, params, batch, lr, steps, beta_for_companion)
    cells = [
        ("SGD_b8",      "SGD",          {},            8,    0.01,  3000, None),
        ("SGD_b2048",   "SGD",          {},            2048, 0.02,  1500, None),
        ("SGDM09_b8",   "SGD-Momentum", {"beta": 0.9}, 8,    0.002, 3500, 0.9),
        ("SGDM09_b2048","SGD-Momentum", {"beta": 0.9}, 2048, 0.006, 4000, 0.9),
    ]
    out = []
    for tag, optn, params, batch, lr, steps, beta in cells:
        print(f"\n=== harvest {tag}: {optn} b={batch} lr={lr} ===", flush=True)
        net, loss_fn, XY = train_to_plateau(optn, params, batch, lr, steps)
        if net is None:
            print(f"  {tag}: DIVERGED during harvest train"); out.append(dict(tag=tag, diverged=True)); continue
        X, Y = XY
        h = harvest_h(net, X, Y, loss_fn, batch, n_batches=200)
        rec = dict(tag=tag, optimizer=optn, batch=batch, lr=lr, beta=beta,
                   h_mean=float(h.mean()), h_std=float(h.std()), h_cv=float(h.std() / (abs(h.mean()) + 1e-30)),
                   Eh2_over_Eh=float(np.mean(h * h) / (np.mean(h) + 1e-30)))
        sgd = sgd_conditions(h, lr)
        rec['sgd_cond'] = sgd
        rec['eta_real_x_Eh'] = float(lr * h.mean())            # ~2 at SGD edge
        if beta is not None:
            rec['mom_cond'] = momentum_conditions(h, lr, beta)
        out.append(rec)
        print(f"  {tag}: h_mean={h.mean():.2f} h_cv={rec['h_cv']:.3f}  "
              f"lr*E[h]={rec['eta_real_x_Eh']:.3f}  "
              f"SGD margins log={sgd['log']:+.4f} mean={sgd['mean']:+.4f} msq={sgd['msq']:+.4f}", flush=True)
        if beta is not None:
            m = rec['mom_cond']
            print(f"        momentum margins: log_gamma={m['log_gamma']:+.4f} "
                  f"msq={m['msq_top']:+.4f} rho_mean={m['rho_mean']:+.4f}", flush=True)

    with open(os.path.join(OUT_DIR, 'synthetic_arbiter.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n===== SUMMARY (margin ~0 => that condition is the operative invariant) =====")
    print("SGD cells -- three scalar-mode margins at eta_real:")
    for r in out:
        if r.get('diverged') or 'sgd_cond' not in r:
            continue
        s = r['sgd_cond']
        print(f"  {r['tag']:14s} h_cv={r['h_cv']:.3f}  log={s['log']:+.4f}  mean={s['mean']:+.4f}  msq={s['msq']:+.4f}")
    print("momentum cells -- companion-mode margins at eta_real:")
    for r in out:
        if r.get('diverged') or 'mom_cond' not in r:
            continue
        m = r['mom_cond']
        print(f"  {r['tag']:14s} h_cv={r['h_cv']:.3f}  log_gamma={m['log_gamma']:+.4f}  "
              f"msq={m['msq_top']:+.4f}  rho_mean={m['rho_mean']:+.4f}")
    print(f"\nwrote {OUT_DIR}/synthetic_arbiter.json")


if __name__ == '__main__':
    main()
