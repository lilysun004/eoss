"""
Experiment 3c (fixed frozen-cocycle) -- removes the null-space confound that made v2's
gamma pin at exactly 0.

v2 bug (caught by the "exactly 0 is sus" sanity check): near interpolation the Hessian is
low-rank; its near-null subspace is ~common across batches (the interpolation manifold), so
H_B * delta ~ 0 there and a_t = 1 exactly. The free renormalized tangent drifts into that
flat subspace and the measured top-Lyapunov gets FLOORED at 0 whenever the true unstable
mode is sub-marginal -- it literally cannot go negative, so it cannot measure the margin.
"gamma_ema = +0.0000 +/- 0.0000, flat across c in [0.6,1.1]" was that floor, not marginality.

Fix: work in REDUCED COORDINATES within the top-K CURVED subspace. V = top-K eigenvectors of
H at the operating point (theta_raw or theta_tilde); represent the tangent as a = V^T dtheta,
so the batch Hessian acts as the K x K matrix M_t = V^T H_{B_t} V. No null space -> gamma can
go NEGATIVE and measure the margin. Keeps the multi-dimensional buffer<->noise interaction
(unlike the scalar arbiter) while excluding the flat directions. K=1 recovers the scalar
top-mode exponent E[log|1 - c*eta*h_t|]; we report K=1 and K=8 to see if multi-D matters.

Cross-check target: the (un-confounded) AR-pole result -- small-batch stateful cells damped
(rho ~ 0.9 < 1) => gamma should come out NEGATIVE for them; at-edge cells (SGD/SGDM b2048,
rho ~ 1) => gamma ~ 0 with c* ~ 1.
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

import experiments.long_train_grid as L
from experiments.frozen_cocycle_v2 import train_with_snapshots
from utils.optimizer import create_optimizer
from utils.measure import (create_hessian_vector_product, flatt, param_vector,
                           compute_eigenvalues, EigenvectorCache)
from utils.curvature_segment import set_params_inplace

OUT_DIR = os.path.join(_REPO, 'results', 'frozen_cocycle_v3')
os.makedirs(OUT_DIR, exist_ok=True)
C_GRID = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0])
EXP2_DIV_MULT = {'SGD_b8': 1.6, 'SGDM09_b8': 1.2, 'Adam_b8': 2.0, 'SGD_b2048': None, 'SGDM09_b2048': None}


def top_k_basis(net, loss_fn, X, Y, k, cap=2048):
    Xs, Ys = (X, Y) if len(X) <= cap else (X[:cap], Y[:cap])
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    eigval, eigvec = compute_eigenvalues(lo, net, k=k, max_iterations=80, reltol=0.01,
                                         eigenvector_cache=EigenvectorCache(k),
                                         return_eigenvectors=True, use_power_iteration=False)
    return eigvec.detach(), eigval.detach()   # [n,k], [k]


def reduced_M(net, loss_fn, Xb, Yb, V):
    """M = V^T H_B V  (K x K), via one batched HVP on the K basis columns."""
    params = [p for p in net.parameters() if p.requires_grad]
    pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
    grads = T.autograd.grad(lo, params, create_graph=True)
    hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=flatt(grads))
    try:
        HV = T.stack([hvp(V[:, i], retain_graph_override=(i < V.shape[1]-1)) for i in range(V.shape[1])], dim=1)
    finally:
        hvp.free_memory()
    return (V.t() @ HV).detach()   # [k,k]


def reduced_pinv(V, pinv_vec):
    return (V.t() @ (pinv_vec.unsqueeze(1) * V)).detach()   # [k,k], V^T diag(pinv) V


def cocycle_reduced(net, loss_fn, XY, optn, params, batch, eta, V, steps=1500, burn=300,
                    n_blocks=4, seed=0, pinv_red=None):
    """Reduced-subspace frozen cocycle. State a in R^K (per c-column), optional buffer in R^K.
    Returns gamma(c) block-mean/std."""
    X, Y = XY; k = V.shape[1]; nc = len(C_GRID); dev = V.device
    beta = float(params.get('beta', 0.0)); is_mom = (optn == 'SGD-Momentum')
    is_adam = (optn == 'Adam'); b1 = float(params.get('beta1', 0.9)) if is_adam else 0.0
    g = T.Generator().manual_seed(seed)
    a = T.randn(k, nc, generator=g).to(dev); a /= a.norm(dim=0, keepdim=True)
    buf = T.zeros(k, nc, device=dev) if (is_mom or is_adam) else None
    cvec = T.tensor(C_GRID, dtype=a.dtype, device=dev) * eta
    logs = [[] for _ in range(nc)]
    for t in range(steps):
        idx = T.randperm(len(X), generator=g)[:batch]
        M = reduced_M(net, loss_fn, X[idx], Y[idx], V)      # [k,k]
        Ma = M @ a                                          # [k,nc]
        if is_mom:
            buf = beta * buf + Ma; a_new = a - cvec * buf
        elif is_adam:
            buf = b1 * buf + (1 - b1) * Ma
            a_new = a - cvec * (pinv_red @ buf)
        else:
            a_new = a - cvec * Ma
        if buf is not None:
            norm = T.sqrt((a_new**2).sum(0) + (buf**2).sum(0)) + 1e-30
        else:
            norm = a_new.norm(dim=0) + 1e-30
        lg = np.log(norm.cpu().numpy())
        if t >= burn:
            for j in range(nc):
                logs[j].append(lg[j])
        a = a_new / norm
        if buf is not None:
            buf = buf / norm
    gm = np.zeros(nc); gs = np.zeros(nc)
    for j in range(nc):
        arr = np.array(logs[j]); bl = np.array_split(arr, n_blocks)
        bm = np.array([b.mean() for b in bl if len(b)])
        gm[j] = bm.mean(); gs[j] = bm.std()
    return gm, gs


def find_cstar(gm):
    for i in range(len(C_GRID)-1):
        if gm[i] <= 0 <= gm[i+1]:
            f = (0 - gm[i]) / (gm[i+1] - gm[i] + 1e-30)
            return float(C_GRID[i] + f*(C_GRID[i+1]-C_GRID[i]))
    return float(C_GRID[0]) if gm[0] > 0 else float('nan')


def eval_point(net, loss_fn, XY, optn, params, batch, eta, theta, K, pinv_vec, seed):
    set_params_inplace(net, theta)
    X, Y = XY
    V, eigval = top_k_basis(net, loss_fn, X, Y, K)
    pinv_red = reduced_pinv(V, pinv_vec) if pinv_vec is not None else None
    Tc = 1500 if batch < 100 else 1000
    gm, gs = cocycle_reduced(net, loss_fn, XY, optn, params, batch, eta, V, steps=Tc, seed=seed, pinv_red=pinv_red)
    # also K=1 scalar top-mode reference
    V1 = V[:, :1]
    pr1 = reduced_pinv(V1, pinv_vec) if pinv_vec is not None else None
    gm1, gs1 = cocycle_reduced(net, loss_fn, XY, optn, params, batch, eta, V1, steps=Tc, seed=seed+7, pinv_red=pr1)
    return dict(gammaK=gm.tolist(), gammaK_sem=(gs/2).tolist(), cstarK=find_cstar(gm),
                gamma1=gm1.tolist(), cstar1=find_cstar(gm1),
                lam_top=float(eigval[0].item()), eta_lam=float(eta*eigval[0].item()))


def main():
    K = 8
    cells = [
        ("SGD_b8",    "SGD",          {},                            8,    0.01,  20000),
        ("SGDM09_b8", "SGD-Momentum", {"beta": 0.9},                 8,    0.002, 20000),
        ("Adam_b8",   "Adam",         {"beta1": 0.9, "beta2": 0.99}, 8,    0.001, 20000),
        ("SGD_b2048", "SGD",          {},                            2048, 0.02,  6000),
        ("SGDM09_b2048","SGD-Momentum",{"beta": 0.9},                2048, 0.006, 6000),
    ]
    i1 = int(np.argmin(np.abs(C_GRID - 1.0)))
    results = []
    for tag, optn, params, batch, lr, steps in cells:
        print(f"\n=== {tag}: {optn} b={batch} lr={lr} K={K} ===", flush=True)
        snap_at = {steps - 3000, steps - 1}
        net, loss_fn, opt, snaps, done = train_with_snapshots(optn, params, batch, lr, steps, snap_at)
        if net is None:
            print(f"  {tag}: diverged @ {done}"); results.append(dict(tag=tag, diverged=True)); continue
        XY = L.get_data(); dev = next(net.parameters()).device
        pinv_vec = None
        if optn == 'Adam':
            dsq = opt.get_preconditioner_inv_sqrt(); pinv_vec = (dsq.to(dev)**2) if dsq is not None else None
        raw_list, ema_list = [], []
        for (st, th_raw, th_ema) in snaps:
            raw_list.append(eval_point(net, loss_fn, XY, optn, params, batch, lr, th_raw, K, pinv_vec, seed=st))
            ema_list.append(eval_point(net, loss_fn, XY, optn, params, batch, lr, th_ema, K, pinv_vec, seed=st+1))
        def agg(lst, key):
            return np.mean([np.array(d[key]) for d in lst], axis=0)
        gK_raw = agg(raw_list, 'gammaK'); gK_ema = agg(ema_list, 'gammaK')
        g1_raw = agg(raw_list, 'gamma1'); g1_ema = agg(ema_list, 'gamma1')
        rec = dict(tag=tag, optimizer=optn, batch=batch, lr=lr, K=K, c_grid=C_GRID.tolist(),
                   gammaK_raw=gK_raw.tolist(), gammaK_ema=gK_ema.tolist(),
                   gamma1_raw=g1_raw.tolist(), gamma1_ema=g1_ema.tolist(),
                   gammaK_raw_c1=float(gK_raw[i1]), gammaK_ema_c1=float(gK_ema[i1]),
                   gamma1_ema_c1=float(g1_ema[i1]),
                   cstarK_raw=find_cstar(gK_raw), cstarK_ema=find_cstar(gK_ema),
                   cstar1_ema=find_cstar(g1_ema),
                   eta_lam_raw=float(np.mean([d['eta_lam'] for d in raw_list])),
                   eta_lam_ema=float(np.mean([d['eta_lam'] for d in ema_list])),
                   exp2_div_mult=EXP2_DIV_MULT.get(tag))
        results.append(rec)
        print(f"  {tag}: gammaK_ema(1)={rec['gammaK_ema_c1']:+.4f}  gamma1_ema(1)={rec['gamma1_ema_c1']:+.4f}  "
              f"c*K_ema={rec['cstarK_ema']:.3f}  eta*lam_ema={rec['eta_lam_ema']:.2f}  [Exp2 mult={rec['exp2_div_mult']}]", flush=True)
        print("   gammaK_ema(c): " + " ".join(f"{g:+.3f}" for g in gK_ema), flush=True)
        with open(os.path.join(OUT_DIR, 'frozen_cocycle_v3.json'), 'w') as f:
            json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in results:
        if r.get('diverged'): continue
        ax.plot(r['c_grid'], r['gammaK_ema'], '-o', ms=4, label=f"{r['tag']} (c*={r['cstarK_ema']:.2f})")
    ax.axhline(0, color='k', ls='--', lw=0.8); ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('lr multiplier c'); ax.set_ylabel('reduced-subspace frozen gamma (EMA theta)')
    ax.set_title(f'Fixed frozen cocycle (top-{K} subspace, no null-space floor)')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'frozen_cocycle_v3.png'), dpi=90); plt.close(fig)

    print("\n===== VERDICT (reduced-subspace, can go negative) =====")
    print(f"{'cell':14s} {'gammaK_ema(1)':>13s} {'gamma1_ema(1)':>13s} {'c*K_ema':>8s} {'eta*lam_ema':>11s} {'Exp2mult':>8s}")
    for r in results:
        if r.get('diverged'):
            print(f"{r['tag']:14s}  diverged"); continue
        print(f"{r['tag']:14s} {r['gammaK_ema_c1']:+13.4f} {r['gamma1_ema_c1']:+13.4f} "
              f"{r['cstarK_ema']:8.3f} {r['eta_lam_ema']:11.2f} {str(r['exp2_div_mult']):>8s}")
    print("\n Cross-check vs AR: small-batch stateful gamma<0 (damped, matches rho<1)? at-edge gamma~0, c*~1?")
    print(" Cross-val: momentum c* ~ Exp2 div_mult (~1.2)?  SGD c* < its div_mult (cubic budget)?")


if __name__ == '__main__':
    main()
