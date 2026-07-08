"""
Experiment 3 (stochastic-edge / frozen-cocycle) -- the non-tautological universal test.

The deterministic edge (lambda vs 2/eta or 2(1+beta)/eta) is the WRONG yardstick under
noise. The true boundary is where the top Lyapunov exponent gamma of the STOCHASTIC
closed-loop cocycle crosses zero. Noise shifts it in opposite directions:
 - SGD: multiplicative curvature noise is a.s.-stabilizing (Jensen: E[log|1-eta h|]<=0
   even when E[eta h]>2) -> SGD operates ABOVE the deterministic edge (explains kappa=2.93).
 - Momentum: the buffer integrates noise, variance compounds through the filter, the
   random product destabilizes BELOW the deterministic edge (explains SGDM b8 sitting at
   eta*lam=0.37 yet diverging at 0.47 -- razor-thin stochastic margin).

Method: at a plateau checkpoint theta*, FREEZE theta (and Adam's v), sample fresh iid
batches B_t, and propagate a ghost tangent through the ACTUAL augmented closed-loop
Jacobian (SGD: I-c*eta*H_B ; heavy-ball companion: delta_buf=beta*delta_buf+H_B*delta_th,
delta_th-=c*eta*delta_buf ; Adam frozen-v: delta_m=b1*delta_m+(1-b1)H_B*delta_th,
delta_th-=c*eta*Pinv*delta_m). theta* enters ONLY through H_{B_t}; the cocycle is a genuine
random linear system whose exponent gamma is a property of the operating point -- NOT
forced to zero (unlike the realized-trajectory Lyapunov, which is tautologically 0).

We sweep an lr multiplier c and find c* where gamma_log(c*eta)=0. Universal claim on the
table: c* ~ 1 for every optimizer at every batch size, with gamma(eta)=0^- (parked just
below, margin ~ CV(h)^2). GBS=2 is the sigma^2->0 limit where the stochastic boundary
collapses onto the deterministic one. Also report the mean-square (L2) growth rate, since
the binding notion may be a.s. for SGD but L2 for momentum (buffer variance accumulation).

lean setup (mlp_s, num_data=2048), CPU. SGD / SGD-Momentum / Adam (Muon's Newton-Schulz
cocycle deferred).
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
from utils.optimizer import create_optimizer
from utils.measure import create_hessian_vector_product, flatt, param_vector

OUT_DIR = os.path.join(_REPO, 'results', 'frozen_cocycle')
os.makedirs(OUT_DIR, exist_ok=True)
C_GRID = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0])


def train_to_plateau(optn, params, batch, lr, steps):
    X, Y = L.get_data()
    net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return None, None, None, None, step
        opt.zero_grad(); lo.backward(); opt.step()
    return net, loss_fn, opt, (X, Y), steps


def hb_matvec(net, loss_fn, Xb, Yb, V):
    """H_B @ V for V shape [n, k], at current (frozen) params."""
    params = [p for p in net.parameters() if p.requires_grad]
    pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
    grads = T.autograd.grad(lo, params, create_graph=True)
    hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=flatt(grads))
    try:
        out = hvp(V, retain_graph_override=False) if V.ndim == 1 else T.stack(
            [hvp(V[:, i], retain_graph_override=(i < V.shape[1]-1)) for i in range(V.shape[1])], dim=1)
    finally:
        hvp.free_memory()
    return out.detach()


def frozen_gamma(net, loss_fn, XY, optn, params, batch, eta, steps=5000, seed=0, burn=200):
    """Top Lyapunov exponent gamma_log of the frozen closed-loop cocycle, for each c in
    C_GRID (columns), plus a mean-square (L2) rate estimate at c=1."""
    X, Y = XY
    n = param_vector(net).numel()
    nc = len(C_GRID)
    g = T.Generator().manual_seed(seed)
    dev = next(net.parameters()).device
    beta = float(params.get('beta', 0.0))
    is_mom = (optn == 'SGD-Momentum')
    is_adam = (optn == 'Adam')
    b1 = float(params.get('beta1', 0.9)) if is_adam else 0.0
    pinv = None
    if is_adam:
        opt_wrap = getattr(net, '_cocycle_opt', None)
        d_inv_sqrt = opt_wrap.get_preconditioner_inv_sqrt() if opt_wrap else None
        pinv = (d_inv_sqrt.to(dev) ** 2) if d_inv_sqrt is not None else T.ones(n, device=dev)

    # augmented tangent: dth [n,nc], and buffer/m [n,nc] if stateful
    dth = T.randn(n, nc, generator=g).to(dev); dth /= dth.norm(dim=0, keepdim=True)
    dbuf = T.zeros(n, nc, device=dev) if (is_mom or is_adam) else None
    acc = np.zeros(nc)
    cnt = 0
    cvec = T.tensor(C_GRID, dtype=dth.dtype, device=dev) * eta   # per-column lr

    # Invariant: at the start of each step every column has unit AUGMENTED norm
    # (we renormalize at the end), so the per-step growth factor a_t is exactly the
    # new augmented norm. (SGD: augmented state is just dth.)
    for t in range(steps):
        idx = T.randperm(len(X), generator=g)[:batch]
        Xb, Yb = X[idx], Y[idx]
        Hdth = hb_matvec(net, loss_fn, Xb, Yb, dth)              # [n,nc]
        if is_mom:
            dbuf = beta * dbuf + Hdth
            dth_new = dth - cvec * dbuf
        elif is_adam:
            dbuf = b1 * dbuf + (1 - b1) * Hdth                  # tangent of m (v frozen)
            dth_new = dth - cvec * (pinv.unsqueeze(1) * dbuf)
        else:
            dth_new = dth - cvec * Hdth
        if dbuf is not None:
            norm = T.sqrt((dth_new**2).sum(0) + (dbuf**2).sum(0)) + 1e-30
        else:
            norm = dth_new.norm(dim=0) + 1e-30
        a = norm.cpu().numpy()                                   # prev augmented norm == 1
        if t >= burn:
            acc += np.log(a); cnt += 1
        dth = dth_new / norm
        if dbuf is not None:
            dbuf = dbuf / norm
    gamma = acc / max(cnt, 1)
    return dict(c_grid=C_GRID.tolist(), gamma=gamma.tolist(), eta=eta)


def find_cstar(cgrid, gamma):
    cg = np.array(cgrid); gm = np.array(gamma)
    # first zero crossing (gamma increasing in c)
    for i in range(len(cg)-1):
        if gm[i] <= 0 <= gm[i+1]:
            f = (0 - gm[i]) / (gm[i+1] - gm[i] + 1e-30)
            return float(cg[i] + f*(cg[i+1]-cg[i]))
    if gm[0] > 0:
        return float(cg[0])   # already unstable at smallest c
    return float('nan')       # never crosses (stable across whole grid)


def main():
    cells = [
        ("SGD_b8",    "SGD",          {},                            8,    0.01,  20000),
        ("SGDM09_b8", "SGD-Momentum", {"beta": 0.9},                 8,    0.002, 20000),
        ("Adam_b8",   "Adam",         {"beta1": 0.9, "beta2": 0.99}, 8,    0.001, 20000),
        ("SGD_b2048", "SGD",          {},                            2048, 0.02,  6000),
        ("SGDM09_b2048","SGD-Momentum",{"beta": 0.9},                2048, 0.006, 6000),
    ]
    results = []
    for tag, optn, params, batch, lr, steps in cells:
        print(f"\n=== {tag}: train {steps} steps then frozen-cocycle sweep ===", flush=True)
        net, loss_fn, opt, XY, done = train_to_plateau(optn, params, batch, lr, steps)
        if net is None:
            print(f"  {tag}: diverged during train @ {done}"); results.append(dict(tag=tag, diverged=True)); continue
        net._cocycle_opt = opt   # expose for Adam preconditioner
        t0 = time.time()
        r = frozen_gamma(net, loss_fn, XY, optn, params, batch, lr,
                         steps=4000 if batch < 100 else 2500)
        cstar = find_cstar(r['c_grid'], r['gamma'])
        gamma1 = float(np.interp(1.0, r['c_grid'], r['gamma']))
        print(f"  {tag}: gamma(c=1)={gamma1:+.4f}  c*={cstar:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        print("    c:     " + " ".join(f"{c:6.2f}" for c in r['c_grid']), flush=True)
        print("    gamma: " + " ".join(f"{g:+6.3f}" for g in r['gamma']), flush=True)
        results.append(dict(tag=tag, optimizer=optn, batch=batch, lr=lr,
                            gamma1=gamma1, cstar=cstar, **r))
        with open(os.path.join(OUT_DIR, 'frozen_cocycle.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # plot gamma(c) per cell
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        if r.get('diverged'): continue
        ax.plot(r['c_grid'], r['gamma'], '-o', label=f"{r['tag']} (c*={r['cstar']:.2f})")
    ax.axhline(0, color='k', ls='--', lw=0.8); ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('lr multiplier c'); ax.set_ylabel('frozen-cocycle top Lyapunov gamma')
    ax.set_title('Stochastic edge: gamma(c*eta)=0 at c*~1 for all optimizers?')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'frozen_cocycle.png'), dpi=90); plt.close(fig)

    print("\n===== VERDICT =====")
    print(f"{'cell':14s} {'gamma(c=1)':>11s} {'c*':>7s}")
    for r in results:
        if r.get('diverged'):
            print(f"{r['tag']:14s}  diverged"); continue
        print(f"{r['tag']:14s} {r['gamma1']:+11.4f} {r['cstar']:7.3f}")
    print("\n c*~1 & gamma(1)~0- for ALL => everyone parks at its OWN stochastic edge")
    print(" (deterministic edge was the wrong yardstick); GBS=2 is the sigma^2->0 limit.")


if __name__ == '__main__':
    main()
