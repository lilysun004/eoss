"""
Experiment 3b (rigorous frozen-cocycle) -- addresses the reviewer's demands so that
"is gamma zero" becomes a decomposition with error bars and a cross-validated c*.

Delivers, per cell:
  * gamma(c) CURVE + zero-crossing c*, not just gamma(1). Cross-validation target:
    momentum b8 c* should match its Exp-2 empirical divergence multiplier (~1.14);
    SGD's c* should sit BELOW its divergence multiplier (the gap = cubic extension).
  * Phase control: gamma evaluated at the RAW plateau checkpoint AND at the EMA-smoothed
    point theta_tilde (proxy for the constrained/central trajectory). If gamma(1)>0 at raw
    but ~0 at theta_tilde, the positive reading was oscillation phase, not physics.
  * Error bars: block-means over the cocycle + averaging over several plateau checkpoints.
  * Convergence-in-T saved (a +0.08 with +/-0.1 wander is a zero).
  * Three-object taxonomy:
      (i)   realized growth of the bounded coordinate    = 0 (tautological)
      (ii)  along-trajectory linearized cocycle (Jacobians at the MOVING theta_t) -- keeps
            the curvature<->state feedback
      (iii) frozen-point cocycle (this experiment's gamma) -- pure operating-point property
    gap (iii)-(i) = total nonlinear stabilization budget; (ii)-(iii) = feedback-correlation.

lean setup (mlp_s, num_data=2048), CPU. SGD / SGD-Momentum / Adam (Muon deferred).
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
from utils.curvature_segment import set_params_inplace

OUT_DIR = os.path.join(_REPO, 'results', 'frozen_cocycle_v2')
os.makedirs(OUT_DIR, exist_ok=True)
C_GRID = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0])
# operating lr and Exp-2 empirical divergence multiplier (divergence_lr / operating_lr)
EXP2_DIV_MULT = {'SGD_b8': 1.6, 'SGDM09_b8': 1.2, 'Adam_b8': 2.0, 'SGD_b2048': None, 'SGDM09_b2048': None}


def hb_matvec(net, loss_fn, Xb, Yb, V):
    params = [p for p in net.parameters() if p.requires_grad]
    pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
    grads = T.autograd.grad(lo, params, create_graph=True)
    hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=flatt(grads))
    try:
        out = T.stack([hvp(V[:, i], retain_graph_override=(i < V.shape[1]-1)) for i in range(V.shape[1])], dim=1)
    finally:
        hvp.free_memory()
    return out.detach()


def cocycle_gamma(net, loss_fn, XY, optn, params, batch, eta, steps=2500, burn=500,
                  n_blocks=4, seed=0, pinv=None):
    """Frozen-cocycle top-Lyapunov gamma(c) with per-block estimates. net params are
    already set to the desired (raw or EMA) operating point."""
    X, Y = XY
    n = param_vector(net).numel(); nc = len(C_GRID); dev = next(net.parameters()).device
    beta = float(params.get('beta', 0.0)); is_mom = (optn == 'SGD-Momentum')
    is_adam = (optn == 'Adam'); b1 = float(params.get('beta1', 0.9)) if is_adam else 0.0
    g = T.Generator().manual_seed(seed)
    dth = T.randn(n, nc, generator=g).to(dev); dth /= dth.norm(dim=0, keepdim=True)
    dbuf = T.zeros(n, nc, device=dev) if (is_mom or is_adam) else None
    cvec = T.tensor(C_GRID, dtype=dth.dtype, device=dev) * eta
    logs = [[] for _ in range(nc)]       # per-column log-growth series (post-burn)
    cum = []                             # convergence-in-T for c=1 column
    i1 = int(np.argmin(np.abs(C_GRID - 1.0)))
    for t in range(steps):
        idx = T.randperm(len(X), generator=g)[:batch]
        Xb, Yb = X[idx], Y[idx]
        Hdth = hb_matvec(net, loss_fn, Xb, Yb, dth)
        if is_mom:
            dbuf = beta * dbuf + Hdth; dth_new = dth - cvec * dbuf
        elif is_adam:
            dbuf = b1 * dbuf + (1 - b1) * Hdth; dth_new = dth - cvec * (pinv.unsqueeze(1) * dbuf)
        else:
            dth_new = dth - cvec * Hdth
        if dbuf is not None:
            norm = T.sqrt((dth_new**2).sum(0) + (dbuf**2).sum(0)) + 1e-30
        else:
            norm = dth_new.norm(dim=0) + 1e-30
        a = np.log(norm.cpu().numpy())
        if t >= burn:
            for j in range(nc):
                logs[j].append(a[j])
            cum.append(a[i1])
        dth = dth_new / norm
        if dbuf is not None:
            dbuf = dbuf / norm
    # per-column: block means -> mean +/- std
    gam_mean = np.zeros(nc); gam_std = np.zeros(nc)
    for j in range(nc):
        arr = np.array(logs[j]); bl = np.array_split(arr, n_blocks)
        bm = np.array([b.mean() for b in bl if len(b)])
        gam_mean[j] = bm.mean(); gam_std[j] = bm.std()
    conv = np.cumsum(cum) / (np.arange(len(cum)) + 1)   # running gamma for c=1
    return gam_mean, gam_std, conv


def along_traj_gamma(optn, params, batch, lr, warm_steps, meas_steps, seed=0):
    """Object (ii): tangent through the cocycle while theta KEEPS UPDATING (real training)
    -- includes the curvature<->state feedback. gamma at c=1 only."""
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    for _ in range(warm_steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6: return float('nan')
        opt.zero_grad(); lo.backward(); opt.step()
    n = param_vector(net).numel(); dev = next(net.parameters()).device
    beta = float(params.get('beta', 0.0)); is_mom = (optn == 'SGD-Momentum')
    is_adam = (optn == 'Adam'); b1 = float(params.get('beta1', 0.9)) if is_adam else 0.0
    g = T.Generator().manual_seed(seed)
    dth = T.randn(n, 1, generator=g).to(dev); dth /= dth.norm()
    dbuf = T.zeros(n, 1, device=dev) if (is_mom or is_adam) else None
    acc, cnt = 0.0, 0
    for t in range(meas_steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        # tangent update at CURRENT theta, then a real optimizer step (theta moves)
        Hdth = hb_matvec(net, loss_fn, Xb, Yb, dth)
        pinv = None
        if is_adam:
            dsq = opt.get_preconditioner_inv_sqrt(); pinv = (dsq.to(dev)**2) if dsq is not None else T.ones(n, device=dev)
        if is_mom:
            dbuf = beta * dbuf + Hdth; dth_new = dth - lr * dbuf
        elif is_adam:
            dbuf = b1 * dbuf + (1 - b1) * Hdth; dth_new = dth - lr * (pinv.unsqueeze(1) * dbuf)
        else:
            dth_new = dth - lr * Hdth
        norm = (T.sqrt((dth_new**2).sum() + (dbuf**2).sum()) if dbuf is not None else dth_new.norm()) + 1e-30
        if t >= 200:
            acc += float(np.log(norm.cpu().numpy())); cnt += 1
        dth = dth_new / norm
        if dbuf is not None: dbuf = dbuf / norm
        # real step
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6: break
        opt.zero_grad(); lo.backward(); opt.step()
    return acc / max(cnt, 1)


def find_cstar(gm):
    for i in range(len(C_GRID)-1):
        if gm[i] <= 0 <= gm[i+1]:
            f = (0 - gm[i]) / (gm[i+1] - gm[i] + 1e-30)
            return float(C_GRID[i] + f*(C_GRID[i+1]-C_GRID[i]))
    return float(C_GRID[0]) if gm[0] > 0 else float('nan')


def train_with_snapshots(optn, params, batch, lr, steps, snap_at, ema_hl=200):
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    ema = param_vector(net).clone(); alpha = 1 - 0.5 ** (1.0 / ema_hl)
    snaps = []
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return None, None, None, None, step
        opt.zero_grad(); lo.backward(); opt.step()
        with T.no_grad():
            ema.mul_(1 - alpha).add_(alpha * param_vector(net))
        if step in snap_at:
            snaps.append((step, param_vector(net).clone(), ema.clone()))
    return net, loss_fn, opt, snaps, steps


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
        print(f"\n=== {tag}: {optn} b={batch} lr={lr} ===", flush=True)
        snap_at = set(range(steps - 6000, steps, 2000)) | {steps - 1}
        net, loss_fn, opt, snaps, done = train_with_snapshots(optn, params, batch, lr, steps, snap_at)
        if net is None:
            print(f"  {tag}: diverged @ {done}"); results.append(dict(tag=tag, diverged=True)); continue
        XY = L.get_data(); dev = next(net.parameters()).device
        pinv = None
        if optn == 'Adam':
            dsq = opt.get_preconditioner_inv_sqrt(); pinv = (dsq.to(dev)**2) if dsq is not None else None
        Tc = 2500 if batch < 100 else 1500
        raw_g, ema_g, conv_save = [], [], None
        for (st, th_raw, th_ema) in snaps:
            set_params_inplace(net, th_raw)
            gmR, gsR, convR = cocycle_gamma(net, loss_fn, XY, optn, params, batch, lr, steps=Tc, pinv=pinv, seed=st)
            set_params_inplace(net, th_ema)
            gmE, gsE, _ = cocycle_gamma(net, loss_fn, XY, optn, params, batch, lr, steps=Tc, pinv=pinv, seed=st+1)
            raw_g.append(gmR); ema_g.append(gmE)
            if conv_save is None: conv_save = convR.tolist()
        raw_g = np.array(raw_g); ema_g = np.array(ema_g)
        raw_mean, raw_sem = raw_g.mean(0), raw_g.std(0) / np.sqrt(len(raw_g))
        ema_mean, ema_sem = ema_g.mean(0), ema_g.std(0) / np.sqrt(len(ema_g))
        i1 = int(np.argmin(np.abs(C_GRID - 1.0)))
        # object (ii) along-trajectory gamma
        gii = along_traj_gamma(optn, params, batch, lr, warm_steps=steps - 3000, meas_steps=2500)
        rec = dict(tag=tag, optimizer=optn, batch=batch, lr=lr,
                   c_grid=C_GRID.tolist(),
                   gamma_raw_mean=raw_mean.tolist(), gamma_raw_sem=raw_sem.tolist(),
                   gamma_ema_mean=ema_mean.tolist(), gamma_ema_sem=ema_sem.tolist(),
                   gamma_raw_c1=float(raw_mean[i1]), gamma_raw_c1_sem=float(raw_sem[i1]),
                   gamma_ema_c1=float(ema_mean[i1]), gamma_ema_c1_sem=float(ema_sem[i1]),
                   cstar_raw=find_cstar(raw_mean), cstar_ema=find_cstar(ema_mean),
                   gamma_along_traj_c1=float(gii), exp2_div_mult=EXP2_DIV_MULT.get(tag),
                   n_snaps=len(snaps), conv_c1=conv_save)
        results.append(rec)
        print(f"  {tag}: gamma_raw(1)={rec['gamma_raw_c1']:+.4f}+/-{rec['gamma_raw_c1_sem']:.4f}  "
              f"gamma_ema(1)={rec['gamma_ema_c1']:+.4f}+/-{rec['gamma_ema_c1_sem']:.4f}  "
              f"along_traj(1)={gii:+.4f}  c*_raw={rec['cstar_raw']:.3f} c*_ema={rec['cstar_ema']:.3f}  "
              f"[Exp2 div_mult={rec['exp2_div_mult']}]", flush=True)
        with open(os.path.join(OUT_DIR, 'frozen_cocycle_v2.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        if r.get('diverged'): continue
        ax[0].errorbar(r['c_grid'], r['gamma_raw_mean'], yerr=r['gamma_raw_sem'], marker='o', ms=3, capsize=2, label=r['tag'])
    ax[0].axhline(0, color='k', ls='--', lw=0.8); ax[0].axvline(1.0, color='gray', ls=':', lw=0.8)
    ax[0].set_xlabel('lr multiplier c'); ax[0].set_ylabel('frozen gamma (raw checkpoint)')
    ax[0].set_title('gamma(c) raw; c* vs Exp2 divergence mult'); ax[0].legend(fontsize=7)
    for r in results:
        if r.get('diverged'): continue
        ax[1].errorbar(r['c_grid'], r['gamma_ema_mean'], yerr=r['gamma_ema_sem'], marker='s', ms=3, capsize=2, label=r['tag'])
    ax[1].axhline(0, color='k', ls='--', lw=0.8); ax[1].axvline(1.0, color='gray', ls=':', lw=0.8)
    ax[1].set_xlabel('lr multiplier c'); ax[1].set_ylabel('frozen gamma (EMA theta_tilde)')
    ax[1].set_title('gamma(c) at EMA point (phase control)'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'frozen_cocycle_v2.png'), dpi=90); plt.close(fig)

    print("\n===== VERDICT (taxonomy + cross-validation) =====")
    print(f"{'cell':14s} {'g_raw(1)':>16s} {'g_ema(1)':>16s} {'g_alongtraj':>11s} {'c*_raw':>7s} {'c*_ema':>7s} {'Exp2mult':>8s}")
    for r in results:
        if r.get('diverged'):
            print(f"{r['tag']:14s}  diverged"); continue
        gr = f"{r['gamma_raw_c1']:+.3f}+/-{r['gamma_raw_c1_sem']:.3f}"
        ge = f"{r['gamma_ema_c1']:+.3f}+/-{r['gamma_ema_c1_sem']:.3f}"
        print(f"{r['tag']:14s} {gr:>16s} {ge:>16s} {r['gamma_along_traj_c1']:+11.3f} "
              f"{r['cstar_raw']:7.3f} {r['cstar_ema']:7.3f} {str(r['exp2_div_mult']):>8s}")
    print("\n Taxonomy: realized=0 (taut); along-traj (ii) has feedback; frozen (iii) is operating-point.")
    print(" Cross-val: momentum c* ~ its Exp2 div_mult (~1.2)? SGD c* < its div_mult (gap=cubic)?")
    print(" Phase: g_raw(1)>0 but g_ema(1)~0 => positive reading was oscillation phase, not physics.")


if __name__ == '__main__':
    main()
