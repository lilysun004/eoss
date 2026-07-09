"""
The mechanism / control-parameter experiment (why momentum specifically can't reach its edge).

Hypothesis (reviewer #4): marginality requires the optimizer's STEP to track the unstable
direction u_B faster than u_B rotates. At small batch u_B (top eigvec of the per-batch
Hessian H_B) rotates rapidly batch-to-batch. SGD's step is the FRESH gradient (always
tracks the current geometry). Momentum's step is the buffer m = ~1/(1-beta)-step average of
past gradients; if u_B rotates faster than the buffer memory 1/(1-beta), the buffer averages
over DIFFERENT unstable directions and cannot concentrate energy in the current u_B ->
sub-edge. The dimensionless control parameter is
    R = (buffer memory  1/(1-beta))  /  (u_B rotation timescale  tau_rot).
R >> 1  => can't track => sub-edge (metastable);  R << 1 => tracks => reaches edge.

We train each cell to its plateau, then over a measurement window record per step:
  - |cos(u_B(t), u_B(t-1))|      -> u_B rotation; tau_rot = 1 / (1 - mean|cos|)
  - |cos(m_t, u_B(t))|           -> momentum buffer alignment with the unstable dir
  - |cos(g_t, u_B(t))|           -> gradient alignment (the SGD-step analog)
  - |cos(s_t, u_B(t))|           -> actual step alignment
and report tau_rot, R = (1/(1-beta))/tau_rot, and the alignments, across batch {8,128,2048}
for SGD-Momentum, plus SGD b8 as the "step=fresh gradient, always tracks" control.

Prediction: small batch -> tau_rot ~ 1 (fast rotation) -> R ~ 10 (beta=0.9) -> low buffer
alignment -> sub-edge; large batch -> tau_rot large -> R < 1 -> buffer aligns -> at edge.
The alignment should track edge-reachability (GBS/kappa) across the batch sweep.
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

import experiments.long_train_grid as L
from utils.optimizer import create_optimizer
from utils.measure import (compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
                           flatt, param_vector)

T.set_num_threads(4)
OUT_DIR = os.path.join(_REPO, 'results', 'mechanism_buffer_rotation')
os.makedirs(OUT_DIR, exist_ok=True)


def buffer_flat(opt, params):
    pieces = []
    for p in params:
        st = opt.inner.state.get(p)
        if st and 'momentum_buffer' in st and st['momentum_buffer'] is not None:
            pieces.append(st['momentum_buffer'].flatten())
        else:
            pieces.append(T.zeros(p.numel()))
    return T.cat(pieces).detach()


def batch_top_eigvec(net, loss_fn, Xb, Yb, cache):
    pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
    lam, u = compute_eigenvalues(lo, net, k=1, max_iterations=60, reltol=0.005,
                                 eigenvector_cache=cache, return_eigenvectors=True,
                                 use_power_iteration=False)
    return float(lam), u.detach()


def cosabs(a, b):
    d = (a * b).sum()
    return float(abs(d) / (a.norm() * b.norm() + 1e-30))


def run_cell(tag, optn, params, batch, lr, train_steps, meas_steps=400):
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    params_l = [p for p in net.parameters() if p.requires_grad]
    beta = float(params.get('beta', 0.0))
    is_mom = (optn == 'SGD-Momentum')
    # train to plateau
    for step in range(train_steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return dict(tag=tag, diverged=True)
        opt.zero_grad(); lo.backward(); opt.step()
    # measurement window
    cache = EigenvectorCache(1)
    u_prev = None
    rot, cos_m_u, cos_g_u, cos_s_u, gbs_list, lam_list = [], [], [], [], [], []
    full_batch = batch >= len(X)
    for t in range(meas_steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        # gradient + step + buffer BEFORE the update
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        grads = T.autograd.grad(lo, params_l, create_graph=True)
        g = flatt(grads); gd = g.detach()
        s = opt.compute_step_direction(g, params_l).detach()
        m = buffer_flat(opt, params_l) if is_mom else gd  # SGD "buffer" = fresh gradient
        # u_B of the current batch (independent converged top eigvec)
        lam, u = batch_top_eigvec(net, loss_fn, Xb, Yb, cache)
        lam_list.append(lam)
        if u_prev is not None:
            rot.append(cosabs(u, u_prev))
        u_prev = u.clone()
        cos_m_u.append(cosabs(m, u)); cos_g_u.append(cosabs(gd, u)); cos_s_u.append(cosabs(s, u))
        # GBS along the way (cheap, reuse graph)
        hvp = create_hessian_vector_product(lo, net, params=params_l, grads=grads, flat_grads=g)
        try:
            Hs = hvp(s, retain_graph_override=False)
            A = -T.dot(gd, s).item(); B = T.dot(s, Hs).item()
        finally:
            hvp.free_memory()
        if abs(A) > 1e-15:
            gbs_list.append(B / A)
        # advance one real step so the buffer/geometry keep evolving
        opt.zero_grad(); lo2 = loss_fn(net(Xb).squeeze(-1), Yb); lo2.backward(); opt.step()
    mean_rot = float(np.mean(rot)) if rot else float('nan')
    tau_rot = 1.0 / max(1e-6, 1.0 - mean_rot)
    buf_mem = 1.0 / (1.0 - beta) if is_mom else 1.0
    R = buf_mem / tau_rot
    lam_mean = float(np.mean(lam_list))
    eta_lam = lr * lam_mean
    edge = 2.0 if not is_mom else 2 * (1 + beta)
    return dict(tag=tag, optimizer=optn, batch=batch, lr=lr, beta=beta, diverged=False,
                mean_cos_uB=mean_rot, tau_rot=tau_rot, buf_mem=buf_mem, R=R,
                cos_buf_uB=float(np.mean(cos_m_u)), cos_grad_uB=float(np.mean(cos_g_u)),
                cos_step_uB=float(np.mean(cos_s_u)),
                gbs=float(np.nanmean(gbs_list)), lam_mean=lam_mean, eta_lam=eta_lam,
                edge=edge, eta_lam_over_edge=eta_lam / edge)


def main():
    cells = [
        ("SGDM_b8",    "SGD-Momentum", {"beta": 0.9}, 8,    0.002, 16000),
        ("SGDM_b128",  "SGD-Momentum", {"beta": 0.9}, 128,  0.003, 9000),
        ("SGDM_b2048", "SGD-Momentum", {"beta": 0.9}, 2048, 0.006, 5000),
        ("SGD_b8",     "SGD",          {},            8,    0.01,  16000),  # control: step=fresh grad
        ("SGD_b2048",  "SGD",          {},            2048, 0.02,  5000),
    ]
    results = []
    for tag, optn, params, batch, lr, ts in cells:
        print(f"\n=== {tag}: {optn} b={batch} lr={lr} ===", flush=True)
        r = run_cell(tag, optn, params, batch, lr, ts)
        results.append(r)
        if r.get('diverged'):
            print(f"  {tag}: DIVERGED"); continue
        print(f"  tau_rot={r['tau_rot']:.2f}  buf_mem={r['buf_mem']:.1f}  R={r['R']:.2f}  "
              f"cos(buf,uB)={r['cos_buf_uB']:.3f} cos(grad,uB)={r['cos_grad_uB']:.3f} "
              f"cos(step,uB)={r['cos_step_uB']:.3f}  GBS={r['gbs']:.3f}  eta*lam/edge={r['eta_lam_over_edge']:.2%}",
              flush=True)
        with open(os.path.join(OUT_DIR, 'mechanism_buffer_rotation.json'), 'w') as f:
            json.dump(results, f, indent=2)
    print("\n===== VERDICT: does R (memory/rotation) predict edge-reachability? =====")
    print(f"{'cell':12s} {'tau_rot':>7s} {'R=mem/rot':>9s} {'cos(step,uB)':>12s} {'GBS':>6s} {'eta*lam/edge':>12s}")
    for r in results:
        if r.get('diverged'): print(f"{r['tag']:12s}  diverged"); continue
        print(f"{r['tag']:12s} {r['tau_rot']:7.2f} {r['R']:9.2f} {r['cos_step_uB']:12.3f} "
              f"{r['gbs']:6.2f} {r['eta_lam_over_edge']:12.1%}")
    print("\n Prediction: R>>1 (buffer can't track fast-rotating uB) => low cos(step,uB) => sub-edge (low GBS/kappa);")
    print(" R<1 or SGD (step=fresh grad, cos high) => reaches edge (GBS~2). If R tracks edge-reachability, it's the control parameter.")


if __name__ == '__main__':
    main()
