"""
Pooled-frame construction + three-estimator kappa_ms (ADDENDUM 3 protocol, registered).

Frame rule (one rule, all cells): V = SVD-orthonormalized union of per-batch top eigvecs u_B
over ~160 fresh CONSTRUCTION batches at the frozen checkpoint, plus the frozen full-Hessian
top-8 anchor columns. K* = smallest leading dimension whose HELD-OUT (40 disjoint batches)
mean capture ||V V^T u_B||^2 >= 0.9. K* reported per cell.

Estimators (trust order per registration):
 (i)   explicit pooled operator: power iteration of Sigma <- mean_pool[J(c) Sigma J(c)^T] on the
       2K x 2K second-moment matrix; rho(c) = growth factor; c*_2 where rho crosses 1.
 (ii)  projected replicas: ms_cocycle._propagate on the pooled-V M-pool (shared renorm).
 (iii) FULL-SPACE replicas, subspace readout: tangent (dtheta, dvel) in R^{2d} propagated through
       the optimizer's own linearized update with fresh-batch HVPs at the frozen point (no
       projection of dynamics); readout = ||V^T dtheta||^2 + ||V^T dvel||^2 across replicas
       (shared renorm by FULL norm for conditioning; growth read on the readout in float64).

Reconciliation (registered): run (i)/(ii)/(iii) at b128_s0 + b8_beta0.9_s0; pick per protocol;
recompute ALL cells (incl. at-edge calibrations) under the winner before verdicts are read.
"""
import os, sys, json, argparse, time
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault("DATASETS", "/Users/xq/Desktop/moonshot/eoss/datasets")
os.environ.setdefault("EOSS_SKIP_CHECKSUM", "1")
if os.environ.get("EOSS_SKIP_CHECKSUM"):
    import torchvision.datasets.cifar as _cifar
    _cifar.check_integrity = lambda *a, **k: True

import experiments.long_train_grid as L
from experiments.frozen_cocycle_v3 import top_k_basis, reduced_M
from experiments.ms_frame_audit import batch_top_u
from experiments.ms_cocycle import _propagate, find_cstar, C_GRID, POOL_P, N_REP, T_STEPS, BURN
from utils.measure import create_hessian_vector_product, flatt
from utils.curvature_segment import set_params_inplace

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "kspec"); MS = os.path.join(OUT, "ms")
N_CONS, N_HELD, CAP, K_MAX = 160, 40, 0.9, 120


def _load_net(tag):
    ck = T.load(os.path.join(MS, f"{tag}_ckpt.pt"), weights_only=False)
    m = ck["meta"]
    T.manual_seed(m["seed"])
    X, Y = L.get_data(); net, loss_fn = L.build()
    set_params_inplace(net, ck["theta"])
    return net, loss_fn, X, Y, m, ck


def build_frame(tag):
    """Pooled V + K* by held-out capture; saves {tag}_framepool.npz."""
    t0 = time.time()
    net, loss_fn, X, Y, m, ck = _load_net(tag)
    g = T.Generator().manual_seed(31337 + m["seed"])
    full_batch = m["batch"] >= len(X)
    n_cons = 1 if full_batch else N_CONS
    cols = []
    V8, _ = top_k_basis(net, loss_fn, X, Y, 8)              # anchor columns (frozen full-H top-8)
    cols.append(V8)
    for _ in range(n_cons):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        cols.append(batch_top_u(net, loss_fn, X[idx], Y[idx]).unsqueeze(1))
    A = T.cat(cols, dim=1)                                   # [d, 8+n_cons]
    U, S, _ = T.linalg.svd(A, full_matrices=False)
    # held-out capture as function of leading dimension
    held = []
    for _ in range(1 if full_batch else N_HELD):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        held.append(batch_top_u(net, loss_fn, X[idx], Y[idx]))
    H = T.stack(held, dim=1)                                 # [d, n_held]
    proj = (U.t() @ H) ** 2                                  # [ncols, n_held]
    cum = T.cumsum(proj, dim=0).mean(dim=1).numpy()          # mean held-out capture vs K
    kstar = int(np.argmax(cum >= CAP) + 1) if (cum >= CAP).any() else -1
    k_use = kstar if kstar > 0 else min(K_MAX, U.shape[1])
    V = U[:, :k_use].contiguous()
    np.savez(os.path.join(MS, f"{tag}_framepool.npz"), V=V.numpy().astype(np.float32),
             capture_curve=cum, kstar=kstar, k_use=k_use,
             capture_at_kuse=float(cum[k_use - 1]), converged=bool(kstar > 0))
    print(f"[frame] {tag}: K*={kstar} (capture {cum[k_use-1]:.3f} at K={k_use}; "
          f"curve[8]={cum[min(7,len(cum)-1)]:.3f}) converged={kstar>0} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return kstar


def build_mpool_V(tag):
    """M-pool under pooled V; saves {tag}_pool_pv.npz (same format as ms_cocycle pools)."""
    net, loss_fn, X, Y, m, ck = _load_net(tag)
    fz = np.load(os.path.join(MS, f"{tag}_framepool.npz"))
    V = T.tensor(fz["V"], dtype=T.float32)
    P = 1 if m["batch"] >= len(X) else POOL_P
    g = T.Generator().manual_seed(777 + m["seed"])
    pool = []
    for _ in range(P):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        pool.append(reduced_M(net, loss_fn, X[idx], Y[idx], V))
    pool = T.stack(pool)
    _, eigval = top_k_basis(net, loss_fn, X, Y, 1)
    eigval = eigval.reshape(-1)
    np.savez(os.path.join(MS, f"{tag}_pool_pv.npz"), pool=pool.numpy(),
             eigval=eigval.numpy(), buf_red=(V.t() @ ck["buf"]).numpy(),
             lam_top=float(eigval[0]), lr=m["lr"], beta=m["beta"], batch=m["batch"],
             optn=np.array(m["optn"]))
    print(f"[mpool-pv] {tag}: P={P} K={V.shape[1]}", flush=True)


def rho_operator(pool, optn, beta, ceta, iters=60, seed=0):
    """(i) explicit pooled MS operator: Sigma <- mean_p J_p Sigma J_p^T, rho = growth factor."""
    P, K, _ = pool.shape
    I = T.eye(K)
    Js = []
    for p in range(P):
        M = pool[p]
        if optn == "SGD-Momentum":
            # z=(a,b): b' = beta b + M a ; a' = a - ceta b'
            top = T.cat([I - ceta * M, -ceta * beta * I], dim=1)
            bot = T.cat([M, beta * I], dim=1)
        elif optn == "SGD-Nesterov":
            # b' = beta b + M a ; a' = a - ceta (M a + beta b')
            top = T.cat([I - ceta * (M + beta * M), -ceta * beta * beta * I], dim=1)
            bot = T.cat([M, beta * I], dim=1)
        else:
            top = T.cat([I - ceta * M, 0 * I], dim=1)
            bot = T.cat([0 * I, 0 * I], dim=1)
        Js.append(T.cat([top, bot], dim=0))
    Js = T.stack(Js)                                         # [P, 2K, 2K]
    g = T.Generator().manual_seed(seed)
    S = T.randn(2 * K, 2 * K, generator=g); S = S @ S.t(); S /= T.linalg.norm(S)
    rho = 1.0
    for it in range(iters):
        S2 = T.einsum("pij,jk,plk->il", Js, S, Js) / P
        r = float(T.linalg.norm(S2))
        S = S2 / r
        if it >= iters - 10:
            rho = r if it == iters - 10 else 0.7 * rho + 0.3 * r
    return rho


def estimator_i(tag):
    z = np.load(os.path.join(MS, f"{tag}_pool_pv.npz"))
    pool = T.tensor(z["pool"]); optn = str(z["optn"]); beta = float(z["beta"]); lr = float(z["lr"])
    rhos = [rho_operator(pool, optn, beta, c * lr) for c in C_GRID]
    lg = np.log(rhos)
    c2 = find_cstar(C_GRID, lg)
    print(f"[est-i] {tag}: rho(c): " + " ".join(f"{r:.4f}" for r in rhos) +
          f"  c*_2={c2:.3f} kappa_ms={2/c2 if np.isfinite(c2) and c2>0 else float('nan'):.3f}", flush=True)
    rec = dict(tag=tag, est="i", rhos=list(map(float, rhos)), cstar2=c2,
               kappa_ms=2 / c2 if np.isfinite(c2) and c2 > 0 else float("nan"))
    _save_recon(rec); return rec


def estimator_ii(tag):
    z = np.load(os.path.join(MS, f"{tag}_pool_pv.npz"))
    pool = T.tensor(z["pool"]); optn = str(z["optn"]); beta = float(z["beta"]); lr = float(z["lr"])
    g2 = []
    for c in C_GRID:
        m2, _ = _propagate(pool, optn, beta, c * lr, N_REP, T_STEPS, BURN, True, 991)
        g2.append(m2)
    c2 = find_cstar(C_GRID, g2)
    print(f"[est-ii] {tag}: c*_2={c2:.3f} kappa_ms={2/c2 if np.isfinite(c2) and c2>0 else float('nan'):.3f}", flush=True)
    rec = dict(tag=tag, est="ii", gamma2=list(map(float, g2)), cstar2=c2,
               kappa_ms=2 / c2 if np.isfinite(c2) and c2 > 0 else float("nan"))
    _save_recon(rec); return rec


def estimator_iii(tag, c_list=(0.85, 1.0, 1.15, 1.3, 1.5, 2.0), n_rep=8, steps=400, burn=60):
    """(iii) full-space replicas, pooled-V readout. The dynamics are NOT projected; V only
    defines the readout. gamma_2 = 0.5 * slope of log(sum_r ||V^T z_r||^2, true scale) vs t.
    Shared full-norm renormalization is pure conditioning (linear dynamics), tracked in
    logscale so the readout is read in true units. 1 HVP graph per step serves all replicas."""
    net, loss_fn, X, Y, m, ck = _load_net(tag)
    fz = np.load(os.path.join(MS, f"{tag}_framepool.npz"))
    V = T.tensor(fz["V"], dtype=T.float32)
    beta = m["beta"]; lr = m["lr"]; optn = m["optn"]
    params = [p for p in net.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    res = {}
    for c in c_list:
        g = T.Generator().manual_seed(5150 + m["seed"])
        dth = T.randn(d, n_rep, generator=g); dth /= dth.norm(dim=0, keepdim=True)
        dv = T.zeros(d, n_rep)
        logscale = 0.0; series = []
        for t in range(steps):
            idx = T.randperm(len(X), generator=g)[:m["batch"]]
            lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
            grads = T.autograd.grad(lo, params, create_graph=True)
            hvp = create_hessian_vector_product(lo, net, params=params, grads=grads,
                                                flat_grads=flatt(grads))
            try:
                Hd = T.stack([hvp(dth[:, r], retain_graph_override=True)
                              for r in range(n_rep)], dim=1)
            finally:
                hvp.free_memory()
            dv = beta * dv + Hd
            if optn == "SGD-Nesterov":
                dth = dth - c * lr * (Hd + beta * dv)
            else:
                dth = dth - c * lr * dv
            ro2 = float(((V.t() @ dth) ** 2).sum() + ((V.t() @ dv) ** 2).sum())
            series.append(np.log(max(ro2, 1e-300)) + 2 * logscale)   # log readout^2, true scale
            full2 = float((dth ** 2).sum() + (dv ** 2).sum())
            s = np.sqrt(full2 / (2 * n_rep)) + 1e-300
            dth /= s; dv /= s; logscale += np.log(s)
        y = np.array(series[burn:]); x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0])
        res[c] = 0.5 * slope                                        # amplitude-scale rate
        print(f"[est-iii] {tag} c={c}: gamma2_readout={res[c]:+.4f}", flush=True)
    cs = sorted(res); gm = [res[c] for c in cs]
    c2 = find_cstar(np.array(cs), gm)
    print(f"[est-iii] {tag}: c*_2={c2:.3f} kappa_ms={2/c2 if np.isfinite(c2) and c2>0 else float('nan'):.3f}", flush=True)
    rec = dict(tag=tag, est="iii", c_list=list(cs), gamma2=gm, cstar2=c2,
               kappa_ms=2 / c2 if np.isfinite(c2) and c2 > 0 else float("nan"))
    _save_recon(rec)
    return rec


def _save_recon(rec):
    p = os.path.join(MS, "recon.json")
    all_ = json.load(open(p)) if os.path.exists(p) else []
    all_ = [r for r in all_ if not (r["tag"] == rec["tag"] and r["est"] == rec["est"])] + [rec]
    json.dump(all_, open(p, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["frame", "mpool", "i", "ii", "iii", "recon"], required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()
    for tag in a.cells:
        if a.stage == "frame":
            build_frame(tag)
        elif a.stage == "mpool":
            build_mpool_V(tag)
        elif a.stage == "i":
            estimator_i(tag)
        elif a.stage == "ii":
            estimator_ii(tag)
        elif a.stage == "iii":
            estimator_iii(tag)
        elif a.stage == "recon":
            build_frame(tag); build_mpool_V(tag)
            estimator_i(tag); estimator_ii(tag); estimator_iii(tag)
