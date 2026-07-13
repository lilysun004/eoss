"""
Debug pass for the (i,ii)-vs-(iii) disagreement at b128 (registered stop-and-debug branch).

A. est-iii FINE: longer horizon (T=800, n_rep=16), fine c-grid around the disputed crossing,
   BLOCK SEs (4 post-burn blocks, slope per block) so the crossing gets an honest CI.
B. ENRICHED FRAME: rebuild V from top-3 eigvecs per construction batch (deflated power
   iteration) instead of top-1 -- the leading hypothesis for the gap is that (i)/(ii) discard
   each batch's 2nd/3rd stiff directions, which (iii) sees. Then re-run (i)/(ii) under the
   enriched frame: if their crossing drops toward (iii)'s ~1.05, the gap was under-inclusive
   framing (operator route survives, with the richer frame rule); if not, the gap is
   irreducible full-space physics and (iii) is the quoted estimator.
"""
import os, sys, json
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
from experiments.ms_frame_pool import _load_net, MS, estimator_i, estimator_ii
from utils.measure import create_hessian_vector_product, flatt

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
N_CONS3, N_HELD, CAP = 120, 40, 0.9


def batch_top_k_deflated(net, loss_fn, Xb, Yb, k=3, iters=14):
    """Top-k eigvecs of one batch Hessian by power iteration with deflation."""
    params = [p for p in net.parameters() if p.requires_grad]
    lo = loss_fn(net(Xb).squeeze(-1), Yb)
    grads = T.autograd.grad(lo, params, create_graph=True)
    g = flatt(grads)
    hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
    us, lams = [], []
    try:
        for j in range(k):
            u = g.detach().clone() if j == 0 else T.randn_like(g)
            for prev in us:
                u -= (prev @ u) * prev
            u /= u.norm() + 1e-30
            for _ in range(iters):
                Hu = hvp(u, retain_graph_override=True).detach()
                for prev, pl in zip(us, lams):
                    Hu -= pl * (prev @ u) * prev
                n = float(Hu.norm())
                if n < 1e-20:
                    break
                u = Hu / n
            lam = float(u @ hvp(u, retain_graph_override=True).detach())
            us.append(u.clone()); lams.append(lam)
    finally:
        hvp.free_memory()
    return us


def build_frame_top3(tag):
    net, loss_fn, X, Y, m, ck = _load_net(tag)
    g = T.Generator().manual_seed(31337 + m["seed"])
    cols = [top_k_basis(net, loss_fn, X, Y, 8)[0]]
    for _ in range(N_CONS3):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        cols += [u.unsqueeze(1) for u in batch_top_k_deflated(net, loss_fn, X[idx], Y[idx], 3)]
    A = T.cat(cols, dim=1)
    U, S, _ = T.linalg.svd(A, full_matrices=False)
    held = []
    for _ in range(N_HELD):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        held += batch_top_k_deflated(net, loss_fn, X[idx], Y[idx], 3)
    H = T.stack(held, dim=1)
    proj = (U.t() @ H) ** 2
    cum = T.cumsum(proj, dim=0).mean(dim=1).numpy()      # held-out capture of top-3 family
    kstar = int(np.argmax(cum >= CAP) + 1) if (cum >= CAP).any() else -1
    k_use = kstar if kstar > 0 else min(96, U.shape[1])
    np.savez(os.path.join(MS, f"{tag}_framepool3.npz"), V=U[:, :k_use].numpy().astype(np.float32),
             capture_curve=cum, kstar=kstar, k_use=k_use, capture_at_kuse=float(cum[k_use - 1]))
    print(f"[frame3] {tag}: K*={kstar} capture={cum[k_use-1]:.3f} at K={k_use}", flush=True)


def mpool3(tag):
    from experiments.ms_cocycle import POOL_P
    net, loss_fn, X, Y, m, ck = _load_net(tag)
    V = T.tensor(np.load(os.path.join(MS, f"{tag}_framepool3.npz"))["V"], dtype=T.float32)
    g = T.Generator().manual_seed(777 + m["seed"])
    pool = []
    for _ in range(POOL_P):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        pool.append(reduced_M(net, loss_fn, X[idx], Y[idx], V))
    _, eigval = top_k_basis(net, loss_fn, X, Y, 1)
    np.savez(os.path.join(MS, f"{tag}_pool_pv3.npz"), pool=T.stack(pool).numpy(),
             eigval=eigval.reshape(-1).numpy(), buf_red=(V.t() @ ck["buf"]).numpy(),
             lam_top=float(eigval.reshape(-1)[0]), lr=m["lr"], beta=m["beta"],
             batch=m["batch"], optn=np.array(m["optn"]))
    print(f"[mpool3] {tag}: K={V.shape[1]}", flush=True)


def est_iii_fine(tag, c_list=(0.9, 1.0, 1.05, 1.1, 1.2, 1.35), n_rep=16, steps=800, burn=100,
                 frame_suffix=""):
    net, loss_fn, X, Y, m, ck = _load_net(tag)
    V = T.tensor(np.load(os.path.join(MS, f"{tag}_framepool{frame_suffix}.npz"))["V"],
                 dtype=T.float32)
    beta = m["beta"]; lr = m["lr"]; optn = m["optn"]
    params = [p for p in net.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    rows = []
    for c in c_list:
        g = T.Generator().manual_seed(6001 + m["seed"])
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
            dth = dth - c * lr * ((Hd + beta * dv) if optn == "SGD-Nesterov" else dv)
            ro2 = float(((V.t() @ dth) ** 2).sum() + ((V.t() @ dv) ** 2).sum())
            series.append(np.log(max(ro2, 1e-300)) + 2 * logscale)
            full2 = float((dth ** 2).sum() + (dv ** 2).sum())
            s = np.sqrt(full2 / (2 * n_rep)) + 1e-300
            dth /= s; dv /= s; logscale += np.log(s)
        y = np.array(series[burn:])
        bl = np.array_split(np.arange(len(y)), 4)
        slopes = [np.polyfit(b, y[b], 1)[0] for b in bl]
        gm, se = 0.5 * float(np.mean(slopes)), 0.5 * float(np.std(slopes) / np.sqrt(3))
        rows.append(dict(c=c, gamma2=gm, se=se))
        print(f"[iii-fine{frame_suffix}] {tag} c={c}: gamma2={gm:+.4f} +/- {se:.4f}", flush=True)
    json.dump(rows, open(os.path.join(MS, f"{tag}_iii_fine{frame_suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["fine", "frame3", "mpool3", "i3", "ii3", "fine3"])
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()
    for tag in a.cells:
        if a.stage == "fine":
            est_iii_fine(tag)
        elif a.stage == "frame3":
            build_frame_top3(tag)
        elif a.stage == "mpool3":
            mpool3(tag)
        elif a.stage == "i3":
            import experiments.ms_frame_pool as FP
            z = np.load(os.path.join(MS, f"{tag}_pool_pv3.npz"))
            pool = T.tensor(z["pool"])
            rhos = [FP.rho_operator(pool, str(z["optn"]), float(z["beta"]), c * float(z["lr"]))
                    for c in FP.C_GRID]
            c2 = FP.find_cstar(FP.C_GRID, np.log(rhos))
            print(f"[est-i3] {tag}: c*_2={c2:.3f} kappa_ms={2/c2:.3f}  "
                  f"rho: {' '.join(f'{r:.4f}' for r in rhos)}", flush=True)
        elif a.stage == "ii3":
            from experiments.ms_cocycle import _propagate, find_cstar, C_GRID, N_REP, T_STEPS, BURN
            z = np.load(os.path.join(MS, f"{tag}_pool_pv3.npz"))
            pool = T.tensor(z["pool"])
            g2 = [_propagate(pool, str(z["optn"]), float(z["beta"]), c * float(z["lr"]),
                             N_REP, T_STEPS, BURN, True, 991)[0] for c in C_GRID]
            c2 = find_cstar(C_GRID, g2)
            print(f"[est-ii3] {tag}: c*_2={c2:.3f} kappa_ms={2/c2:.3f}", flush=True)
        elif a.stage == "fine3":
            est_iii_fine(tag, frame_suffix="3")
