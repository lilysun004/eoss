"""
kappa_ms — mean-square marginality of the measured closed loop (pre-registered in
KSPEC_PREREG_ANNOTATIONS.md ADDENDUM before any gamma_2 was computed).

Pipeline per ladder cell:
 1. --replay   deterministic measurement-free replay of the cell to its LIVE-phase checkpoint
               step (same seeds / batch generator as slow_sweep.run_cell; verified against the
               logged loss trace), saving (theta*, buffer*).
 2. --pool     at the frozen checkpoint: V = top-K eigenbasis (K=8, v3 null-space fix), pool of
               P=384 i.i.d. fresh-batch reduced Hessians M_p = V^T H_B V (batched HVPs).
               Full-batch cells (batch >= n) have deterministic M -> pool collapses to 1.
 3. --gamma    gamma_2(c): N=128 tangent replicas in reduced coords, each step drawing an
               INDEPENDENT M from the pool per replica, propagated through the optimizer's OWN
               linearized recursion (the recursion is the optimizer's definition, not a threshold
               formula; certify: no (1+beta)/(1-beta)/(1+2*beta) THRESHOLD expression here).
               SHARED renormalization by s_t = RMS over replicas -> gamma_2 = mean log s_t.
               gamma_1 (per-replica normalization) from the same pool for the moment contrast.
               c*_2 = zero crossing on the c-grid; kappa_ms = 2/c*_2.

Registered spec: K=8, P=384, N=128, T=3000, burn=300, c-grid below. Paper-law (Eq 21) evaluation
from the same pool lives in ms_paperlaw.py (that side may use beta formulas; this side must not).
"""
import os, sys, json, glob, argparse, time
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
from utils.optimizer import create_optimizer
from utils.measure import flatt
from utils.curvature_segment import set_params_inplace

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "kspec")
MS = os.path.join(OUT, "ms"); os.makedirs(MS, exist_ok=True)

C_GRID = np.array([0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 2.0])
K_RED, POOL_P, N_REP, T_STEPS, BURN = 8, 384, 128, 3000, 300
# LIVE-phase checkpoint steps (ring-down excluded at b512/b2048: live block = steps 4000-6048)
CKPT_STEP = {"L_b8_beta0.9": 20000, "L_b32_beta0.9": 20000, "L_b128_beta0.9": 12000,
             "L_b512_beta0.9": 5000, "L_b2048_beta0.9": 5000, "L_b8_beta0.99": 20000,
             "L_nest_b8_beta0.9": 20000, "L_nest_b128_beta0.9": 12000, "L_nest_b2048_beta0.9": 5000,
             "L_adam_b8": 20000, "L_adam_b128": 12000, "L_adam_b2048": 6000,
             "L_adam05_b2048": 6000, "L_b64_beta0.9": 16000,
             "L_nest_b256_beta0.9": 6000, "L_nest_b512_beta0.9": 5000}


def cellmeta(tag):
    return json.load(open(os.path.join(OUT, tag, "meta.json")))


def replay(tag, ck_step):
    """Measurement-free deterministic replay to ck_step; verify vs logged loss trace."""
    m = cellmeta(tag)
    T.manual_seed(m["seed"])
    X, Y = L.get_data(); net, loss_fn = L.build()
    params_dict = ({} if m["optn"] == "SGD" else
                   ({"beta1": m["beta"], "beta2": 0.99} if m["optn"] == "Adam"
                    else {"beta": m["beta"]}))
    opt = create_optimizer(m["optn"], net, m["lr"], params_dict)
    params = [p for p in net.parameters() if p.requires_grad]
    gen = T.Generator().manual_seed(1000 + m["seed"])
    logged = np.load(os.path.join(OUT, tag, "dense.npz"))["loss"]
    checks = {}
    for step in range(ck_step):
        idx = T.randperm(len(X), generator=gen)[:m["batch"]]
        lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        if step in (100, 1000, 5000, 10000, 19999):
            checks[step] = (float(lo), float(logged[step]) if step < len(logged) else float("nan"))
        grads = T.autograd.grad(lo, params, create_graph=True)   # identical op order to runner
        opt.zero_grad()
        for p, gr in zip(params, grads):
            p.grad = gr.detach()
        opt.step()
    theta = flatt([p.detach() for p in params]).clone()
    # buffer (momentum state) as flat vector
    bufs = []
    for p in params:
        st = opt.inner.state.get(p)
        bufs.append(st["momentum_buffer"].flatten() if st and "momentum_buffer" in st
                    else T.zeros(p.numel()))
    buf = T.cat(bufs).clone()
    rel = max(abs(a - b) / (abs(b) + 1e-12) for a, b in checks.values() if np.isfinite(b))
    ok = rel < 1e-4
    T.save(dict(theta=theta, buf=buf, checks=checks, replay_ok=bool(ok), ck_step=ck_step,
                opt_state=opt.inner.state_dict(),      # full state: Adam v-hat etc. for brackets
                meta=m), os.path.join(MS, f"{tag}_ckpt.pt"))
    print(f"[replay] {tag} @ {ck_step}: max rel loss err {rel:.2e} -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    return ok


def build_pool(tag, K=K_RED, suffix=""):
    ck = T.load(os.path.join(MS, f"{tag}_ckpt.pt"), weights_only=False)
    m = ck["meta"]
    T.manual_seed(m["seed"])
    X, Y = L.get_data(); net, loss_fn = L.build()
    set_params_inplace(net, ck["theta"])
    V, eigval = top_k_basis(net, loss_fn, X, Y, K)
    P = 1 if m["batch"] >= len(X) else POOL_P
    g = T.Generator().manual_seed(777 + m["seed"])
    pool = []
    for _ in range(P):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        pool.append(reduced_M(net, loss_fn, X[idx], Y[idx], V))
    pool = T.stack(pool)                                   # [P,K,K]
    # buffer* in reduced coords
    buf_red = (V.t() @ ck["buf"]).detach()
    np.savez(os.path.join(MS, f"{tag}_pool{suffix}.npz"), pool=pool.numpy(),
             eigval=eigval.numpy(), buf_red=buf_red.numpy(),
             lam_top=float(eigval[0]), lr=m["lr"], beta=m["beta"], batch=m["batch"],
             optn=np.array(m["optn"]))
    h = pool[:, 0, 0].numpy()
    print(f"[pool] {tag}: P={P} lam_top={float(eigval[0]):.0f} "
          f"h(u0): mean={h.mean():.1f} cv2={(h.var()/h.mean()**2 if P > 1 else 0):.3f}", flush=True)


def _propagate(pool, optn, beta, ceta, n_rep, steps, burn, shared, seed, buf0=None):
    """Reduced tangent cocycle; returns growth rate (log/step) and block SE.
    shared=True -> gamma_2 (RMS-across-replicas normalization); False -> gamma_1."""
    P, K, _ = pool.shape
    g = T.Generator().manual_seed(seed)
    a = T.randn(K, n_rep, generator=g); a /= a.norm(dim=0, keepdim=True)
    buf = T.zeros(K, n_rep)
    logs = []
    for t in range(steps):
        ii = T.randint(P, (n_rep,), generator=g)
        Msel = pool[ii]                                     # [N,K,K]
        Ma = T.bmm(Msel, a.t().unsqueeze(2)).squeeze(2).t() # [K,N]
        if optn == "SGD-Momentum":
            buf = beta * buf + Ma
            a = a - ceta * buf
        elif optn == "SGD-Nesterov":
            buf = beta * buf + Ma
            a = a - ceta * (Ma + beta * buf)
        else:
            a = a - ceta * Ma
        sq = (a ** 2).sum(0) + (buf ** 2).sum(0)
        if shared:
            s = float(T.sqrt(sq.mean()))
            a /= s; buf /= s
            if t >= burn:
                logs.append(np.log(s))
        else:
            s = T.sqrt(sq) + 1e-300
            a /= s; buf /= s
            if t >= burn:
                logs.append(float(T.log(s).mean()))
    arr = np.array(logs); bl = np.array_split(arr, 4)
    bm = np.array([b.mean() for b in bl])
    return float(bm.mean()), float(bm.std() / 2)


def find_cstar(cg, gm):
    for i in range(len(cg) - 1):
        if gm[i] <= 0 <= gm[i + 1]:
            f = -gm[i] / (gm[i + 1] - gm[i] + 1e-30)
            return float(cg[i] + f * (cg[i + 1] - cg[i]))
    return float("nan")


def gamma_cell(tag, k_use=None, pool_sub=None, seed=0, suffix=""):
    z = np.load(os.path.join(MS, f"{tag}_pool{suffix}.npz"))
    pool = T.tensor(z["pool"]); optn = str(z["optn"]); beta = float(z["beta"]); lr = float(z["lr"])
    if k_use:
        pool = pool[:, :k_use, :k_use]
    if pool_sub and pool.shape[0] > pool_sub:
        pool = pool[:pool_sub]
    g2, g2se, g1 = [], [], []
    for c in C_GRID:
        m2, se2 = _propagate(pool, optn, beta, c * lr, N_REP, T_STEPS, BURN, True, seed + 11)
        m1, _ = _propagate(pool, optn, beta, c * lr, N_REP, T_STEPS, BURN, False, seed + 12)
        g2.append(m2); g2se.append(se2); g1.append(m1)
    c2, c1 = find_cstar(C_GRID, g2), find_cstar(C_GRID, g1)
    out = dict(tag=tag, optn=optn, beta=beta, lr=lr, batch=int(z["batch"]), K=int(pool.shape[1]),
               P=int(pool.shape[0]), c_grid=C_GRID.tolist(), gamma2=g2, gamma2_se=g2se, gamma1=g1,
               cstar2=c2, cstar1=c1,
               kappa_ms=2 / c2 if np.isfinite(c2) and c2 > 0 else float("nan"),
               lam_top=float(z["lam_top"]), kappa_raw_ck=lr * float(z["lam_top"]))
    json.dump(out, open(os.path.join(MS, f"{tag}_gamma{suffix}.json"), "w"), indent=1)
    print(f"[gamma] {tag}: c*_2={c2:.3f} kappa_ms={out['kappa_ms']:.3f}  c*_1={c1:.3f}  "
          f"gamma2(1.0)={g2[list(C_GRID).index(1.0)]:+.4f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "pool", "gamma", "all"], default="all")
    ap.add_argument("--cells", nargs="*", default=None)
    a = ap.parse_args()
    tags = a.cells or [d for d in sorted(os.listdir(OUT))
                       if d.startswith("L_") and os.path.isdir(os.path.join(OUT, d))
                       and "_".join(d.split("_")[:-1]) in CKPT_STEP]
    for tag in tags:
        base = "_".join(tag.split("_")[:-1])
        if a.stage in ("replay", "all") and not os.path.exists(os.path.join(MS, f"{tag}_ckpt.pt")):
            replay(tag, CKPT_STEP[base])
        if a.stage in ("pool", "all") and not os.path.exists(os.path.join(MS, f"{tag}_pool.npz")):
            build_pool(tag)
        if a.stage in ("gamma", "all"):
            gamma_cell(tag)
