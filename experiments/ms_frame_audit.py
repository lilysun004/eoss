"""
Frame-capture audit (registered GATE in KSPEC_PREREG_ANNOTATIONS.md ADDENDUM 2, before the
noise-dominated gamma_2 verdicts are read): the pool is M = V^T H_B V with V frozen top-K at the
checkpoint; if per-batch top directions u_B live partly OUTSIDE span(V) (the R-mechanism), the
cocycle under-counts energy injection and c*_2 biases UP at exactly the decisive cells.

Per cell: recompute V (top-K, same routine as the pool build), then for S sample batches compute
the batch-Hessian top eigvec u_B (gradient-seeded warm power iteration, as the runner does) and
report captured mass ||V^T u_B||^2. RULE (registered): median mass < 0.8 -> K=8 verdict not read;
re-run gamma at K=32.
"""
import os, sys, json, argparse
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
from experiments.frozen_cocycle_v3 import top_k_basis
from utils.measure import create_hessian_vector_product, flatt
from utils.curvature_segment import set_params_inplace

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "kspec"); MS = os.path.join(OUT, "ms")


def batch_top_u(net, loss_fn, Xb, Yb, iters=14):
    params = [p for p in net.parameters() if p.requires_grad]
    lo = loss_fn(net(Xb).squeeze(-1), Yb)
    grads = T.autograd.grad(lo, params, create_graph=True)
    g = flatt(grads)
    hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
    try:
        u = g.detach().clone(); u /= u.norm() + 1e-30
        for _ in range(iters):
            Hu = hvp(u, retain_graph_override=True)
            n = float(Hu.norm())
            if n < 1e-20:
                break
            u = (Hu / n).detach()
    finally:
        hvp.free_memory()
    return u


def audit(tag, K=8, S=48):
    ck = T.load(os.path.join(MS, f"{tag}_ckpt.pt"), weights_only=False)
    m = ck["meta"]
    T.manual_seed(m["seed"])
    X, Y = L.get_data(); net, loss_fn = L.build()
    set_params_inplace(net, ck["theta"])
    V, _ = top_k_basis(net, loss_fn, X, Y, K)
    g = T.Generator().manual_seed(4242 + m["seed"])
    S_eff = 1 if m["batch"] >= len(X) else S
    mass = []
    for _ in range(S_eff):
        idx = T.randperm(len(X), generator=g)[:m["batch"]]
        u = batch_top_u(net, loss_fn, X[idx], Y[idx])
        mass.append(float(((V.t() @ u) ** 2).sum()))
    mass = np.array(mass)
    rec = dict(tag=tag, K=K, S=S_eff, median=float(np.median(mass)),
               q10=float(np.percentile(mass, 10)), q90=float(np.percentile(mass, 90)),
               pass_gate=bool(np.median(mass) >= 0.8))
    print(f"[audit] {tag} K={K}: captured mass median={rec['median']:.3f} "
          f"[q10={rec['q10']:.3f}, q90={rec['q90']:.3f}] -> "
          f"{'PASS (verdict readable)' if rec['pass_gate'] else 'FAIL (K=8 verdict withheld; K=32 rerun)'}",
          flush=True)
    p = os.path.join(MS, "frame_audit.json")
    all_ = json.load(open(p)) if os.path.exists(p) else []
    all_ = [r for r in all_ if not (r["tag"] == tag and r["K"] == K)] + [rec]
    json.dump(all_, open(p, "w"), indent=1)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()
    tags = a.cells or sorted(f.replace("_ckpt.pt", "") for f in os.listdir(MS) if f.endswith("_ckpt.pt"))
    for t in tags:
        audit(t, K=a.K)
