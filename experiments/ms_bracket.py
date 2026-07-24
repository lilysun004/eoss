"""
Ground-truth divergence bracket (ADDENDUM 4 gating item): continue REAL training from the
frozen plateau checkpoint (theta*, buffer* restored) at c x operating lr, short horizon
(3000 steps, limits lambda re-adaptation -- the registered clean comparator). Frame- and
estimator-independent arbiter for the (i,ii)-vs-(iii) c*_2 disagreement.
Interpretation map pre-committed in KSPEC_PREREG_ANNOTATIONS.md ADDENDUM 4.
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
from utils.optimizer import create_optimizer
from utils.measure import create_hessian_vector_product, flatt
from utils.curvature_segment import set_params_inplace

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", os.environ.get("EOSS_KSPEC_OUT", "kspec")); MS = os.path.join(OUT, "ms")


def run_bracket(tag, c, steps=3000, div_cap=1e6, lam_every=50):
    ck = T.load(os.path.join(MS, f"{tag}_ckpt.pt"), weights_only=False)
    m = ck["meta"]
    T.manual_seed(m["seed"])
    X, Y = L.get_data(); net, loss_fn = L.build()
    set_params_inplace(net, ck["theta"])
    lr = c * m["lr"]
    params_dict = ({} if m["optn"] == "SGD" else
                   ({"beta1": m["beta"], "beta2": 0.99} if m["optn"] == "Adam"
                    else {"momentum": m["beta"]} if m["optn"] == "Muon"
                    else {"beta": m["beta"]}))
    opt = create_optimizer(m["optn"], net, lr, params_dict)
    params = [p for p in net.parameters() if p.requires_grad]
    if "opt_state" in ck:                              # full state restore (any optimizer)
        opt.inner.load_state_dict(ck["opt_state"])
        for grp in opt.inner.param_groups:
            grp["lr"] = lr                             # state dict carries old lr -- override
    else:
        off = 0
        for p in params:                               # legacy ckpt: momentum buffer only
            n = p.numel()
            opt.inner.state[p] = {"momentum_buffer": ck["buf"][off:off + n].view_as(p).clone()}
            off += n
    gen = T.Generator().manual_seed(9000 + m["seed"])
    losses, kappas, u_prev = [], [], None
    died = None
    for step in range(steps):
        idx = T.randperm(len(X), generator=gen)[:m["batch"]]
        lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        lv = float(lo)
        losses.append(lv)
        if not np.isfinite(lv) or lv > div_cap:
            died = step; break
        grads = T.autograd.grad(lo, params, create_graph=(step % lam_every == 0))
        if step % lam_every == 0:
            hvp = create_hessian_vector_product(lo, net, params=params, grads=grads,
                                                flat_grads=flatt(grads))
            try:
                u = (u_prev if u_prev is not None else flatt(grads).detach()).clone()
                u /= u.norm() + 1e-30
                for _ in range(8):
                    Hu = hvp(u, retain_graph_override=True)
                    n_ = float(Hu.norm())
                    if n_ < 1e-20: break
                    u = (Hu / n_).detach()
                lam = float(u @ hvp(u, retain_graph_override=True))
                kappas.append((step, lr * lam)); u_prev = u
            finally:
                hvp.free_memory()
        opt.zero_grad()
        for p, gr in zip(params, grads):
            p.grad = gr.detach()
        opt.step()
    rec = dict(tag=tag, c=c, lr=lr, died_at=died, max_loss=float(np.nanmax(losses)),
               final_loss=losses[-1] if died is None else None,
               kappa_trace=[(int(s), float(k)) for s, k in kappas])
    p = os.path.join(MS, "bracket.json")
    all_ = json.load(open(p)) if os.path.exists(p) else []
    all_.append(rec); json.dump(all_, open(p, "w"), indent=1)
    ktr = " ".join(f"{k:.2f}" for _, k in kappas[:8]) + " ... " + \
          " ".join(f"{k:.2f}" for _, k in kappas[-3:]) if len(kappas) > 10 else \
          " ".join(f"{k:.2f}" for _, k in kappas)
    print(f"[bracket] {tag} c={c}: {'DIED at step ' + str(died) if died is not None else 'SURVIVED ' + str(steps)} "
          f"max_loss={rec['max_loss']:.3g} kappa: {ktr}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L_b128_beta0.9_s0")
    ap.add_argument("--cs", nargs="+", type=float, default=[1.1, 1.3, 1.5])
    a = ap.parse_args()
    for c in a.cs:
        run_bracket(a.tag, c)
