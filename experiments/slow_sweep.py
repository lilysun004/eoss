"""
Dense slow-variable logging runner (one cell). No fast-coordinate kicks -- the archetype 2x2
proved those are blind at small batch. Logs the slow loop densely so every metastable question
(kappa pinned vs wandering, restoring force, catapult renewal, Kesten moments) becomes offline
analysis, and so each SGDM cell can be compared to its SGD twin at matched (batch, lr).

Per measured step (stride-controlled), one HVP (H_B s) + one warm LOBPCG (lambda_max, u_B):
  loss, gbs = s'Hs/(-g's), kappa = lr*lam_batch, a_t = 1 - lr*(s'Hs/||s||^2)  [one-step multiplier],
  alpha_g = gbs/kappa, grad/step/buf norms, cos(u_Bt,u_Bt+1), cos(buf/grad/step, u_B), cos(buf,grad),
  cos(step,grad).
Sparse: lam_full (subset-2048 sharpness) every `lam_full_every`; R_noise (exogenous rotation at
frozen theta, `n_noise` resampled batches) every `rnoise_every` -- the non-circular denominator
for R.

Stop = catapult budget: after `warmup`, once detect_catapults(loss) >= catapult_target, or step
>= max_steps. Resume-safe: dense arrays flushed to dense.npz every `flush_every` steps (overwrite),
meta.json written with status last (a cell is 'done' only when fully finished; partial cells are
re-run fresh by the driver but their partial dense.npz survives for inspection).
"""
import os, sys, json, time
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
import experiments.mechanism_buffer_rotation as M
from utils.optimizer import create_optimizer
from utils.measure import (compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
                           flatt, param_vector)

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))

FIELDS = ["step", "loss", "gbs", "kappa", "lam_batch", "a_t", "alpha_g", "grad_norm", "step_norm",
          "buf_norm", "cos_uu", "cos_bu", "cos_gu", "cos_su", "cos_bg", "cos_sg"]


def _rnoise(net, loss_fn, X, Y, batch, memory, n_noise=5):
    """Exogenous rotation: at FROZEN theta, resample n_noise batches, measure u_B dispersion.
    R_noise = memory / tau_noise, tau_noise = 1/(1 - mean pairwise |cos|). Non-circular (no motion)."""
    cache = EigenvectorCache(1); us = []
    g = T.Generator().manual_seed(12345)
    for _ in range(n_noise):
        idx = T.randperm(len(X), generator=g)[:batch]
        _, u = M.batch_top_eigvec(net, loss_fn, X[idx], Y[idx], cache)
        us.append(u)
    cs = [M.cosabs(us[i], us[j]) for i in range(len(us)) for j in range(i + 1, len(us))]
    rot = float(np.mean(cs)) if cs else float("nan")
    tau = 1.0 / max(1e-6, 1.0 - rot)
    return memory / tau, tau


def run_cell(tag, optn, beta, batch, lr, out_dir, catapult_target=25, max_steps=40000,
             warmup=6000, stride=1, seed=0, lam_full_every=100, rnoise_every=400,
             flush_every=2000, div_cap=1e6, n_power=6):
    t0 = time.time()
    cell = os.path.join(out_dir, tag); os.makedirs(cell, exist_ok=True)
    meta_path = os.path.join(cell, "meta.json")
    T.manual_seed(seed)
    X, Y = L.get_data(); net, loss_fn = L.build()
    params_dict = {} if optn == "SGD" else {"beta": beta}
    opt = create_optimizer(optn, net, lr, params_dict)
    params = [p for p in net.parameters() if p.requires_grad]
    memory = 1.0 / (1.0 - beta) if beta > 0 else 1.0
    is_mom = beta > 0
    cache = EigenvectorCache(1); u_prev = None
    dense = {k: [] for k in FIELDS}
    sparse = dict(lf_step=[], lam_full=[], rn_step=[], R_noise=[], tau_noise=[])
    gen = T.Generator().manual_seed(1000 + seed)
    diverged = False; ncat = 0

    def flush(status):
        np.savez(os.path.join(cell, "dense.npz"),
                 **{k: np.array(v, float) for k, v in dense.items()},
                 **{k: np.array(v, float) for k, v in sparse.items()})
        json.dump(dict(tag=tag, optn=optn, beta=beta, batch=batch, lr=lr, seed=seed,
                       status=status, steps=len(dense["step"]), n_catapults=int(ncat),
                       diverged=diverged, catapult_target=catapult_target, stride=stride,
                       elapsed_s=time.time() - t0), open(meta_path, "w"), indent=1)

    step = 0
    while step < max_steps:
        idx = T.randperm(len(X), generator=gen)[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb); lv = lo.item()
        if not np.isfinite(lv) or lv > div_cap:
            diverged = True; break
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads); gd = g.detach()
        s = opt.compute_step_direction(g, params).detach()
        measure = (step % stride == 0)
        if measure:
            m = M.buffer_flat(opt, params) if is_mom else gd
            # ONE HVP graph for everything: Hs (GBS/a_t) + warm power iteration for (lam_max, u_B).
            # LOBPCG-per-step is 243ms on this 789k-param net; warm power iter (~6 HVPs) is ~10ms.
            hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
            try:
                Hs = hvp(s, retain_graph_override=True)
                sHs = float(T.dot(s, Hs)); A = -float(T.dot(gd, s)); ss = float(T.dot(s, s))
                u = u_prev.clone() if u_prev is not None else T.randn(g.numel())
                u = u / (u.norm() + 1e-30)
                for _ in range(n_power):
                    Hu = hvp(u, retain_graph_override=True); nrm = float(Hu.norm())
                    if nrm < 1e-20:
                        break
                    u = (Hu / nrm).detach()
                lam = float(T.dot(u, hvp(u, retain_graph_override=True)))   # Rayleigh quotient
            finally:
                hvp.free_memory()
            gbs = sHs / A if abs(A) > 1e-15 else float("nan")
            kappa = lr * lam if np.isfinite(lam) else float("nan")
            a_t = 1.0 - lr * (sHs / ss) if ss > 1e-24 else float("nan")
            dense["step"].append(step); dense["loss"].append(lv)
            dense["gbs"].append(gbs); dense["kappa"].append(kappa); dense["lam_batch"].append(lam)
            dense["a_t"].append(a_t); dense["alpha_g"].append(gbs / kappa if kappa else float("nan"))
            dense["grad_norm"].append(float(gd.norm())); dense["step_norm"].append(float(s.norm()))
            dense["buf_norm"].append(float(m.norm()))
            dense["cos_uu"].append(M.cosabs(u, u_prev) if u_prev is not None else float("nan"))
            dense["cos_bu"].append(M.cosabs(m, u)); dense["cos_gu"].append(M.cosabs(gd, u))
            dense["cos_su"].append(M.cosabs(s, u)); dense["cos_bg"].append(M.cosabs(m, gd))
            dense["cos_sg"].append(M.cosabs(s, gd)); u_prev = u.clone()
            if len(dense["step"]) % lam_full_every == 0:
                Xs, Ys = X[:2048], Y[:2048]; los = loss_fn(net(Xs).squeeze(-1), Ys)
                lf = float(compute_eigenvalues(los, net, k=1, max_iterations=60, reltol=0.01,
                           eigenvector_cache=EigenvectorCache(1), return_eigenvectors=False,
                           use_power_iteration=False))
                sparse["lf_step"].append(step); sparse["lam_full"].append(lr * lf)
            if len(dense["step"]) % rnoise_every == 0:
                rn, tn = _rnoise(net, loss_fn, X, Y, batch, memory)
                sparse["rn_step"].append(step); sparse["R_noise"].append(rn); sparse["tau_noise"].append(tn)
        # optimizer step from already-computed grads (no second backward)
        opt.zero_grad()
        for p, gr in zip(params, grads):
            p.grad = gr.detach()
        opt.step(); step += 1
        # catapult budget
        if step >= warmup and step % 200 == 0:
            ncat = len(L.detect_catapults(np.array(dense["loss"], float)))
            if ncat >= catapult_target:
                break
        if step % flush_every == 0:
            flush("running")
    ncat = len(L.detect_catapults(np.array(dense["loss"], float))) if dense["loss"] else 0
    flush("diverged" if diverged else "done")
    return meta_path


if __name__ == "__main__":
    # direct single-cell invocation: --optn --beta --batch --lr --tag --seed ... (driver uses this)
    import argparse
    ap = argparse.ArgumentParser()
    for k in ["tag", "optn"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--beta", type=float, default=0.0); ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "results", "slow_sweep"))
    ap.add_argument("--catapult_target", type=int, default=25); ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=6000); ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    run_cell(a.tag, a.optn, a.beta, a.batch, a.lr, a.out_dir, a.catapult_target, a.max_steps,
             a.warmup, a.stride, a.seed)
    print(f"[slow_sweep] {a.tag} done")
