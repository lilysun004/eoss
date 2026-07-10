"""
Slow-variable kick -- the DECISIVE test (promoted from icing) after tonight's paired sweep showed
every PASSIVE event statistic is batch-noise-set (clustering/rate/excursion-size all track batch,
not the buffer). Only equilibrium position (kappa/GBS) discriminates. The one question passive
stats provably cannot answer: is the position difference accompanied by a FEEDBACK-STRUCTURE
difference (a thermostat that shaves-from-above at the edge, absent in the basin -> phases), or is
it a smooth shift of operating point with the same dynamics everywhere (continuum)?

Design, with tonight's lessons built in:
  - MATCHED-BATCH CONTRAST is built into the experiment (not bolted on): at each matched (batch, lr)
    we kick the SGD twin (marginal, at edge) and the SGDM twin (below edge) with the SAME protocol
    and report the CONTRAST. Every instrument that skipped this control produced a false headline.
  - TRACK THE HELD-OUT SLOW VARIABLE, not per-batch kappa: kappa_slow = lr*lambda_max(fixed 512
    probe) via warm power iteration each step. Per-batch kappa is noise-dominated (tonight's
    finding) and would drown the relaxation signal.
  - kappa-CALIBRATED amplitude: pulse lr by (1 +/- eps); report the displacement in units of the
    measured baseline kappa_slow fluctuation (disp/sigma), so we know the kick is supra-noise.
  - QUALITATIVE asymmetry prediction (no threshold choice): marginal shaves-from-above hard
    (restore-from-above rate stays O(1)); basin re-parks weakly/symmetrically (rate -> 0).

Pairs span R (metastable -> crossover -> marginal-with-memory) so restoring can DISSOCIATE from
position at mid-R -- the phases-vs-continuum discriminator. Verdict:
  restoring qualitatively different / dissociates from position  -> PHASES (position = order param).
  restoring weakens smoothly with R, tracks position everywhere  -> R-CONTINUUM with endpoints.
"""
import os, sys, json, copy
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault("DATASETS", "/Users/xq/Desktop/moonshot/eoss/datasets")
os.environ.setdefault("EOSS_SKIP_CHECKSUM", "1")
if os.environ.get("EOSS_SKIP_CHECKSUM"):
    import torchvision.datasets.cifar as _cifar
    _cifar.check_integrity = lambda *a, **k: True

import experiments.perturb_relax as PR
from utils.measure import create_hessian_vector_product, flatt, param_vector

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "slow_kick")
os.makedirs(OUT, exist_ok=True)


def lam_probe(net, loss_fn, Xp, Yp, u_prev, n_power=10):
    """Held-out lambda_max on a FIXED probe batch via warm power iteration (clean slow variable)."""
    pr = net(Xp).squeeze(-1); lo = loss_fn(pr, Yp)
    params = [p for p in net.parameters() if p.requires_grad]
    grads = T.autograd.grad(lo, params, create_graph=True); g = flatt(grads)
    hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
    try:
        u = u_prev.clone() if u_prev is not None else g.detach().clone()
        u = u / (u.norm() + 1e-30)
        for _ in range(n_power):
            Hu = hvp(u, retain_graph_override=True); nrm = float(Hu.norm())
            if nrm < 1e-20:
                break
            u = (Hu / nrm).detach()
        lam = float(T.dot(u, hvp(u, retain_graph_override=True)))
    finally:
        hvp.free_memory()
    return lam, u


def run_segment(state, lr_mults, seg_lens, seed, probe=512):
    """Run consecutive lr-pulse segments; log held-out kappa_slow = lr*lambda_probe each step."""
    net, opt, loss_fn = state["net"], state["opt"], state["loss_fn"]
    X, Y = state["XY"]; batch = state["batch"]; base_lr = state["lr"]
    Xp, Yp = X[:probe], Y[:probe]
    net.load_state_dict(copy.deepcopy(state["net_sd"])); PR.set_optimizer_state(opt, state["opt_sd"])
    g = T.Generator().manual_seed(seed); u_prev = None
    kap, seg = [], []
    for si, (mult, n) in enumerate(zip(lr_mults, seg_lens)):
        opt.inner.param_groups[0]["lr"] = base_lr * mult
        for _ in range(n):
            lam, u_prev = lam_probe(net, loss_fn, Xp, Yp, u_prev)
            kap.append(base_lr * mult * lam); seg.append(si)
            idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            if not np.isfinite(lo.item()) or lo.item() > 1e6:
                opt.inner.param_groups[0]["lr"] = base_lr
                return np.array(kap), np.array(seg), True
            opt.zero_grad(); lo.backward(); opt.step()
    opt.inner.param_groups[0]["lr"] = base_lr
    return np.array(kap), np.array(seg), False


def fit_relax_rate(dk):
    dk = dk[np.isfinite(dk)]
    if len(dk) < 8:
        return np.nan
    a = float(np.dot(dk[:-1], dk[1:]) / (np.dot(dk[:-1], dk[:-1]) + 1e-30))
    return float(1.0 - a) if 0 < a < 1.5 else np.nan


def kick_cell(name, optn, params, batch, lr, steps, M=300, P=80, Rlx=300, eps=0.5):
    st = PR.train_plateau(optn, params, batch, lr, steps)
    if st is None:
        print(f"  {name}: diverged in training", flush=True); return dict(name=name, diverged=True)
    kb, _, _ = run_segment(st, [1.0], [M], seed=11)          # baseline -> kappa0, sigma, passive S3
    k = kb[np.isfinite(kb)]; kappa0 = float(np.median(k)); sigma = float(np.std(k)) + 1e-30
    x = k[:-1] - k.mean(); passive = -float(np.dot(x, np.diff(k)) / (np.dot(x, x) + 1e-30))
    res = dict(name=name, optn=optn, batch=batch, lr=lr, diverged=False,
               kappa0=kappa0, kappa_sigma=sigma, passive_restore=passive)
    for tag, mult in [("up", 1 + eps), ("down", 1 - eps)]:
        kk, seg, bad = run_segment(st, [1.0, mult, 1.0], [60, P, Rlx], seed=23)
        res[f"gamma_{tag}"] = fit_relax_rate(kk[seg == 2] - kappa0)
        disp = float(np.nanmedian(kk[seg == 1]) - kappa0)
        res[f"disp_{tag}_over_sigma"] = disp / sigma        # power check: kick supra-noise?
        np.savez(os.path.join(OUT, f"trace_{name}_{tag}.npz"), kappa=kk, seg=seg, kappa0=kappa0)
    res["asymmetry"] = float((res.get("gamma_up") or np.nan) - (res.get("gamma_down") or np.nan))
    print(f"  {name:14s} kap0={kappa0:.2f} sig={sigma:.3f} passive={passive:+.3f} "
          f"g_up={res.get('gamma_up'):+.3f} g_dn={res.get('gamma_down'):+.3f} asym={res['asymmetry']:+.3f} "
          f"disp/sig(up)={res['disp_up_over_sigma']:+.1f}", flush=True)
    return res


# matched (batch, lr) pairs from tonight's live cells, spanning R: SGD twin (marginal) vs SGDM twin
PAIRS = [
    ("b32_lr005",  32,  0.005, [("SGD", {}), ("SGD-Momentum", {"beta": 0.9})]),   # metastable SGDM (k 2.5 vs 0.6)
    ("b8_lr004",   8,   0.004, [("SGD", {}), ("SGD-Momentum", {"beta": 0.6})]),   # partial (k 2.3 vs 1.2)
    ("b128_lr006", 128, 0.006, [("SGD", {}), ("SGD-Momentum", {"beta": 0.9})]),   # crossover R~1 (both ~edge)
    ("b512_lr008", 512, 0.008, [("SGD", {}), ("SGD-Momentum", {"beta": 0.9})]),   # marginal-with-memory (both edge)
]


def main():
    results = []
    for pname, batch, lr, twins in PAIRS:
        print(f"\n=== PAIR {pname} (b{batch} lr{lr}) ===", flush=True)
        pair = {}
        for optn, params in twins:
            short = "SGD" if optn == "SGD" else f"SGDM{params['beta']}"
            r = kick_cell(f"{pname}_{short}", optn, params, batch, lr, steps=4000)
            pair[short] = r; results.append(r)
            json.dump(results, open(os.path.join(OUT, "slow_kick.json"), "w"), indent=2, default=str)
        # CONTRAST at matched batch: restoring of SGD (marginal) vs SGDM (below edge)
        sgd = pair.get("SGD"); sgdm = next((v for k, v in pair.items() if k != "SGD"), None)
        if sgd and sgdm and not sgd.get("diverged") and not sgdm.get("diverged"):
            print(f"  CONTRAST {pname}: kappa0 {sgd['kappa0']:.2f}->{sgdm['kappa0']:.2f} | "
                  f"g_up {sgd.get('gamma_up'):+.3f}->{sgdm.get('gamma_up'):+.3f} | "
                  f"asym {sgd['asymmetry']:+.3f}->{sgdm['asymmetry']:+.3f}", flush=True)

    print("\n===== VERDICT (restoring-force contrast across R) =====")
    print(f"{'pair':12s}{'cell':16s}{'kappa0':>8}{'g_up':>8}{'g_down':>8}{'asym':>8}{'disp/sig':>9}")
    for r in results:
        if r.get("diverged"):
            print(f"{r['name']:28s} diverged"); continue
        print(f"{r['name']:28s}{r['kappa0']:8.2f}{r.get('gamma_up',float('nan')):8.3f}"
              f"{r.get('gamma_down',float('nan')):8.3f}{r['asymmetry']:8.3f}{r.get('disp_up_over_sigma',float('nan')):9.1f}")
    print("\n PHASES: marginal (SGD) restores hard & asymmetric, SGDM restoring -> 0 / dissociates at crossover.")
    print(" CONTINUUM: restoring weakens smoothly with kappa0 (position), same character everywhere.")


if __name__ == "__main__":
    main()
