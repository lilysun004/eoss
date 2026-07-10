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

THE KKT FRAME (the correct form of the distinction). The weather-universality result = a quasi-
potential U_eff(kappa): noise supplies fluctuations, DRIFT supplies the landscape, system parks at
min U_eff. drift(kappa) ~ alpha(kappa) - c*E[x^2](kappa): alpha = progressive-sharpening drive,
E[x^2] = stationary amplitude of the unstable coordinate (cubic self-stabilization shaves sharpness
in proportion, Damian et al.). E[x^2] needs AMPLIFICATION, which needs coupling to u_B. SGD couples
(fresh gradient) -> E[x^2] diverges as kappa -> stochastic edge -> shaving is a WALL -> kappa* pins
AT it (constraint ACTIVE, GBS=2 = saturation read off the path, multiplier>0). Momentum small-batch
can't couple (R>>1) -> no amplification wall -> kappa* set by where the DRIVE dies (alpha->0:
interpolation/alignment saturation) in the constraint INTERIOR (constraint SLACK). GD-at-EoS solves
min L s.t. S<=edge; the two regimes = the inequality active vs slack. Genuine BINARY (LP phases,
not thermodynamic) coexisting with continuous weather; order parameter = the multiplier, not any
fluctuation moment. kappa* = min(kappa_constraint [=edge, computable, GBS=2], kappa_exhaustion
[loss-geometry+history, not a stability quantity]) -- the north star's universal position exists
exactly on the branch where stability binds.

SHARPENED KICK PREDICTION -- restoring vs NEUTRAL DRIFT (calibration-free, immune to the threshold
ambiguities that killed the passive stats):
  ACTIVE (marginal SGD): kick kappa UP past edge -> immediate hard shave-back (wall is right there);
    kick DOWN into interior -> slow return via drive. RETURN FRACTION high (esp. up). Attractor.
  SLACK (metastable SGDM): kick kappa within the slack interior (up toward the wall OR down) ->
    NO restoring, kappa re-parks at the DISPLACED value (return fraction ~0 BOTH directions), until
    a kick past the wall triggers catapult/divergence. Parking lot, not attractor.
  => discriminator = return-fraction (does displaced kappa relax back to kappa0, or stay put?), NOT
     restoring-rate magnitude. Report the SGD-vs-SGDM contrast at matched batch; the crossover pair
     (b128, R~1) is where it can dissociate from position.
COMPANION (path dependence, M0'): active constraint = attractor (kappa0 reproducible across seeds/
warm-starts); slack = parking lot (history-dependent). Cheap check available from sweep seed-spread
+ a warm-start-from-flatter run (climbs back = drive alive; stays = exhausted).
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
        relax = kk[seg == 2]
        kpulse = float(np.nanmedian(kk[seg == 1]))             # displaced kappa (end of pulse)
        krelax = float(np.nanmedian(relax[-len(relax) // 3:])) if len(relax) > 6 else np.nan  # settled
        # RETURN FRACTION: 1 = relaxed back to kappa0 (ACTIVE/attractor); 0 = stayed displaced (SLACK)
        denom = kpulse - kappa0
        res[f"return_{tag}"] = float((kpulse - krelax) / denom) if abs(denom) > 1e-9 else np.nan
        res[f"gamma_{tag}"] = fit_relax_rate(relax - kappa0)    # secondary: rate
        res[f"disp_{tag}_over_sigma"] = denom / sigma          # power check: kick supra-noise?
        res[f"diverged_{tag}"] = bool(bad)                     # kicked past the wall?
        np.savez(os.path.join(OUT, f"trace_{name}_{tag}.npz"), kappa=kk, seg=seg, kappa0=kappa0)
    res["asymmetry"] = float((res.get("return_up") or np.nan) - (res.get("return_down") or np.nan))
    print(f"  {name:16s} kap0={kappa0:.2f} sig={sigma:.3f} | RETURN up={res.get('return_up'):+.2f} "
          f"dn={res.get('return_down'):+.2f} (disp/sig up={res['disp_up_over_sigma']:+.1f}) "
          f"| {'ACTIVE/attractor' if (res.get('return_up') or 0)>0.5 else 'SLACK/parking-lot'}", flush=True)
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
            print(f"  CONTRAST {pname}: kappa0 {sgd['kappa0']:.2f}(SGD)->{sgdm['kappa0']:.2f}(SGDM) | "
                  f"return_up {sgd.get('return_up'):+.2f}->{sgdm.get('return_up'):+.2f} "
                  f"=> {'DISSOCIATES (active->slack)' if (sgd.get('return_up') or 0)>0.5 and (sgdm.get('return_up') or 1)<0.3 else 'same character'}",
                  flush=True)

    print("\n===== VERDICT (return-fraction contrast across R = active vs slack constraint) =====")
    print(f"{'cell':30s}{'kappa0':>8}{'ret_up':>8}{'ret_dn':>8}{'constraint':>16}")
    for r in results:
        if r.get("diverged"):
            print(f"{r['name']:30s} diverged"); continue
        ru = r.get("return_up") or np.nan
        state = "ACTIVE" if ru > 0.5 else ("SLACK" if ru < 0.3 else "intermediate")
        print(f"{r['name']:30s}{r['kappa0']:8.2f}{ru:8.2f}{(r.get('return_down') or np.nan):8.2f}{state:>16}")
    print("\n KKT-PHASES: SGD returns (active/attractor, ret~1); SGDM neutral (slack/parking-lot, ret~0)")
    print("   at matched batch, with the crossover pair (b128) showing where active->slack flips.")
    print(" CONTINUUM: return fraction varies smoothly with kappa0, no sharp active/slack boundary.")


if __name__ == "__main__":
    main()
