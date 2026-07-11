"""
Slow-variable kick = perturb-and-relax on SHARPNESS (a scalar -> no rotating-coordinate problem),
the DECISIVE causal test after the paired sweep showed every PASSIVE event statistic is batch-noise-
set (weather is optimizer-independent). Question: is the metastable operating point held by a
RESTORING FORCE (active constraint / thermostat) or just PARKED (KKT slack constraint, nothing
pushes it in the interior)? Passive stats can't answer; displace the slow variable and watch.

Lineage: this is sharpening-suppression redesigned around the three ways its ancestor died --
  (i)  watch the RETURN of a displaced kappa (needs the shaving force, strong when active), not the
       resume of the sharpening DRIVE (weak/dead near interpolation -- what broke the old test);
  (ii) GENTLE displacements calibrated in sigma_lambda units (not a 10x hammer that moves the edge
       itself and measures a giant transient);
  (iii) matched-PAIR contrast + dense-enough logging + both directions + an amplitude LADDER.

KKT frame: quasi-potential U_eff(lambda), drift ~ alpha(lambda) - c*E[x^2]. SGD couples -> E[x^2]
walls up at the edge -> constraint ACTIVE (lambda* pinned, GBS=2). Momentum small-batch can't
couple (R>>1) -> no wall -> lambda* set by drive exhaustion in the SLACK interior. The object of
the experiment is the force-vs-displacement curve F(dlambda):
  ACTIVE (marginal):  ~0 force below, HARD WALL above -> one-sided, kinked at the edge.
  SLACK (metastable): F~0 across an interior RANGE (return fraction ~0 both directions), then a
                      remote wall -> the width of the flat region = the constraint's slack.
A single amplitude gives one point and cannot tell "weak spring" from "no spring then wall"; the
LADDER traces the curve. Deliverable: F(dlambda) per cell, SGD vs SGDM overlaid at matched (B,lr).

BOOKKEEPING (the lr-pulse subtlety): during a pulse eta changes, so kappa=eta*lambda moves even if
the landscape lambda hasn't -- track raw LAMBDA (held-out) as the state variable throughout; use
kappa only in baseline-eta phases. Displacement = "pulse moved lambda from lambda* to lambda*+d";
readout = "does lambda return to lambda* at the ORIGINAL eta". Log primitives (lambda_full, eta_t,
loss); kappa/GBS derived offline.

CAVEAT / control status: the lr-pulse is an INDIRECT actuator (it displaces lambda via the
constraint, so it works cleanly for the ACTIVE/marginal cell but weakly displaces a SLACK/metastable
lambda -- there's no wall to shave against). The eta-clean TRANSPLANT actuator (warm-start the SGDM
optimizer from the SGD twin's sharper checkpoint, capped below the wall) is the confirmation gate
and is NOT built here yet -- so until it is, a "phases" read from pulses alone is actuator-uncontrolled
and must be stated as such. b128 crossover pair: both twins sit near the edge, so BOTH returning is
the PREDICTED outcome -- it is the dissociation probe, not scored as continuum evidence.
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
from utils.measure import create_hessian_vector_product, flatt

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "slow_kick")
os.makedirs(OUT, exist_ok=True)


def lam_probe(net, loss_fn, Xp, Yp, u_prev, n_power=10):
    """Held-out lambda_max on a FIXED probe batch (warm power iteration). This is the STATE variable
    (landscape curvature), eta-independent -- unlike kappa=eta*lambda which moves during a pulse."""
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


def ar_tau(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 30 or np.std(x) < 1e-12:
        return np.nan
    z = x - x.mean(); a1 = np.dot(z[:-1], z[1:]) / (np.dot(z[:-1], z[:-1]) + 1e-30)
    return float(-1.0 / np.log(a1)) if 0 < a1 < 1 else np.nan


def block_sigma(x, nb=12):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < nb * 3:
        return float(np.std(x)) + 1e-30
    return float(np.std([b.mean() for b in np.array_split(x, nb)])) + 1e-30


def run_seg(state, lr_mults, seg_lens, seed, probe=512):
    """Run consecutive lr segments; log per step: lambda_full (STATE), eta, seg id, loss."""
    net, opt, loss_fn = state["net"], state["opt"], state["loss_fn"]
    X, Y = state["XY"]; batch = state["batch"]
    Xp, Yp = X[:probe], Y[:probe]
    net.load_state_dict(copy.deepcopy(state["net_sd"])); PR.set_optimizer_state(opt, state["opt_sd"])
    base_lr = float(opt.inner.param_groups[0]["lr"])     # base eta (train_plateau state has no 'lr')
    g = T.Generator().manual_seed(seed); u_prev = None
    lam, eta, seg, loss = [], [], [], []
    diverged = False
    for si, (mult, n) in enumerate(zip(lr_mults, seg_lens)):
        cur = base_lr * mult; opt.inner.param_groups[0]["lr"] = cur
        for _ in range(n):
            lv, u_prev = lam_probe(net, loss_fn, Xp, Yp, u_prev)
            lam.append(lv); eta.append(cur); seg.append(si)
            idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb); loss.append(lo.item())
            if not np.isfinite(lo.item()) or lo.item() > 1e6:
                diverged = True; break
            opt.zero_grad(); lo.backward(); opt.step()
        if diverged:
            break
    opt.inner.param_groups[0]["lr"] = base_lr
    return (np.array(lam), np.array(eta), np.array(seg), np.array(loss)), diverged


def _settle(a, frac=0.34):
    a = a[np.isfinite(a)]
    return float(np.median(a[-max(3, int(len(a) * frac)):])) if len(a) else np.nan


def kick_cell(name, optn, params, batch, lr, steps, seed, order, eps_ladder=(0.15, 0.30, 0.50)):
    """Baseline -> lambda*, sigma_lambda, tau; then an ascending amplitude LADDER per direction
    (stop-on-divergence), windows sized from tau. Returns F(dlambda) points on the STATE variable."""
    st = PR.train_plateau(optn, params, batch, lr, steps)
    if st is None:
        print(f"  {name}: diverged in training", flush=True); return dict(name=name, diverged=True)
    (lb, _, _, _), _ = run_seg(st, [1.0], [max(1200, 1000)], seed=11)   # baseline M>=1000
    lam_star = float(np.median(lb[np.isfinite(lb)])); sig = block_sigma(lb); tau = ar_tau(lb)
    tau = 250.0 if not np.isfinite(tau) else tau
    Plen = int(min(max(2.0 * tau, 400), 1200))          # pulse ~2 tau (moves lambda), capped
    Rlen = int(min(max(5.0 * tau, 1500), 3500))         # relax >=5 tau (the #2 fix), capped
    res = dict(name=name, optn=optn, batch=batch, lr=lr, seed=seed, diverged=False,
               lam_star=lam_star, sigma_lam=sig, tau=tau, Plen=Plen, Rlen=Rlen,
               kappa0=lr * lam_star, points=[])
    for direction in order:                              # randomized per seed (drift != asymmetry)
        s = +1 if direction == "up" else -1
        for eps in eps_ladder:                           # ascending; stop this direction on divergence
            (lam, eta, seg, loss), bad = run_seg(st, [1.0, 1 + s * eps, 1.0], [60, Plen, Rlen], seed=23 + seed)
            lam_pulse = _settle(lam[seg == 1]); lam_end = _settle(lam[seg == 2])
            dlam = lam_pulse - lam_star                  # ACHIEVED displacement (measured, not assumed)
            rf = float((lam_pulse - lam_end) / dlam) if abs(dlam) > 1e-9 else np.nan
            pt = dict(direction=direction, eps=eps, dlam=dlam, dlam_over_sigma=dlam / sig,
                      return_frac=rf, lam_pulse=lam_pulse, lam_end=lam_end, diverged=bool(bad))
            res["points"].append(pt)
            np.savez(os.path.join(OUT, f"trace_{name}_{direction}_{eps}.npz"),
                     lam=lam, eta=eta, seg=seg, loss=loss, lam_star=lam_star)
            print(f"    {name:18s} {direction:4s} eps{eps:.2f}: dlam/sig={dlam/sig:+5.1f} "
                  f"return={rf:+.2f} {'DIVERGED(wall)' if bad else ''}", flush=True)
            if bad:
                break
    return res


# matched (batch, lr) pairs from the sweep's live cells, DECISIVE ones first (per priority note).
PAIRS = [
    ("b32_lr005",  32,  0.005, [("SGD", {}), ("SGD-Momentum", {"beta": 0.9})]),   # metastable SGDM (2.5 vs 0.6)
    ("b8_lr004",   8,   0.004, [("SGD", {}), ("SGD-Momentum", {"beta": 0.6})]),   # partial drop
    ("b512_lr008", 512, 0.008, [("SGD", {}), ("SGD-Momentum", {"beta": 0.9})]),   # marginal-with-memory anchor
    ("b128_lr006", 128, 0.006, [("SGD", {}), ("SGD-Momentum", {"beta": 0.9})]),   # crossover R~1 (BOTH return = predicted)
]


def summarize(cell):
    """collapse F(dlambda) to per-direction return fractions (median over supra-noise, non-diverged
    points) -- the interior response; report where the ladder hit the wall."""
    out = {}
    for d in ("up", "down"):
        pts = [p for p in cell.get("points", []) if p["direction"] == d and abs(p["dlam_over_sigma"]) > 1.0]
        interior = [p["return_frac"] for p in pts if not p["diverged"] and np.isfinite(p["return_frac"])]
        wall = next((p["eps"] for p in pts if p["diverged"]), None)
        out[d] = (float(np.median(interior)) if interior else np.nan, wall)
    return out


def main():
    results = []
    marginal_only = os.environ.get("EOSS_MARGINAL_ONLY") == "1"   # pulse is a constraint-side
    # actuator: it can only displace an ACTIVE-constraint (marginal) lambda. Slack (metastable)
    # cells get displacement ~noise (return=noise/noise) -> prune them; the transplant actuator
    # (transplant.py) is what probes the slack interior. Pulse run keeps: marginal F(dlambda) +
    # the eps-at-divergence = each marginal cell's measured stochastic WALL POSITION.
    for pi, (pname, batch, lr, twins) in enumerate(PAIRS):
        if marginal_only:
            twins = [t for t in twins if t[0] == "SGD"]
        print(f"\n=== PAIR {pname} (b{batch} lr{lr}) ===", flush=True)
        cells = {}
        for optn, prm in twins:
            short = "SGD" if optn == "SGD" else f"SGDM{prm['beta']}"
            for seed in (0, 1):                          # 2 seeds, alternating direction order
                order = ["up", "down"] if seed == 0 else ["down", "up"]
                r = kick_cell(f"{pname}_{short}_s{seed}", optn, prm, batch, lr, 4000, seed, order)
                results.append(r); cells.setdefault(short, []).append(r)
            json.dump(results, open(os.path.join(OUT, "slow_kick.json"), "w"), indent=2, default=str)
        # CONTRAST at matched batch (interior return fraction, seed-pooled)
        def pooled(short):
            rs = [summarize(c) for c in cells.get(short, []) if not c.get("diverged")]
            return {d: np.nanmean([s[d][0] for s in rs]) for d in ("up", "down")} if rs else None
        sgd = pooled("SGD"); sgdm = pooled(next((k for k in cells if k != "SGD"), ""))
        note = "  [crossover: BOTH-return is PREDICTED, not continuum evidence]" if pname == "b128_lr006" else ""
        if sgd and sgdm:
            print(f"  CONTRAST {pname}: interior return_up SGD={sgd['up']:+.2f} vs SGDM={sgdm['up']:+.2f} | "
                  f"return_down SGD={sgd['down']:+.2f} vs SGDM={sgdm['down']:+.2f}{note}", flush=True)

    print("\n===== VERDICT (F(dlambda) interior return, matched-batch contrast) =====")
    print(" KKT-PHASES: SGD interior return ~1 (active/wall), SGDM interior return ~0 both directions")
    print("   (slack/parking-lot) -- the flat interior + a finite wall eps = measured slack width.")
    print(" CONTINUUM: interior return varies smoothly with lambda*, no active/slack step.")
    print(" ACTUATOR CAVEAT: lr-pulse weakly displaces a slack lambda; a 'phases' read is NOT")
    print("   confirmed until the eta-clean TRANSPLANT control agrees. Build that before claiming phases.")


if __name__ == "__main__":
    main()
