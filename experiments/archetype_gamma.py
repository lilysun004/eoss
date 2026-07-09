"""
Archetype coupled-twin gamma with ROTATION-ROBUST coordinates.

Motivation: in the sweep, gamma_proj is fit to <dv, u_FIXED> where u_FIXED is the kick-time top
eigenvector. At small batch the unstable direction rotates out from under that fixed coordinate
within ~1 step (tau_rot~1), so <dv,u_fixed> decays because the COORDINATE rotates, not because dv
contracts -- dragging gamma_proj spuriously negative and destroying the marginal (gamma~0) vs
metastable (gamma<0) separation. (Corrections-log item 4 reappearing inside the instrument.)
The full dv vectors were never saved by the sweep, so this can't be fixed post-hoc -> rerun on a
few archetypes, measuring the difference trajectory in FOUR coordinates simultaneously:

  fixed  : <dv, u0>            (reproduces the sweep; expected artifact-prone at small batch)
  track  : <dv, u_t>          (u_t = top eigvec recomputed each step, warm-started, sign-aligned)
  subK   : ||V^T dv||         (V = top-K subspace at theta*; robust if u stays in span(V))
  full   : ||dv||             (bulk; also tells us if the coupled dynamics are chaotic off-u)

Decisive test (all coupled twins, identical batches, amp <= 1x natural, several seeds):
  - marginal small-batch (SGD b8) should read gamma_track ~ 0 while gamma_fixed < 0  -> artifact real
  - metastable (SGDM b8)          should read gamma_track < 0                         -> genuine decay
  If track/subK separate marginal from metastable where fixed does not, gamma_proj is vindicated
  as a *concept* and the fix is known (track u). If even track can't separate them, the instrument
  itself is suspect. Either way this is the cheap decisive check (a few cells, not 65).
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
import experiments.long_train_grid as L
from utils.optimizer import create_optimizer
from utils.measure import compute_eigenvalues, EigenvectorCache, param_vector

T.set_num_threads(4)
OUT = os.path.join(_REPO, "results", "archetype_gamma")
os.makedirs(OUT, exist_ok=True)


def set_params(net, vec):
    i = 0
    with T.no_grad():
        for p in net.parameters():
            n = p.numel(); p.copy_(vec[i:i + n].view_as(p)); i += n


def top_eigvec(net, loss_fn, X, Y, cache, probe=512):
    Xs, Ys = X[:probe], Y[:probe]
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, u = compute_eigenvalues(lo, net, k=1, max_iterations=40, reltol=0.01,
                               eigenvector_cache=cache, return_eigenvectors=True,
                               use_power_iteration=False)
    u = u.detach().reshape(-1); return u / (u.norm() + 1e-30)


def topK_subspace(net, loss_fn, X, Y, K=6, probe=2048):
    Xs, Ys = X[:probe], Y[:probe]
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, V = compute_eigenvalues(lo, net, k=K, max_iterations=80, reltol=0.01,
                               eigenvector_cache=EigenvectorCache(K),
                               return_eigenvectors=True, use_power_iteration=False)
    return V.detach()   # [n, K]


def fit_gamma(series, frac=0.6):
    """slope of log|series| vs step over the first `frac` (before saturation); nan-safe."""
    y = np.abs(np.asarray(series, float)); n = len(y)
    y = y[:max(5, int(n * frac))]
    t = np.arange(len(y)); ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 5:
        return np.nan
    return float(np.polyfit(t[ok], np.log(y[ok]), 1)[0])


def coupled_multi(state, u0, V, optn, params, lr, amp, K, seed, probe=512):
    """Coupled kicked/reference twins on identical batches; return dv in 4 coordinates."""
    loss_fn = state["loss_fn"]; X, Y = state["XY"]; theta = state["theta_star"]; batch = state["batch"]
    netK, _ = L.build(); netK.load_state_dict(copy.deepcopy(state["net_sd"]))
    optK = create_optimizer(optn, netK, lr, params); PR.set_optimizer_state(optK, state["opt_sd"])
    set_params(netK, theta + amp * u0)
    netR, _ = L.build(); netR.load_state_dict(copy.deepcopy(state["net_sd"]))
    optR = create_optimizer(optn, netR, lr, params); PR.set_optimizer_state(optR, state["opt_sd"])
    cache = EigenvectorCache(1); u_prev = u0.clone()
    fixed, track, subK, full = [], [], [], []
    g = T.Generator().manual_seed(seed)
    for t in range(K):
        with T.no_grad():
            dv = param_vector(netK) - param_vector(netR)
        fixed.append(float(T.dot(dv, u0))); full.append(float(dv.norm()))
        subK.append(float((V.t() @ dv).norm()))
        u_t = top_eigvec(netR, loss_fn, X, Y, cache, probe=probe)
        if float(T.dot(u_t, u_prev)) < 0:
            u_t = -u_t
        track.append(float(T.dot(dv, u_t))); u_prev = u_t
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        bad = False
        for net, opt in ((netK, optK), (netR, optR)):
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            if not np.isfinite(lo.item()) or lo.item() > 1e6:
                bad = True
            opt.zero_grad(); lo.backward(); opt.step()
        if bad:
            break
    return dict(fixed=np.array(fixed), track=np.array(track), subK=np.array(subK), full=np.array(full))


def run_archetype(name, optn, params, batch, lr, steps, expect, K=100, seeds=3):
    print(f"\n=== {name}: {optn} {params} b{batch} lr{lr}  (expect {expect}) ===", flush=True)
    st = PR.train_plateau(optn, params, batch, lr, steps)
    if st is None:
        print("  diverged"); return dict(name=name, diverged=True, expect=expect)
    u0 = st["u_hess"]; V = topK_subspace(st["net"], st["loss_fn"], *st["XY"], K=6)
    # natural amplitude from the free u-projection
    x0, _ = PR.relax(st, u0, 0.0, K=K, seed=1); scale = float(np.nanstd(x0)) + 1e-12
    amp = 0.5 * scale
    coords = ["fixed", "track", "subK", "full"]
    acc = {c: [] for c in coords}
    for s in range(seeds):
        ser = coupled_multi(st, u0, V, optn, params, lr, amp, K, seed=200 + s)
        for c in coords:
            acc[c].append(fit_gamma(ser[c]))
        if s == 0:
            np.savez(os.path.join(OUT, f"series_{name}.npz"), **ser)
    rec = dict(name=name, optn=optn, params=params, batch=batch, lr=lr, expect=expect,
               diverged=False, natural_scale=scale, amp=amp, n_seeds=seeds)
    for c in coords:
        a = np.array(acc[c], float)
        rec[f"gamma_{c}_mean"] = float(np.nanmean(a)); rec[f"gamma_{c}_std"] = float(np.nanstd(a))
    print(f"  gamma  fixed={rec['gamma_fixed_mean']:+.4f}+/-{rec['gamma_fixed_std']:.4f}  "
          f"track={rec['gamma_track_mean']:+.4f}+/-{rec['gamma_track_std']:.4f}  "
          f"subK={rec['gamma_subK_mean']:+.4f}+/-{rec['gamma_subK_std']:.4f}  "
          f"full={rec['gamma_full_mean']:+.4f}+/-{rec['gamma_full_std']:.4f}", flush=True)
    return rec


ARCHETYPES = [
    ("SGD_b2048",     "SGD",          {},            2048, 0.02,  4000, "marginal (large batch)"),
    ("SGD_b8",        "SGD",          {},            8,    0.01,  4000, "marginal (small batch)"),
    ("SGDM0.9_b8",    "SGD-Momentum", {"beta": 0.9}, 8,    0.002, 4000, "metastable"),
    ("SGDM0.9_b2048", "SGD-Momentum", {"beta": 0.9}, 2048, 0.035, 4000, "marginal (large batch)"),
]


def main():
    results = []
    for name, optn, params, batch, lr, steps, expect in ARCHETYPES:
        results.append(run_archetype(name, optn, params, batch, lr, steps, expect))
        json.dump(results, open(os.path.join(OUT, "archetype_gamma.json"), "w"), indent=2)
    print("\n===== VERDICT (gamma per coordinate) =====")
    print(f"{'archetype':16s}{'expect':22s}{'fixed':>9s}{'track':>9s}{'subK':>9s}{'full':>9s}")
    for r in results:
        if r.get("diverged"):
            print(f"{r['name']:16s}{r['expect']:22s}   diverged"); continue
        print(f"{r['name']:16s}{r['expect']:22s}{r['gamma_fixed_mean']:+9.4f}{r['gamma_track_mean']:+9.4f}"
              f"{r['gamma_subK_mean']:+9.4f}{r['gamma_full_mean']:+9.4f}")
    print("\n track separates marginal(~0) from metastable(<0) while fixed does not => gamma_proj")
    print(" concept vindicated, fix = track u. full>0 everywhere => coupled bulk is chaotic off-u.")


if __name__ == "__main__":
    main()
