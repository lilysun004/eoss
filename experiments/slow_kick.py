"""
Slow-variable regime instrument: perturb the OPERATING POINT (sharpness kappa), not a fast
coordinate, and watch it relax. This is the regime-defining measurement -- the marginal/
metastable distinction is a property of the feedback loop on the slow variable, and the fast
iterate-kick provably can't see it at small batch (archetype_gamma: SGD b8 and SGDM b8 both
gamma~0 in every projection). No rotating coordinate anywhere here: kappa is a scalar.

Design, per archetype (dense kappa_t = lr * lambda_max(subset H) logged every step):
  (A) BASELINE: M steps at the plateau, lr fixed. Gives kappa0 and -- crucially -- the PASSIVE
      restoring statistic S3 on the RIGHT variable: regress d kappa_{t+1} on (kappa_t - kappa0)
      from natural fluctuations. Slope = passive restoring rate (no intervention).
  (B) SLOW KICK up & down: pulse lr *= (1 +/- eps) for P steps (displaces kappa via extra/less
      instability), then RESTORE lr and fit the relaxation of (kappa - kappa0) back:
      d(dkappa)/dt = -gamma_slow * dkappa. Report gamma_up, gamma_down and their ASYMMETRY.
Predictions:
  marginal   : kappa is a thermostat set-point -> strong restoring (short relax), and ASYMMETRIC
               (shaving from above fast via instability; sharpening from below slow near interp).
  metastable : kappa not pinned -> weak/near-zero restoring, roughly symmetric re-park.
Acceptance: the slow kick (causal) must agree in sign/character with the passive S3 (correlational)
on the same run. Agreement => passive suite validated => sweep re-analysis is just plotting.
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
from utils.measure import compute_eigenvalues, EigenvectorCache

T.set_num_threads(4)
OUT = os.path.join(_REPO, "results", "slow_kick")
os.makedirs(OUT, exist_ok=True)


def lam_subset(net, loss_fn, X, Y, cache, cap=2048):
    Xs, Ys = X[:cap], Y[:cap]
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    return float(compute_eigenvalues(lo, net, k=1, max_iterations=40, reltol=0.01,
                                     eigenvector_cache=cache, return_eigenvectors=False,
                                     use_power_iteration=False))


def run_segment(state, lr_mults, seg_lens, seed):
    """Run consecutive segments with given lr multipliers; log kappa=lr*lambda each step."""
    net, opt, loss_fn = state["net"], state["opt"], state["loss_fn"]
    X, Y = state["XY"]; batch = state["batch"]; base_lr = state["lr"]
    net.load_state_dict(copy.deepcopy(state["net_sd"])); PR.set_optimizer_state(opt, state["opt_sd"])
    cache = EigenvectorCache(1); g = T.Generator().manual_seed(seed)
    kappa, seg_id = [], []
    for si, (mult, n) in enumerate(zip(lr_mults, seg_lens)):
        opt.inner.param_groups[0]["lr"] = base_lr * mult
        for _ in range(n):
            lam = lam_subset(net, loss_fn, X, Y, cache)
            kappa.append(base_lr * mult * lam); seg_id.append(si)   # kappa at the ACTIVE lr
            idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            if not np.isfinite(lo.item()) or lo.item() > 1e6:
                opt.inner.param_groups[0]["lr"] = base_lr
                return np.array(kappa), np.array(seg_id), True
            opt.zero_grad(); lo.backward(); opt.step()
    opt.inner.param_groups[0]["lr"] = base_lr
    return np.array(kappa), np.array(seg_id), False


def fit_relax_rate(dk):
    """fit d(dk)/dt = -gamma*dk over a relaxation window; gamma>0 = restoring."""
    dk = dk[np.isfinite(dk)]
    if len(dk) < 8:
        return np.nan
    a = float(np.dot(dk[:-1], dk[1:]) / (np.dot(dk[:-1], dk[:-1]) + 1e-30))  # dk_{t+1}=a dk_t
    if a <= 0 or a >= 1.5:
        return np.nan
    return float(1.0 - a)   # restoring rate per step (>0 mean-reverting)


def passive_restore(kappa_base):
    """S3 on kappa: regress dkappa_{t+1} on (kappa - mean). slope<0 = restoring; return -slope."""
    k = kappa_base[np.isfinite(kappa_base)]
    if len(k) < 20:
        return np.nan, np.nan
    x = k[:-1] - k.mean(); dy = np.diff(k)
    slope = float(np.dot(x, dy) / (np.dot(x, x) + 1e-30))
    return -slope, float(k.mean())


def run_archetype(name, optn, params, batch, lr, steps, expect, M=250, P=60, Rlx=250, eps=0.35):
    print(f"\n=== {name}: {optn} {params} b{batch} lr{lr} (expect {expect}) ===", flush=True)
    st = PR.train_plateau(optn, params, batch, lr, steps)
    if st is None:
        print("  diverged in training"); return dict(name=name, diverged=True, expect=expect)
    # (A) baseline natural fluctuations -> passive S3 on kappa
    kb, _, bad = run_segment(st, [1.0], [M], seed=11)
    p_rate, kappa0 = passive_restore(kb)
    # (B) slow kicks: [baseline, pulse, relax]
    res = dict(name=name, optn=optn, params=params, batch=batch, lr=lr, expect=expect,
               diverged=False, kappa0=kappa0, passive_restore_rate=p_rate)
    for tag, mult in [("up", 1 + eps), ("down", 1 - eps)]:
        kk, seg, bad = run_segment(st, [1.0, mult, 1.0], [80, P, Rlx], seed=23)
        relax = kk[seg == 2] - kappa0            # after lr restored
        res[f"gamma_{tag}"] = fit_relax_rate(relax)
        res[f"disp_{tag}"] = float(np.nanmedian(kk[seg == 1]) - kappa0)   # displacement during pulse
        np.savez(os.path.join(OUT, f"trace_{name}_{tag}.npz"), kappa=kk, seg=seg, kappa0=kappa0)
    np.savez(os.path.join(OUT, f"baseline_{name}.npz"), kappa=kb, kappa0=kappa0)
    res["asymmetry"] = float((res.get("gamma_up") or np.nan) - (res.get("gamma_down") or np.nan))
    print(f"  kappa0={kappa0:.3f}  passive_restore={p_rate:+.4f}  "
          f"gamma_up={res.get('gamma_up'):+.4f} gamma_down={res.get('gamma_down'):+.4f} "
          f"asym={res['asymmetry']:+.4f}", flush=True)
    return res


ARCHETYPES = [
    ("SGD_b2048",  "SGD",          {},            2048, 0.02,  4000, "marginal-large"),
    ("SGD_b8",     "SGD",          {},            8,    0.01,  4000, "marginal-small"),
    ("SGDM0.9_b8", "SGD-Momentum", {"beta": 0.9}, 8,    0.002, 4000, "metastable"),
]


def main():
    results = []
    for name, optn, params, batch, lr, steps, expect in ARCHETYPES:
        results.append(run_archetype(name, optn, params, batch, lr, steps, expect))
        json.dump(results, open(os.path.join(OUT, "slow_kick.json"), "w"), indent=2, default=str)
    print("\n===== SLOW-VARIABLE VERDICT =====")
    print(f"{'archetype':14s}{'expect':16s}{'kappa0':>8s}{'passiveS3':>11s}{'g_up':>9s}{'g_down':>9s}{'asym':>9s}")
    for r in results:
        if r.get("diverged"):
            print(f"{r['name']:14s}{r['expect']:16s}  diverged"); continue
        print(f"{r['name']:14s}{r['expect']:16s}{r['kappa0']:8.3f}{r['passive_restore_rate']:11.4f}"
              f"{r.get('gamma_up',float('nan')):9.4f}{r.get('gamma_down',float('nan')):9.4f}{r['asymmetry']:9.4f}")
    print("\n marginal: strong restoring (g_up,g_down>0), asymmetric; passiveS3 agrees in sign.")
    print(" metastable: weak restoring (g~0), symmetric. If SGD_b8 restores but SGDM_b8 doesn't,")
    print(" the regimes separate on the SLOW variable where the fast kick could not.")


if __name__ == "__main__":
    main()
