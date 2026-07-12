"""
Decisive PARK-vs-ATTRACTOR test (settles the metastable-phase question the k(R) run left ambiguous).

The k(R) run used a thin SGD-source ladder (all lambda ~85, above plateau) so "parks at 75" couldn't
tell SLACK (parks wherever put) from an ATTRACTOR at 75. Two diagnostics (2026-07-11) cut against
slack: (a) b8 beta0.6 converges to ONE point (~105) from BOTH sides = attractor; (b) SGDM b8 kappa0
is reproducible across seeds (CV~0.003, as tight as SGD) = not a scattered parking lot. So we need
the clean test: a source ladder spanning the plateau from BELOW and ABOVE, and the readout
  slope = d(settle)/d(source):   slope ~ 1  => PARK (settle tracks source)  => SLACK phase
                                 slope ~ 0  => all converge to one attractor => REGULATED (continuum)

Sources come from SGDM's OWN descent trajectory (init ~85 -> plateau ~64, dipping to ~44), which
gives genuine BELOW-plateau sources (impossible from SGD, whose lambda starts above the plateau).
Transplant each into a FRESH SGDM optimizer, two buffer conditions:
  - zeroed buffer (as k(R) run) ; and
  - PRE-WARMED buffer (~3/(1-beta) tiny-lr steps at the source, so the momentum buffer is spun up
    source-consistently BEFORE the relax) -- rules out the zeroed-buffer transient doing SGD-like
    shaving in the first steps (the "85->75 then stop" concern). If both give the same slope, PARK
    is buffer-transient-robust.
Long relax (N=6000) so settle is a real asymptote, 2 seeds.
"""
import os, sys, copy, json
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
from experiments.slow_kick import lam_probe
from experiments.transplant import train_ckpts
from utils.optimizer import create_optimizer

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "park_test"); os.makedirs(OUT, exist_ok=True)


def relax_from(source_sd, params, batch, lr, XY, loss_fn, N, seed, prewarm_steps=0, probe=512):
    X, Y = XY; net, _ = L.build(); net.load_state_dict(copy.deepcopy(source_sd))
    opt = create_optimizer("SGD-Momentum", net, lr, params); Xp, Yp = X[:probe], Y[:probe]
    g = T.Generator().manual_seed(seed)
    if prewarm_steps:                       # spin up the momentum buffer source-consistently at tiny lr
        opt.inner.param_groups[0]["lr"] = lr * 0.01
        for _ in range(prewarm_steps):
            idx = T.randperm(len(X), generator=g)[:batch]
            lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx]); opt.zero_grad(); lo.backward(); opt.step()
        opt.inner.param_groups[0]["lr"] = lr
    u = None; lam = []
    for _ in range(N):
        lv, u = lam_probe(net, loss_fn, Xp, Yp, u); lam.append(lv)
        idx = T.randperm(len(X), generator=g)[:batch]
        lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            lam += [np.nan] * (N - len(lam)); break
        opt.zero_grad(); lo.backward(); opt.step()
    a = np.array(lam)
    return a, float(np.nanmedian(a[-N // 4:]))     # (series, settle asymptote)


def run(name, batch, lr, beta, N=6000, seeds=2):
    print(f"\n=== {name} (b{batch} lr{lr} beta{beta}) ===", flush=True)
    params = {"beta": beta}; warm = int(3 / (1 - beta))
    sgdm_ck, loss_fn, XY = train_ckpts("SGD-Momentum", params, batch, lr, 5000, ckpt_every=40)  # descent sources
    lams = np.array([c["lam"] for c in sgdm_ck]); plat = float(np.median(lams[-8:]))
    # ladder spanning BELOW and ABOVE plateau, from SGDM's own descent
    targets = [0.72 * plat, 0.85 * plat, 1.0 * plat, 1.25 * plat, 1.55 * plat]
    idxs = sorted(set(int(np.argmin(np.abs(lams - t))) for t in targets))
    srcs, set_zero, set_warm = [], [], []
    for i in idxs:
        src = sgdm_ck[i]; sl = src["lam"]
        sz = np.nanmean([relax_from(src["sd"], params, batch, lr, XY, loss_fn, N, 7 + s)[1] for s in range(seeds)])
        sw = relax_from(src["sd"], params, batch, lr, XY, loss_fn, N, 7, prewarm_steps=warm)[1]
        srcs.append(sl); set_zero.append(sz); set_warm.append(sw)
        print(f"  src {sl:6.1f} ({sl/plat:.2f}x plat) -> settle zeroed={sz:6.1f} prewarm={sw:6.1f}", flush=True)
    srcs = np.array(srcs)
    slope_z = float(np.polyfit(srcs, set_zero, 1)[0]); slope_w = float(np.polyfit(srcs, set_warm, 1)[0])
    verdict = "PARK/SLACK" if slope_z > 0.5 else ("ATTRACTOR/REGULATED" if slope_z < 0.2 else "INTERMEDIATE")
    print(f"  ==> plateau={plat:.1f} | slope(settle vs source) zeroed={slope_z:.2f} prewarm={slope_w:.2f} "
          f"[{verdict}]  (slope~1=park/slack, ~0=attractor)", flush=True)
    json.dump(dict(name=name, plateau=plat, srcs=srcs.tolist(), settle_zero=set_zero, settle_warm=set_warm,
                   slope_zero=slope_z, slope_warm=slope_w, verdict=verdict),
              open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    return slope_z, slope_w


def main():
    for name, b, lr, beta in [("b8_b0.9", 8, 0.002, 0.9), ("b32_b0.9", 32, 0.005, 0.9)]:
        run(name, b, lr, beta)
    print("\n slope~0 across both R~9 cells, both buffer conditions => METASTABLE POSITION IS REGULATED")
    print("   (attractor, not force-free) -> the 'force-free phase' claim is overturned to continuum.")
    print(" slope~1 => genuine PARK/slack, buffer-robust -> phase stands.")


if __name__ == "__main__":
    main()
