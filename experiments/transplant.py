"""
Transplant actuator -- the eta-clean, DIRECT displacement of the slow variable lambda, and the test
the phases verdict actually rests on. The lr-pulse (slow_kick.py) is a CONSTRAINT-SIDE actuator: it
displaces lambda by shaving against an ACTIVE constraint, so it works on the marginal cell (clean
F(dlambda), measured wall) but cannot move a SLACK (metastable) lambda -- displacement ~ noise. To
probe the slack INTERIOR we must displace lambda directly, with eta never changing.

METHOD (the SGD twin's own progressive-sharpening trajectory IS a graded lambda-ladder at the same
(B,lr,data)): train SGD saving checkpoints -> lambda climbs gradually -> a natural ladder of theta
sources spanning lambda. Transplant each theta into the SGDM optimizer (buffer ZEROED; exclude the
~1/(1-beta) buffer warm-up transient from the fit), run at the target lr with NO other intervention,
watch lambda_probe (held-out). eta never excursions -> no actuator-coupling artifact.

PRE-REGISTERED READINGS (fixed before running):
  PARK   : lambda stays at the transplanted value (drift <= baseline diffusion) -> slack interior
           confirmed -> the operating point is NOT force-regulated -> PHASES (KKT slack).
  RETURN : lambda relaxes back down to the cell's own plateau lambda* -> there IS a restoring force
           the pulse couldn't see -> regulated operating point -> CONTINUUM (or a 3rd mechanism).
  CATAPULT: lambda/loss takes off -> the transplant landed past the wall -> step DOWN the ladder
           (another wall measurement, not a failure).
Ladder (>=3 source levels) needed for the same reason the amplitude ladder was: one level can't tell
interior-flat from wall.

CONTROLS (no instrument ships without one):
  (a) transplant into SGD too (its own earlier checkpoints): from below its edge it should climb back
      via sharpening (slow near interpolation -- log, don't over-read); from a hotter source above it
      should shave back fast. Validates the machinery shows restoring where restoring is known.
  (b) LOSS confound: an earlier checkpoint carries higher loss AND higher lambda, so lambda-relaxation
      could be loss-relaxation. Mitigate: prefer LATE sources (high lambda, low loss) and LOG LOSS
      through the relax so the two can be separated in analysis.
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

import experiments.long_train_grid as L
from experiments.slow_kick import lam_probe
from utils.optimizer import create_optimizer

T.set_num_threads(int(os.environ.get("EOSS_THREADS", "4")))
OUT = os.path.join(_REPO, "results", "transplant")
os.makedirs(OUT, exist_ok=True)


def train_ckpts(optn, params, batch, lr, steps, n_ckpt=12, probe=512):
    """Train, saving (step, lambda_full, loss, state_dict) checkpoints along the sharpening path."""
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    Xp, Yp = X[:probe], Y[:probe]
    g = T.Generator().manual_seed(0); u = None; every = max(1, steps // n_ckpt); ck = []
    for s in range(steps):
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            break
        opt.zero_grad(); lo.backward(); opt.step()
        if (s + 1) % every == 0:
            lam, u = lam_probe(net, loss_fn, Xp, Yp, u)
            with T.no_grad():
                ls = float(loss_fn(net(Xp).squeeze(-1), Yp))
            ck.append(dict(step=s + 1, lam=lam, loss=ls, sd=copy.deepcopy(net.state_dict())))
    return ck, loss_fn, (X, Y)


def transplant_relax(source_sd, optn, params, batch, lr, XY, loss_fn, N, warm_excl, probe=512):
    """Load source theta into a FRESH `optn` optimizer (buffer zeroed), run N steps, log lambda+loss."""
    X, Y = XY; net, _ = L.build(); net.load_state_dict(copy.deepcopy(source_sd))
    opt = create_optimizer(optn, net, lr, params)
    Xp, Yp = X[:probe], Y[:probe]; g = T.Generator().manual_seed(7); u = None
    lam, loss = [], []
    for _ in range(N):
        lv, u = lam_probe(net, loss_fn, Xp, Yp, u)
        with T.no_grad():
            ls = float(loss_fn(net(Xp).squeeze(-1), Yp))
        lam.append(lv); loss.append(ls)
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        lo = loss_fn(net(Xb).squeeze(-1), Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            lam += [float("nan")] * (N - len(lam)); loss += [float("nan")] * (N - len(loss)); break
        opt.zero_grad(); lo.backward(); opt.step()
    return np.array(lam), np.array(loss)


def classify(lam, source_lam, plateau_lam, sigma, warm_excl):
    a = lam[warm_excl:]; a = a[np.isfinite(a)]
    if len(a) < 20 or not np.all(np.isfinite(lam)):
        return "CATAPULT", float("nan")
    settled = float(np.median(a[-len(a) // 3:]))
    # fraction of the source->plateau gap traversed: 0 = parked at source, 1 = returned to plateau
    gap = source_lam - plateau_lam
    frac = float((source_lam - settled) / gap) if abs(gap) > 1e-9 else float("nan")
    if not np.isfinite(frac):
        return "n/a", settled
    if abs(source_lam - settled) < 3 * sigma:
        return "PARK", settled          # stayed at transplant (slack)
    if frac > 0.5:
        return "RETURN", settled        # relaxed toward own plateau (restoring)
    return "PARTIAL", settled


def run_cell(name, batch, lr, sgdm_params, N=1200, probe=512):
    print(f"\n=== {name} (b{batch} lr{lr}, SGDM {sgdm_params}) ===", flush=True)
    beta = sgdm_params["beta"]; warm = int(3 / (1 - beta))          # ~buffer warm-up transient
    # SGD source ladder + baseline diffusion sigma
    sgd_ck, loss_fn, XY = train_ckpts("SGD", {}, batch, lr, 4000, n_ckpt=12)
    # SGDM's own plateau lambda* (the return target) + its lambda diffusion
    sgdm_ck, _, _ = train_ckpts("SGD-Momentum", sgdm_params, batch, lr, 4000, n_ckpt=8)
    plateau_lam = float(np.median([c["lam"] for c in sgdm_ck[-4:]]))
    sigma = float(np.std([c["lam"] for c in sgdm_ck[-4:]])) + 1e-9
    # pick source levels by TARGET lambda relative to SGDM's plateau: just above (interior) up toward
    # the wall. (Late SGD sources at lam~edge overshoot SGDM's wall -> only catapult; useless for the
    # interior test.) For each target, take the SGD checkpoint whose lam is closest.
    lams = np.array([c["lam"] for c in sgd_ck])
    # INTERIOR-targeted (metastable interior is narrow; 1.2-2x overshot into the wall). Fine sources
    # just above plateau probe the flat slack region; env EOSS_WIDE=1 keeps the old wall-spanning set.
    if os.environ.get("EOSS_WIDE") == "1":
        targets = [1.2 * plateau_lam, 1.5 * plateau_lam, 2.0 * plateau_lam]
    else:
        targets = [1.05 * plateau_lam, 1.12 * plateau_lam, 1.20 * plateau_lam, 1.32 * plateau_lam]
    idxs = sorted(set(int(np.argmin(np.abs(lams - t))) for t in targets))
    res = dict(name=name, batch=batch, lr=lr, beta=beta, plateau_lam=plateau_lam, sigma=sigma,
               warm=warm, sources=[])
    for i in idxs:
        src = sgd_ck[i]
        lam_t, loss_t = transplant_relax(src["sd"], "SGD-Momentum", sgdm_params, batch, lr, XY, loss_fn, N, warm)
        verdict, settled = classify(lam_t, src["lam"], plateau_lam, sigma, warm)
        # control: same source into SGD (should relax toward SGD's own edge, i.e. restore)
        lam_c, loss_c = transplant_relax(src["sd"], "SGD", {}, batch, lr, XY, loss_fn, N, warm)
        ctl, settled_c = classify(lam_c, src["lam"], float(np.median([c["lam"] for c in sgd_ck[-4:]])), sigma, warm)
        rec = dict(src_lam=src["lam"], src_loss=src["loss"], settled_sgdm=settled, verdict=verdict,
                   settled_sgd_ctl=settled_c, ctl_verdict=ctl,
                   loss_start=float(loss_t[warm]) if len(loss_t) > warm else float("nan"),
                   loss_end=float(np.nanmedian(loss_t[-N // 3:])))
        res["sources"].append(rec)
        np.savez(os.path.join(OUT, f"trace_{name}_src{src['lam']:.0f}.npz"),
                 lam_sgdm=lam_t, loss_sgdm=loss_t, lam_sgd=lam_c, source_lam=src["lam"], plateau_lam=plateau_lam)
        print(f"  source lam={src['lam']:6.1f} (loss={src['loss']:.4f}) -> SGDM settles {settled:6.1f} "
              f"[{verdict:8s}] (plateau {plateau_lam:.1f}) | SGD-control settles {settled_c:6.1f} [{ctl}]",
              flush=True)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    return res


CELLS = [
    ("b32_b0.9",  32,  0.005, {"beta": 0.9}),    # metastable (plateau lam ~126, sources up to ~500)
    ("b8_b0.6",   8,   0.004, {"beta": 0.6}),    # partial-metastable
    ("b512_b0.9", 512, 0.008, {"beta": 0.9}),    # SGDM marginal-with-memory (control: should NOT park low)
]


def main():
    allres = [run_cell(*c) for c in CELLS]
    print("\n===== TRANSPLANT VERDICT (slack interior test) =====")
    print(f"{'cell':12s}{'plateau':>9}{'src_lam':>9}{'sgdm_settle':>12}{'verdict':>10}{'sgd_ctl':>9}")
    for r in allres:
        for s in r["sources"]:
            print(f"{r['name']:12s}{r['plateau_lam']:9.1f}{s['src_lam']:9.1f}{s['settled_sgdm']:12.1f}"
                  f"{s['verdict']:>10}{s['ctl_verdict']:>9}")
    print("\n PHASES: SGDM PARKs at transplanted lambda (all sources) while SGD-control RETURNs")
    print("   -> slack interior has no restoring force (only the remote wall). Position = order param.")
    print(" CONTINUUM: SGDM RETURNs to its plateau -> a restoring force the pulse couldn't see.")
    print(" (loss logged alongside lambda in traces to separate lambda-relax from loss-relax.)")


if __name__ == "__main__":
    main()
