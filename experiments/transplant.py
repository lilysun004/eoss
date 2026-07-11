"""
Transplant actuator -- eta-clean DIRECT displacement of the slow variable lambda (only theta moves,
lr and beta are NEVER touched, so no cell can be tipped into divergence/convergence). Used here to
verify the strong claim: is there a metastable PHASE -- a finite REGION where the operating point is
genuinely FORCE-FREE (KKT constraint slack, no restoring) -- inside COMFORTABLY-LIVE cells, not just
the knife-edge deep endpoint near the diverge-or-crawl boundary.

Method: SGD's progressive-sharpening checkpoints (saved DENSELY, ~every 250 steps) form a graded
lambda-ladder at the same (B,lr,data). Transplant each theta into the SGDM optimizer (buffer zeroed;
~1/(1-beta) warm-up excluded), run LONG at the target lr, watch lambda_probe (held-out).

DELIVERABLE = the restoring-RATE curve k(R): k = rate at which lambda relaxes toward the cell's own
plateau (fit -slope of log|lambda(t)-plateau| over the relax). PARK <=> k ~ 0 (lambda stays at the
transplant); RETURN <=> k > 0 (relaxes to plateau). Reported per (cell, source, seed) with:
  - the DRIFT-NULL: the |k| a stationary plateau (pure diffusion) fakes over the same window -- k must
    exceed it to count as real restoring;
  - the SGD-twin's k at the MATCHED source (cancels loss confound; validates the machinery restores
    where restoring is known);
  - a TIMESCALE bound: 1/k in steps vs the SGD twin's, so "parked for N steps" becomes "restoring
    timescale, if any, > M x the SGD twin's".
PHASE = k hits 0 (CI includes 0, excludes SGD-twin's k) at finite R and STAYS there across sources
AND seeds, from ABOVE and BELOW the plateau. CONTINUUM = k decays smoothly toward 0 without reaching
it inside the live region ("asymptotic slackness" -- still coherent, softer sentence).

The BELOW-plateau source is the most discriminating test: KKT-slack predicts park DOWNWARD too; if
lambda instead CLIMBS back to the plateau from below, the position is a genuine attractor (residual
drive), not parked-by-exhaustion -- and the phase picture needs revision. (Below-plateau sources come
from early SGD training -> higher loss, so the loss column matters most there.)

Cells span R inside validated-live windows ONLY (no dial pushed toward the dead region).
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


def train_ckpts(optn, params, batch, lr, steps, ckpt_every=250, probe=512):
    """Train, saving (lambda_full, loss, state_dict) DENSELY along the sharpening path."""
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params); Xp, Yp = X[:probe], Y[:probe]
    g = T.Generator().manual_seed(0); u = None; ck = []
    for s in range(steps):
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        lo = loss_fn(net(Xb).squeeze(-1), Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            break
        opt.zero_grad(); lo.backward(); opt.step()
        # dense early (every 15 steps, first 600) so the sub-plateau ramp is captured -> a genuine
        # BELOW-plateau source exists (lambda climbs past the SGDM plateau within ~50 steps); coarser after.
        every = 15 if s < 600 else ckpt_every
        if (s + 1) % every == 0:
            lam, u = lam_probe(net, loss_fn, Xp, Yp, u)
            with T.no_grad():
                ls = float(loss_fn(net(Xp).squeeze(-1), Yp))
            ck.append(dict(step=s + 1, lam=lam, loss=ls, sd=copy.deepcopy(net.state_dict())))
    return ck, loss_fn, (X, Y)


def transplant_relax(source_sd, optn, params, batch, lr, XY, loss_fn, N, seed, probe=512):
    X, Y = XY; net, _ = L.build(); net.load_state_dict(copy.deepcopy(source_sd))
    opt = create_optimizer(optn, net, lr, params); Xp, Yp = X[:probe], Y[:probe]
    g = T.Generator().manual_seed(seed); u = None; lam, loss = [], []
    for _ in range(N):
        lv, u = lam_probe(net, loss_fn, Xp, Yp, u)
        with T.no_grad():
            ls = float(loss_fn(net(Xp).squeeze(-1), Yp))
        lam.append(lv); loss.append(ls)
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        lo = loss_fn(net(Xb).squeeze(-1), Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            lam += [np.nan] * (N - len(lam)); loss += [np.nan] * (N - len(loss)); break
        opt.zero_grad(); lo.backward(); opt.step()
    return np.array(lam), np.array(loss)


def fit_k(lam, plateau, warm):
    """restoring rate toward plateau: -slope of log|lambda - plateau| vs t over the relax. k~0 = park."""
    a = lam[warm:]; ok = np.isfinite(a)
    a = a[ok]
    if len(a) < 30 or not np.all(np.isfinite(lam)):
        return np.nan, np.nan          # (k, settled); nan k => catapult
    d = np.abs(a - plateau) + 1e-9; t = np.arange(len(a))
    slope = np.polyfit(t, np.log(d), 1)[0]
    return float(-slope), float(np.median(a[-len(a) // 3:]))


def run_cell(name, batch, lr, sgdm_params, R, N=4000, seeds=2):
    print(f"\n=== {name} (b{batch} lr{lr}, SGDM {sgdm_params}, R~{R}) ===", flush=True)
    beta = sgdm_params["beta"]; warm = int(3 / (1 - beta))
    # lambda ramps FAST through the interior -> checkpoint densely (every 60 steps) for fine lambda coverage
    sgd_ck, loss_fn, XY = train_ckpts("SGD", {}, batch, lr, 6000, ckpt_every=60)
    sgdm_ck, _, _ = train_ckpts("SGD-Momentum", sgdm_params, batch, lr, 4000, ckpt_every=250)
    plat = float(np.median([c["lam"] for c in sgdm_ck[-6:]]))
    # DRIFT-NULL: proper null-displacement baseline -- transplant SGDM's OWN final theta back into
    # SGDM and run N steps; lambda should stay at plateau, so |k| here is the diffusion floor.
    lam_null, _ = transplant_relax(sgdm_ck[-1]["sd"], "SGD-Momentum", sgdm_params, batch, lr, XY, loss_fn, N, 7)
    drift_null = abs(fit_k(lam_null, plat, warm)[0])
    lams = np.array([c["lam"] for c in sgd_ck])
    fracs = [0.85, 1.0, 1.1, 1.2, 1.3, 1.5]     # below, null, interior x3, above-wall
    res = dict(name=name, batch=batch, lr=lr, beta=beta, R=R, plateau=plat, drift_null=drift_null,
               warm=warm, sources=[])
    for fr in fracs:
        i = int(np.argmin(np.abs(lams - fr * plat))); src = sgd_ck[i]
        kk_sgdm, kk_sgd = [], []
        for sd in range(seeds):
            lam_m, loss_m = transplant_relax(src["sd"], "SGD-Momentum", sgdm_params, batch, lr, XY, loss_fn, N, 7 + sd)
            lam_c, _ = transplant_relax(src["sd"], "SGD", {}, batch, lr, XY, loss_fn, N, 7 + sd)
            k_m, set_m = fit_k(lam_m, plat, warm); k_c, set_c = fit_k(lam_c, plat, warm)
            kk_sgdm.append(k_m); kk_sgd.append(k_c)
            if sd == 0:
                np.savez(os.path.join(OUT, f"trace_{name}_f{fr}.npz"), lam_sgdm=lam_m, loss_sgdm=loss_m,
                         lam_sgd=lam_c, source_lam=src["lam"], plateau=plat)
        rec = dict(frac=fr, src_lam=src["lam"], src_loss=src["loss"], settled_sgdm=set_m, settled_sgd=set_c,
                   k_sgdm=float(np.nanmean(kk_sgdm)), k_sgdm_std=float(np.nanstd(kk_sgdm)),
                   k_sgd=float(np.nanmean(kk_sgd)))
        res["sources"].append(rec)
        tag = "PARK" if (np.isfinite(rec["k_sgdm"]) and rec["k_sgdm"] < 2 * drift_null) else \
              ("CATAPULT" if not np.isfinite(rec["k_sgdm"]) else "RETURN")
        print(f"  src {src['lam']:6.1f} ({fr:.2f}x, loss {src['loss']:.3f}) -> SGDM k={rec['k_sgdm']:.5f}"
              f"+-{rec['k_sgdm_std']:.5f} [{tag}] settle {set_m:6.1f} (plat {plat:.1f}, null {drift_null:.5f}) "
              f"| SGD-ctl k={rec['k_sgd']:.5f} settle {set_c:6.1f}", flush=True)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    return res


# validated-live cells spanning R (NO dial pushed toward the dead region)
CELLS = [
    ("b8_b0.9_R9",   8,   0.002, {"beta": 0.9}, 9),    # R~9, decisive (different batch, same R as b32)
    ("b32_b0.9_R9",  32,  0.005, {"beta": 0.9}, 9),    # R~9, deep endpoint with dense sources now
    ("b128_b0.9_R3", 128, 0.006, {"beta": 0.9}, 3),    # R~3, mid
    ("b8_b0.6_R2",   8,   0.004, {"beta": 0.6}, 2),    # R~2 anchor (expected to restore)
    ("b32_b0.6_R2",  32,  0.005, {"beta": 0.6}, 2),    # R~2
]


def main():
    allres = [run_cell(*c) for c in CELLS]
    print("\n===== k(R) VERDICT (restoring rate vs R; PARK = k~drift-null, RETURN = k>>null) =====")
    print(f"{'cell':16s}{'R':>3}{'plateau':>8}{'null':>8} | per-source SGDM k (frac: k)")
    for r in allres:
        ks = " ".join(f"{s['frac']}:{s['k_sgdm']:.4f}" for s in r["sources"])
        print(f"{r['name']:16s}{r['R']:>3}{r['plateau']:8.1f}{r['drift_null']:8.4f} | {ks}")
    print("\n PHASE: SGDM k ~ drift_null (PARK) across sources+seeds, from above AND below plateau, at")
    print("   finite R, while SGD-ctl k >> null. CONTINUUM: k decays smoothly toward null without reaching.")
    print(" Below-plateau (0.85x) is decisive: park-down => slack; climb-up => attractor (revise phase).")


if __name__ == "__main__":
    main()
