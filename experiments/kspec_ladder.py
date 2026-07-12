"""
kappa_spec interpolation-ladder driver (KSPEC_DESIGN.md): SGD-Momentum beta0.9 b8->b2048 +
beta0.99 b8 (100x-gain cell), 2 seeds each, STRIDE 1 EVERYWHERE (stride 2 would alias the omega=pi
period-2 signal to DC -- the whole phase signal), u0 frozen at plateau start (fixed-frame check).

MANDATORY liveness-bisect pre-flight (standing rule, LESSONS.md): before committing a full cell,
probe the lr with a short run; bisect lr DOWN from canonical until non-diverging AND non-crawling.
Liveness gate (calibrated on the old sweep -- frac_up/loss-bounce does NOT separate live from dead,
batch noise bounces everything):
  live  = probe not diverged
          AND kappa_late >= 0.4 * 2(1-beta)   [not below even the DC-edge floor -> dead/quiescent]
          AND step_norm log-slope > -0.02/step [not in freefall collapse to a converged point]
NOTE the DC floor uses the plateau law -- allowed here (driver/gate side); the kappa_spec ESTIMATOR
(kspec_estimator.py) is the grep-certified formula-free code, not this driver.

Usage:
  python -m experiments.kspec_ladder --preflight        # bisect lrs (resume-safe, serial, ~30 min)
  python -m experiments.kspec_ladder --run              # launch 12 cells at validated lrs, conc 3
  python -m experiments.kspec_ladder --status
Nesterov trio (after ladder): --trio {preflight,run} uses SGD-Nesterov on b8/b128/b2048 beta0.9.
"""
import os, sys, json, time, argparse, subprocess
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec")
PRE = os.path.join(OUT, "preflight")
os.makedirs(PRE, exist_ok=True)
PRE_JSON = os.path.join(OUT, "preflight.json")

# canonical lrs = validated-live from the old slow_sweep (beta0.9); beta0.99 b8 canonical 2e-4
# (DEPTH cells diverged there after 1.5-2.8k steps -> bisect will walk down).
# max_steps sized for spectral record length; u0_at = plateau start (fixed-frame freeze + analysis window).
#          name                optn            beta  batch  lr0     max_steps  u0_at  probe
LADDER = [("b8_beta0.9",      "SGD-Momentum",  0.90,    8, 0.0020,  30000,     8000,  3000),
          ("b32_beta0.9",     "SGD-Momentum",  0.90,   32, 0.0050,  30000,     8000,  3000),
          ("b128_beta0.9",    "SGD-Momentum",  0.90,  128, 0.0060,  20000,     5000,  2000),
          ("b512_beta0.9",    "SGD-Momentum",  0.90,  512, 0.0080,  16000,     4000,  1500),
          ("b2048_beta0.9",   "SGD-Momentum",  0.90, 2048, 0.0065,  16000,     4000,  1500),
          ("b8_beta0.99",     "SGD-Momentum",  0.99,    8, 0.0002,  30000,     8000,  3000)]
TRIO   = [("nest_b8_beta0.9",   "SGD-Nesterov", 0.90,    8, 0.0020,  30000,    8000,  3000),
          ("nest_b128_beta0.9", "SGD-Nesterov", 0.90,  128, 0.0060,  20000,    5000,  2000),
          ("nest_b2048_beta0.9","SGD-Nesterov", 0.90, 2048, 0.0065,  16000,    4000,  1500)]
SEEDS = 2
MAX_PROBES = 6


def probe_once(name, optn, beta, batch, lr, probe_steps):
    """Run one short probe in-process; return (verdict, kappa_late, note)."""
    import experiments.slow_sweep as S
    tag = f"PRE_{name}_lr{lr:.6g}"
    mp = os.path.join(PRE, tag, "meta.json")
    if not (os.path.exists(mp) and json.load(open(mp)).get("status") in ("done", "diverged")):
        S.run_cell(tag, optn, beta, batch, lr, PRE, catapult_target=10**9,
                   max_steps=probe_steps, warmup=10**9, stride=1, seed=0)
    m = json.load(open(mp))
    if m.get("diverged"):
        return "diverged", float("nan"), f"died at measured step {m['steps']}"
    z = np.load(os.path.join(PRE, tag, "dense.npz"))
    n = len(z["kappa"]); w0 = int(n * 0.6)
    kap = float(np.nanmedian(z["kappa"][w0:]))
    sn = np.log(np.maximum(z["step_norm"][w0:], 1e-30))
    slope = float(np.polyfit(np.arange(len(sn)), sn, 1)[0]) if len(sn) > 10 else 0.0
    floor = 0.4 * 2 * (1 - beta)
    if kap < floor:
        return "crawl", kap, f"kappa_late {kap:.4f} < DC floor {floor:.4f}"
    if slope < -0.02:
        return "crawl", kap, f"step_norm freefall slope {slope:.4f}/step"
    return "live", kap, f"kappa_late {kap:.3f}, sn_slope {slope:.5f}"


def preflight(cells):
    res = json.load(open(PRE_JSON)) if os.path.exists(PRE_JSON) else {}
    for (name, optn, beta, batch, lr0, _ms, _u0, probe_steps) in cells:
        if res.get(name, {}).get("lr"):
            print(f"[pre] {name}: already validated lr={res[name]['lr']}", flush=True)
            continue
        lr, hi, lo, hist, accepted = lr0, None, None, [], None
        for it in range(MAX_PROBES):
            t0 = time.time()
            verdict, kap, note = probe_once(name, optn, beta, batch, lr, probe_steps)
            hist.append(dict(lr=lr, verdict=verdict, kappa=None if np.isnan(kap) else kap, note=note))
            print(f"[pre] {name} probe {it+1}: lr={lr:.6g} -> {verdict} ({note}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if verdict == "live":
                accepted = lr; break
            if verdict == "diverged":
                hi = lr; lr = float(np.sqrt(lo * hi)) if lo else lr / 2
            else:
                lo = lr; lr = float(np.sqrt(lo * hi)) if hi else lr * 1.5
        if accepted is None:
            # best effort: hottest non-diverging probe, flagged
            nondiv = [h for h in hist if h["verdict"] != "diverged"]
            accepted = max(nondiv, key=lambda h: h["lr"])["lr"] if nondiv else None
            print(f"[pre] {name}: NO clean live lr in {MAX_PROBES} probes; "
                  f"best-effort {accepted}", flush=True)
        res[name] = dict(lr=accepted, clean=any(h["verdict"] == "live" for h in hist), history=hist)
        json.dump(res, open(PRE_JSON, "w"), indent=1)
    print("[pre] preflight complete:", {k: v["lr"] for k, v in res.items()}, flush=True)


def build_cells(cells):
    res = json.load(open(PRE_JSON))
    out = []
    for (name, optn, beta, batch, _lr0, max_steps, u0_at, _p) in cells:
        lr = res[name]["lr"]
        assert lr, f"{name}: no validated lr"
        for s in range(SEEDS):
            out.append(dict(tag=f"L_{name}_s{s}", optn=optn, beta=beta, batch=batch, lr=lr,
                            seed=s, max_steps=max_steps, u0_at=u0_at))
    return out


def is_done(tag):
    mp = os.path.join(OUT, tag, "meta.json")
    try:
        return json.load(open(mp)).get("status") in ("done", "diverged")
    except Exception:
        return False


def run(cells, concurrency):
    todo = [c for c in build_cells(cells) if not is_done(c["tag"])]
    print(f"[run] {len(todo)} cells (concurrency {concurrency})", flush=True)
    procs = {}; it = iter(todo)
    def launch(c):
        cmd = [sys.executable, "-m", "experiments.slow_sweep", "--tag", c["tag"],
               "--optn", c["optn"], "--beta", str(c["beta"]), "--batch", str(c["batch"]),
               "--lr", str(c["lr"]), "--seed", str(c["seed"]),
               "--catapult_target", str(10**9), "--max_steps", str(c["max_steps"]),
               "--warmup", str(10**9), "--stride", "1", "--u0_at", str(c["u0_at"]),
               "--out_dir", OUT]
        p = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, c["tag"] + ".log"), "w"),
                             stderr=subprocess.STDOUT)
        procs[p] = c; print(f"  launch {c['tag']} lr={c['lr']:.6g} pid={p.pid}", flush=True)
    for _ in range(concurrency):
        try: launch(next(it))
        except StopIteration: break
    while procs:
        time.sleep(10)
        for p in list(procs):
            if p.poll() is not None:
                c = procs.pop(p); print(f"  finished {c['tag']} rc={p.returncode}", flush=True)
                try: launch(next(it))
                except StopIteration: pass
    print("[run] all cells complete", flush=True)


def status(cells):
    for c in build_cells(cells):
        mp = os.path.join(OUT, c["tag"], "meta.json")
        if not os.path.exists(mp):
            print(f"  {c['tag']:28s} —"); continue
        m = json.load(open(mp))
        print(f"  {c['tag']:28s} {m['status']:>8s} div={m.get('diverged')} "
              f"measured={m['steps']}/{c['max_steps']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true"); ap.add_argument("--trio", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    cells = TRIO if a.trio else LADDER
    if a.preflight:
        preflight(cells)
    elif a.run:
        run(cells, a.concurrency)
    elif a.status:
        status(cells)
    else:
        print("specify --preflight / --run / --status (optionally --trio)")
