"""
CB sweep: confirmation cells for the noise-gap edge law (analysis/CB_LAW_PREREG.md).
kappa_B = kappa_full + 2(1-beta) at small-batch memory-edge plateaus.
Usage: python -m experiments.cb_sweep --auto [--concurrency 3] | --status | --assemble
Results: results/kspec_cb/CB_*; tables kspec_results/cb/; doc CB_RESULTS.md (data only).
"""
import os, sys, json, time, argparse, subprocess
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.kspec_ladder as KL
import experiments.gold_sweep as G

OUT = os.path.join(_REPO, "results", "kspec_cb")
PRE = os.path.join(OUT, "preflight"); os.makedirs(PRE, exist_ok=True)
KL.PRE, KL.PRE_JSON = PRE, os.path.join(OUT, "preflight.json")
RES = os.path.join(_REPO, "kspec_results", "cb"); os.makedirs(RES, exist_ok=True)
MD = os.path.join(_REPO, "CB_RESULTS.md")

#        name          optn            beta  batch  lr0     max_steps u0_at probe
CELLS = [("hb03_b8",    "SGD-Momentum", 0.30,    8, 0.008,  30000, 8000, 3000),
         ("hb08_b8",    "SGD-Momentum", 0.80,    8, 0.003,  30000, 8000, 3000),
         ("hb095_b8",   "SGD-Momentum", 0.95,    8, 0.001,  30000, 8000, 3000),
         ("sgd_b16",    "SGD",          0.00,   16, 0.010,  30000, 8000, 3000),
         ("hb05_b16",   "SGD-Momentum", 0.50,   16, 0.006,  30000, 8000, 3000),
         ("hb09_b16",   "SGD-Momentum", 0.90,   16, 0.003,  30000, 8000, 3000),
         ("hb097_b16",  "SGD-Momentum", 0.97,   16, 0.0008, 30000, 8000, 3000),
         ("nest_b16",   "SGD-Nesterov", 0.90,   16, 0.003,  30000, 8000, 3000),
         ("muon095_b16","Muon",         0.95,   16, 0.001,  30000, 8000, 3000),
         ("muon09_b16", "Muon",         0.90,   16, 0.001,  30000, 8000, 3000)]

def tag_of(n): return f"CB_{n}_s0"

def is_done(tag):
    mp = os.path.join(OUT, tag, "meta.json")
    if not os.path.exists(mp): return False
    m = json.load(open(mp))
    return m.get("status") in ("done", "diverged")

def launch(name, optn, beta, batch, lr, max_steps, u0_at):
    tag = tag_of(name)
    cmd = [sys.executable, "-m", "experiments.slow_sweep", "--tag", tag, "--optn", optn,
           "--beta", str(beta), "--batch", str(batch), "--lr", str(lr), "--out_dir", OUT,
           "--catapult_target", str(10**9), "--max_steps", str(max_steps),
           "--warmup", str(10**9), "--stride", "1", "--u0_at", str(u0_at), "--seed", "0"]
    lf = open(os.path.join(OUT, tag + ".log"), "a")
    return subprocess.Popen(cmd, stdout=lf, stderr=lf)

def run_auto(conc):
    lrs = KL.preflight(CELLS)  # bisects, resume-safe, writes preflight.json
    todo = [(n, o, b, bb, lrs.get(n), ms, u0) for (n, o, b, bb, _l, ms, u0, _p) in CELLS
            if lrs.get(n) and not is_done(tag_of(n))]
    print(f"[run] {len(todo)} cells (conc {conc})", flush=True)
    procs = []
    while todo or procs:
        procs = [p for p in procs if p.poll() is None]
        while todo and len(procs) < conc:
            n, o, b, bb, lr, ms, u0 = todo.pop(0)
            print(f"[run] launch {tag_of(n)} lr={lr}", flush=True)
            procs.append(launch(n, o, b, bb, lr, ms, u0))
        time.sleep(20)
    print("[run] all cells complete", flush=True)

def assemble():
    import csv, datetime
    rows = []
    for (n, optn, beta, batch, _l, _ms, _u0, _p) in CELLS:
        d = os.path.join(OUT, tag_of(n))
        if not os.path.exists(os.path.join(d, "dense.npz")):
            rows.append(dict(cell=tag_of(n), ok=False, why="no run")); continue
        try:
            r = G.analyze(d)  # kspec + health-masked GBS etc.
            z = np.load(os.path.join(d, "dense.npz")); meta = json.load(open(os.path.join(d, "meta.json")))
            lr = meta["lr"]; k = lr * z["lam_batch"]
            dx = z["dxu"] / z["su"]; ok = np.isfinite(k) & (np.abs(dx - 1) <= 0.05)
            idx = np.where(ok)[0]; h = idx[len(idx)//2:]
            kB = float(np.nanmedian(k[h]))
            k1 = float(np.nanmedian(k[idx[len(idx)//4:len(idx)//2]]))
            lf, ls = z["lam_full"], z["lf_step"]
            fm = (ls >= z["step"][idx[len(idx)//2]]) & np.isfinite(lf)
            kfull = float(np.median(lf[fm])) if fm.sum() >= 3 else float("nan")
            mem = 1/(1-beta) if beta < 1 and beta > 0 else 1.0
            r.update(kB_late=kB, kfull=kfull, gap=kB-kfull, gap_pred=2*(1-beta) if beta > 0 else float("nan"),
                     gap_x_mem=(kB-kfull)*mem, drift_late=kB/k1-1 if k1 else float("nan"), mem=mem)
        except Exception as e:
            r = dict(cell=tag_of(n), ok=False, why=repr(e))
        rows.append(r)
    cols = ["cell","optn","beta","batch","lr","kB_late","kfull","gap","gap_pred","gap_x_mem","mem",
            "drift_late","gbs_med","kappa_spec","r1_dxu","death_step","healthy_frac","stationary","diverged","ok","why"]
    with open(os.path.join(RES, "cb_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in cols})
        json.dump(rows, open(os.path.join(RES, "cb_rows.json"), "w"), indent=1, default=str)
    L = [f"# CB_RESULTS.md — noise-gap edge law confirmation cells ({datetime.date.today()})\n",
         "Registered predictions: `analysis/CB_LAW_PREREG.md` (committed before these runs). DATA ONLY — no verdict here.\n",
         "| cell | β/mom | b | lr | κ_B | κ_full | gap | predicted 2(1−β) | **gap×mem** | drift | GBS | flags |", "|" + "---|"*12]
    for r in rows:
        if not r.get("ok"):
            L.append(f"| {r['cell']} | | | | | | | | | | | CENSORED: {r.get('why')} |"); continue
        fl = [x for x, c in (("diverged", r.get("diverged")), ("nonstat", abs(r.get("drift_late", 0)) >= 0.15),
              ("dead@" + str(r.get("death_step")), r.get("death_step"))) if c]
        g = lambda k, nd=3: (f"{r[k]:.{nd}f}" if isinstance(r.get(k), float) and np.isfinite(r[k]) else "n/a")
        L.append(f"| {r['cell']} | {r.get('beta')} | {r.get('batch')} | {r.get('lr')} | {g('kB_late')} | {g('kfull')} | "
                 f"{g('gap')} | {g('gap_pred')} | **{g('gap_x_mem',2)}** | {g('drift_late',2)} | {g('gbs_med',2)} | {', '.join(fl)} |")
    open(MD, "w").write("\n".join(L) + "\n")
    print(f"[assemble] {len(rows)} cells -> {MD}", flush=True)

def status():
    for (n, *_r) in [(c[0],) for c in CELLS]:
        t = tag_of(n); mp = os.path.join(OUT, t, "meta.json")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        print(f"{t:22s} {m.get('status','—'):9s} steps={m.get('steps','—')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true"); ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--status", action="store_true"); ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    if a.auto: run_auto(a.concurrency); assemble()
    elif a.assemble: assemble()
    elif a.status: status()
