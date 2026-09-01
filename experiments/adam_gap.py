"""Adam small-batch gap test (3-outcome, see CB_LAW_PREREG.md Adam addendum).
Whitened frame: kB = lr*lam_batch (already whitened for Adam in slow_sweep), kfull_w = lr*lam_full_w.
Usage: python -m experiments.adam_gap --auto | --assemble | --status"""
import os, sys, json, time, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.cb_sweep as CB
import experiments.kspec_ladder as KL

CELLS = [("adam_b8",  "Adam", 0.90,  8, 0.001, 30000, 8000, 3000),
         ("adam_b16", "Adam", 0.90, 16, 0.001, 30000, 8000, 3000),
         ("adam_b32", "Adam", 0.90, 32, 0.001, 30000, 8000, 3000)]
MD = os.path.join(_REPO, "ADAM_GAP_RESULTS.md")

def run_auto(conc):
    KL.preflight(CELLS)
    pj = json.load(open(KL.PRE_JSON))
    lrs = {n: (v.get("lr") if isinstance(v, dict) else v) for n, v in pj.items()}
    todo = [(n, o, b, bb, lrs.get(n), ms, u0) for (n, o, b, bb, _l, ms, u0, _p) in CELLS
            if lrs.get(n) and not CB.is_done(CB.tag_of(n))]
    print(f"[run] {len(todo)} cells (conc {conc})", flush=True)
    procs = []
    while todo or procs:
        procs = [p for p in procs if p.poll() is None]
        while todo and len(procs) < conc:
            n, o, b, bb, lr, ms, u0 = todo.pop(0)
            print(f"[run] launch {CB.tag_of(n)} lr={lr}", flush=True)
            procs.append(CB.launch(n, o, b, bb, lr, ms, u0))
        time.sleep(20)
    print("[run] all cells complete", flush=True)

def assemble():
    import datetime
    L = [f"# ADAM_GAP_RESULTS.md — Adam whitened-frame gap test ({datetime.date.today()})\n",
         "Registered 3-outcome test (CB_LAW_PREREG.md Adam addendum, committed before these runs). DATA ONLY.\n",
         "| cell | b | lr | κ̃_B | κ̃_full (whitened) | **gap_w** | κ_full raw | drift | GBS | flags |", "|" + "---|"*10]
    for (n, optn, beta, batch, _l, _ms, _u0, _p) in CELLS:
        d = os.path.join(CB.OUT, CB.tag_of(n))
        if not os.path.exists(os.path.join(d, "dense.npz")):
            L.append(f"| {CB.tag_of(n)} | | | | | | | | | CENSORED: no run |"); continue
        z = np.load(os.path.join(d, "dense.npz")); meta = json.load(open(os.path.join(d, "meta.json")))
        lr = meta["lr"]; k = lr * z["lam_batch"]; dx = z["dxu"] / z["su"]
        ok = np.isfinite(k) & (np.abs(dx - 1) <= 0.05); idx = np.where(ok)[0]; h = idx[len(idx)//2:]
        kB = float(np.nanmedian(k[h])); k1 = float(np.nanmedian(k[idx[len(idx)//4:len(idx)//2]]))
        ls = z["lf_step"]; fm = (ls >= z["step"][idx[len(idx)//2]])
        kfw = float(np.nanmedian(z["lam_full_w"][fm])) if "lam_full_w" in z.files else float("nan")
        kfr = float(np.nanmedian(z["lam_full"][fm & np.isfinite(z["lam_full"])])) if fm.sum() else float("nan")
        fl = [x for x, c in (("diverged", meta.get("diverged")), ("nonstat", abs(kB/k1-1) >= 0.15)) if c]
        L.append(f"| {CB.tag_of(n)} | {batch} | {lr} | {kB:.3f} | {kfw:.3f} | **{kB-kfw:.3f}** | {kfr:.3f} | "
                 f"{kB/k1-1:+.2f} | {float(np.nanmedian(z['gbs'][h])):.2f} | {', '.join(fl)} |")
    open(MD, "w").write("\n".join(L) + "\n")
    print(f"[assemble] -> {MD}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true"); ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    if a.auto: run_auto(a.concurrency); assemble()
    elif a.assemble: assemble()
