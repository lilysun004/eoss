"""
P1 -- iso-R contours (designed cells): the falsification test of R's ratio FORM, per P1_DESIGN.md.
Cells at 3-4 (beta, B) combos per target R (held fixed via measured rotation), at validated-live lrs.
Dense logging (existing slow_sweep runner) gives position (kappa/gbs/lam_full), both R axes
(cos_uu -> R_traj; R_noise sparse), and energy-weighted coupling primitives (cos_su, step_norm) for
the OUT-OF-SAMPLE mediation replication (these cells did not participate in mediator selection).
Pre-registered readout in P1_DESIGN.md: var(position) along-iso-R / across < 0.3 (+ coupling-constant
companion). NOTE split-half u_B logging is NOT added here (would need runner surgery); lam_full is the
decorrelated position available, so the circularity check runs on lam_full out-of-sample.
"""
import os, sys, json, glob, subprocess, time
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.slow_sweep_driver as D
OUT = os.path.join(_REPO, "results", "p1_isoR"); os.makedirs(OUT, exist_ok=True)


def rot_rate_per_batch():
    """median (1 - cos_uu) per batch from the clean sweep cells = noise-driven rotation rate (beta-indep)."""
    rr = {}
    tmp = {}
    for f in glob.glob(os.path.join(_REPO, "results/slow_sweep/*/meta.json")):
        m = json.load(open(f))
        if m["tag"].startswith(("DEPTH", "SMOKE", "V_", "DV")) or m.get("diverged") or m.get("status") != "done":
            continue
        try:
            z = np.load(f.replace("meta.json", "dense.npz")); step = z["step"]; n = len(step)
            if n < 200:
                continue
            pl = step >= step[int(n * 0.6)]
            cuu = np.median(z["cos_uu"][pl][np.isfinite(z["cos_uu"][pl])])
            if 0 < cuu < 1:
                tmp.setdefault(m["batch"], []).append(1 - cuu)
        except Exception:
            pass
    for b, v in tmp.items():
        rr[b] = float(np.median(v))
    return rr


def build_contours():
    rr = rot_rate_per_batch()
    print(f"[p1] measured rotation-rate (1-cos_uu) per batch: { {b: round(v,3) for b,v in sorted(rr.items())} }", flush=True)
    betas = [0.3, 0.6, 0.9, 0.95, 0.99]
    combos = []
    for Rt in (2.0, 8.0):
        hit = []
        for b in sorted(rr):
            for be in betas:
                R = (1 / (1 - be)) * rr[b]
                if abs(np.log(R / Rt)) < 0.30:               # within ~35% of target R
                    lr = D.PER_BATCH_LR[b][len(D.PER_BATCH_LR[b]) - 2]  # canonical live lr
                    hit.append((b, be, lr, R))
        # keep up to 4 combos spanning batches
        hit = sorted(hit, key=lambda x: x[0])[:4]
        for (b, be, lr, R) in hit:
            combos.append(dict(Rt=Rt, batch=b, beta=be, lr=lr, R=R))
        print(f"[p1] contour R~{Rt}: {[(b, be, round(R,1)) for (b,be,lr,R) in hit]}", flush=True)
    return combos


def main():
    combos = build_contours()
    cells = []
    for c in combos:
        for s in range(2):
            cells.append((f"isoR{c['Rt']:.0f}_b{c['batch']}_beta{c['beta']}_s{s}",
                          c["batch"], c["beta"], c["lr"], s))
    print(f"[p1] {len(cells)} cells to run", flush=True)
    procs = {}; it = iter(cells)
    def launch(cell):
        tag, B, beta, lr, s = cell
        cmd = [sys.executable, "-m", "experiments.slow_sweep", "--tag", tag, "--optn", "SGD-Momentum",
               "--beta", str(beta), "--batch", str(B), "--lr", str(lr), "--seed", str(s),
               "--catapult_target", "20", "--max_steps", "30000", "--warmup", "5000",
               "--stride", str(D.STRIDE[B]), "--out_dir", OUT]
        p = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, tag + ".log"), "w"), stderr=subprocess.STDOUT)
        procs[p] = tag; print(f"  launch {tag} pid={p.pid}", flush=True)
    for _ in range(3):
        try: launch(next(it))
        except StopIteration: break
    while procs:
        time.sleep(5)
        for p in list(procs):
            if p.poll() is not None:
                t = procs.pop(p); print(f"  done {t} rc={p.returncode}", flush=True)
                try: launch(next(it))
                except StopIteration: pass
    print("[p1] all cells complete", flush=True)


if __name__ == "__main__":
    main()
