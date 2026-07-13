"""
Overnight firming queue 2026-07-13 (user-specified, priority-ordered; standing rules apply:
liveness-bisect for unvalidated cells, raw signed primitives only, resume-safe, NO writes to
SUMMARY/KSPEC_RESULTS tonight — tables only, thinking tomorrow).

  A. seed replication of the flagship at-edge cells (s2-s4): SGDM b512/b2048, nest_b2048,
     nest_b128 at the validated lrs -- the paper's "2.007 +/- CI across N seeds" table.
  B. offline kspec estimator on the new seeds (json tables only, no interpretation).
  (C. Adam trio runs via kspec_ladder --adam after the runner patch; separate launcher.)
  D. onset brackets at remaining ladder + trio cells (margin-vs-noise dataset), via
     ms_cocycle replay checkpoints + ms_bracket.
  E. kappa_ms single-construction recompute (pooled-frame est-i/ii) across valid cells.

Usage: python -m experiments.overnight_0713 --phase A|B|D|E  (each resume-safe)
"""
import os, sys, json, time, argparse, subprocess
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec")

# (base tag, optn, beta, batch, lr, max_steps, u0_at) -- validated lrs from the ladder/trio
FLAGSHIP = [("L_b512_beta0.9",      "SGD-Momentum", 0.9,  512, 0.0080, 16000, 4000),
            ("L_b2048_beta0.9",     "SGD-Momentum", 0.9, 2048, 0.0065, 16000, 4000),
            ("L_nest_b2048_beta0.9","SGD-Nesterov", 0.9, 2048, 0.0065, 16000, 4000),
            ("L_nest_b128_beta0.9", "SGD-Nesterov", 0.9,  128, 0.0060, 20000, 5000)]
NEW_SEEDS = [2, 3, 4]


def is_done(tag):
    try:
        return json.load(open(os.path.join(OUT, tag, "meta.json"))).get("status") in ("done", "diverged")
    except Exception:
        return False


def run_pool(cells, concurrency=3):
    procs = {}; it = iter([c for c in cells if not is_done(c[0])])
    def launch(c):
        tag, optn, beta, batch, lr, ms, u0 , seed = c
        cmd = [sys.executable, "-m", "experiments.slow_sweep", "--tag", tag, "--optn", optn,
               "--beta", str(beta), "--batch", str(batch), "--lr", str(lr), "--seed", str(seed),
               "--catapult_target", str(10**9), "--max_steps", str(ms), "--warmup", str(10**9),
               "--stride", "1", "--u0_at", str(u0), "--out_dir", OUT]
        p = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, tag + ".log"), "w"),
                             stderr=subprocess.STDOUT)
        procs[p] = tag; print(f"  launch {tag} pid={p.pid}", flush=True)
    for _ in range(concurrency):
        try: launch(next(it))
        except StopIteration: break
    while procs:
        time.sleep(10)
        for p in list(procs):
            if p.poll() is not None:
                print(f"  finished {procs.pop(p)} rc={p.returncode}", flush=True)
                try: launch(next(it))
                except StopIteration: pass
    print("[phase] pool complete", flush=True)


def phase_A():
    cells = [(f"{b}_s{s}", o, be, ba, lr, ms, u0, s)
             for (b, o, be, ba, lr, ms, u0) in FLAGSHIP for s in NEW_SEEDS]
    print(f"[A] {len(cells)} seed-replication cells", flush=True)
    run_pool(cells)


def phase_B():
    from experiments.kspec_estimator import analyze_cell
    for (b, *_ ) in FLAGSHIP:
        for s in NEW_SEEDS:
            tag = f"{b}_s{s}"; d = os.path.join(OUT, tag)
            if not os.path.exists(os.path.join(d, "dense.npz")):
                continue
            r = analyze_cell(d)
            json.dump(r, open(os.path.join(OUT, "ms", f"{tag}_kspec.json"), "w"), indent=1)
            print(f"[B] {tag}: kspec={r.get('kappa_spec', float('nan')):.3f}", flush=True)


def phase_D():
    """Onset brackets for remaining cells: b512/b2048 (SGDM), trio; 2 seeds where ckpts exist.
    Fine c near expected walls; ms_cocycle.replay creates missing checkpoints first."""
    import experiments.ms_cocycle as MC
    todo = [("L_b512_beta0.9_s0", [1.05, 1.15, 1.3]), ("L_b512_beta0.9_s1", [1.05, 1.15, 1.3]),
            ("L_b2048_beta0.9_s0", [1.05, 1.15, 1.3]), ("L_b2048_beta0.9_s1", [1.05, 1.15, 1.3]),
            ("L_nest_b2048_beta0.9_s0", [1.05, 1.15, 1.3]), ("L_nest_b128_beta0.9_s0", [1.1, 1.25, 1.4]),
            ("L_nest_b8_beta0.9_s0", [1.1, 1.2, 1.3]),
            ("L_b128_beta0.9_s1", [1.1, 1.2, 1.3]), ("L_b8_beta0.9_s1", [1.15, 1.25, 1.3]),
            ("L_b32_beta0.9_s1", [1.05, 1.15, 1.3]), ("L_b8_beta0.99_s1", [1.2, 1.35, 1.5])]
    from experiments.ms_bracket import run_bracket
    for tag, cs in todo:
        base = "_".join(tag.split("_")[:-1])
        ckp = os.path.join(OUT, "ms", f"{tag}_ckpt.pt")
        if not os.path.exists(ckp):
            if base not in MC.CKPT_STEP:
                print(f"[D] skip {tag} (no ckpt step)"); continue
            MC.replay(tag, MC.CKPT_STEP[base])
        done_cs = set()
        bp = os.path.join(OUT, "ms", "bracket.json")
        if os.path.exists(bp):
            done_cs = {(r["tag"], r["c"]) for r in json.load(open(bp))}
        for c in cs:
            if (tag, c) in done_cs:
                continue
            run_bracket(tag, c)


def phase_E():
    """Single-construction kappa_ms table: pooled-frame est-(i)/(ii) for all cells whose
    pooled frame converges (b8-family expected frame-invalid -> recorded as such)."""
    import experiments.ms_frame_pool as FP
    tags = [d for d in sorted(os.listdir(OUT))
            if os.path.isdir(os.path.join(OUT, d)) and d.startswith("L_")
            and os.path.exists(os.path.join(OUT, "ms", f"{d}_ckpt.pt"))]
    for tag in tags:
        try:
            fz = os.path.join(OUT, "ms", f"{tag}_framepool.npz")
            if not os.path.exists(fz):
                FP.build_frame(tag)
            z = np.load(fz)
            if not bool(z["converged"]):
                print(f"[E] {tag}: frame unconverged (K* -1) -> kappa_ms invalid, recorded", flush=True)
                continue
            if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_pool_pv.npz")):
                FP.build_mpool_V(tag)
            FP.estimator_i(tag); FP.estimator_ii(tag)
        except Exception as e:
            print(f"[E] {tag}: ERROR {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True)
    a = ap.parse_args()
    dict(A=phase_A, B=phase_B, D=phase_D, E=phase_E)[a.phase]()
