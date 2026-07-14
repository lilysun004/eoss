"""Second overnight queue (ADDENDUM 7), strict priority order so partial completion still
delivers #1-#2. Data only; no fits; no SUMMARY/KSPEC_RESULTS writes."""
import os, sys, json, subprocess, time
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec")
from experiments.overnight_0713 import run_pool, is_done
PY = sys.executable

def sh(mod_args):
    print(f"[chain] {' '.join(mod_args)}", flush=True)
    subprocess.run([PY, "-m"] + mod_args, check=False)

def cells_of(specs, seeds):
    return [(f"L_{n}_s{s}", o, b, ba, lr, ms, u0, s)
            for (n, o, b, ba, lr, ms, u0) in specs for s in seeds]

def main():
    # ---- #1: discriminating Nesterov cells (preflight then 2 seeds each)
    sh(["experiments.kspec_ladder", "--extra", "--preflight"])
    pf = json.load(open(os.path.join(OUT, "preflight.json")))
    def lrof(name): return pf[name]["lr"]
    n256 = ("nest_b256_beta0.9", "SGD-Nesterov", 0.9, 256, lrof("nest_b256_beta0.9"), 18000, 4500)
    n512 = ("nest_b512_beta0.9", "SGD-Nesterov", 0.9, 512, lrof("nest_b512_beta0.9"), 16000, 4000)
    run_pool(cells_of([n256, n512], [0, 1]))
    # ---- #2: adam b2048 seeds + adam05 (fifth threshold)
    a2048 = ("adam_b2048", "Adam", 0.9, 2048, 0.001, 16000, 4000)
    a05   = ("adam05_b2048", "Adam", 0.5, 2048, lrof("adam05_b2048"), 16000, 4000)
    run_pool(cells_of([a2048], [2, 3, 4]) + cells_of([a05], [0, 1]))
    # kspec tables for everything new so far
    from experiments.kspec_estimator import analyze_cell
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d)
        if not (d.startswith("L_") and os.path.isdir(p)): continue
        out = os.path.join(OUT, "ms", f"{d}_kspec.json")
        if os.path.exists(out) or not os.path.exists(os.path.join(p, "dense.npz")): continue
        try:
            json.dump(analyze_cell(p), open(out, "w"), indent=1)
            print(f"[kspec] {d} written", flush=True)
        except Exception as e:
            print(f"[kspec] {d} ERROR {e}", flush=True)
    # ---- #2/#3 brackets: adam + nest s1 + new cells
    import experiments.ms_cocycle as MC
    from experiments.ms_bracket import run_bracket
    todo = [("L_adam_b2048_s0", [1.05, 1.15, 1.3]), ("L_adam_b2048_s1", [1.05, 1.15, 1.3]),
            ("L_adam_b128_s0", [1.05, 1.15, 1.3]), ("L_adam_b128_s1", [1.05, 1.15, 1.3]),
            ("L_nest_b2048_beta0.9_s1", [1.05, 1.15, 1.3]),
            ("L_nest_b128_beta0.9_s1", [1.1, 1.25, 1.4]),
            ("L_nest_b8_beta0.9_s1", [1.1, 1.2, 1.3]),
            ("L_nest_b256_beta0.9_s0", [1.05, 1.15, 1.3]), ("L_nest_b256_beta0.9_s1", [1.05, 1.15, 1.3]),
            ("L_nest_b512_beta0.9_s0", [1.05, 1.15, 1.3]),
            ("L_adam05_b2048_s0", [1.05, 1.15, 1.3])]
    done = set()
    bp = os.path.join(OUT, "ms", "bracket.json")
    if os.path.exists(bp):
        done = {(r["tag"], r["c"]) for r in json.load(open(bp))}
    for tag, cs in todo:
        base = "_".join(tag.split("_")[:-1])
        if not is_done(tag): continue
        if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_ckpt.pt")):
            try: MC.replay(tag, MC.CKPT_STEP[base])
            except Exception as e: print(f"[D2] replay {tag} ERROR {e}", flush=True); continue
        for c in cs:
            if (tag, c) not in done:
                try: run_bracket(tag, c)
                except Exception as e: print(f"[D2] bracket {tag} {c} ERROR {e}", flush=True)
    # ---- #3: b64 densification cells + brackets
    b64 = ("b64_beta0.9", "SGD-Momentum", 0.9, 64, lrof("b64_beta0.9"), 24000, 6000)
    run_pool(cells_of([b64], [0, 1]))
    for s in (0, 1):
        tag = f"L_b64_beta0.9_s{s}"
        if not is_done(tag): continue
        if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_ckpt.pt")):
            try: MC.replay(tag, MC.CKPT_STEP["L_b64_beta0.9"])
            except Exception as e: print(f"[D2] {e}", flush=True); continue
        for c in (1.05, 1.15, 1.3):
            if (tag, c) not in done:
                try: run_bracket(tag, c)
                except Exception as e: print(f"[D2] {e}", flush=True)
        out = os.path.join(OUT, "ms", f"{tag}_kspec.json")
        if not os.path.exists(out):
            json.dump(analyze_cell(os.path.join(OUT, tag)), open(out, "w"), indent=1)
    # ---- #4: b128 extra seeds + kappa_ms seed padding
    b128 = ("b128_beta0.9", "SGD-Momentum", 0.9, 128, 0.006, 20000, 5000)
    run_pool(cells_of([b128], [2, 3]))
    pad = ([f"L_b128_beta0.9_s{s}" for s in (2, 3)] +
           [f"L_{b}_s{s}" for b in ("b512_beta0.9", "b2048_beta0.9",
                                    "nest_b128_beta0.9", "nest_b2048_beta0.9") for s in (2, 3)])
    import experiments.ms_frame_pool as FP
    for tag in pad:
        if not is_done(tag): continue
        try:
            base = "_".join(tag.split("_")[:-1])
            if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_ckpt.pt")):
                MC.replay(tag, MC.CKPT_STEP[base])
            if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_framepool.npz")):
                FP.build_frame(tag)
            z = np.load(os.path.join(OUT, "ms", f"{tag}_framepool.npz"))
            if not bool(z["converged"]):
                print(f"[E2] {tag}: frame unconverged, recorded", flush=True); continue
            if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_pool_pv.npz")):
                FP.build_mpool_V(tag)
            FP.estimator_i(tag); FP.estimator_ii(tag)
        except Exception as e:
            print(f"[E2] {tag} ERROR {e}", flush=True)
    # ---- #5 leftover: Muon b2048 (preflight inline, one cell, raw logging only)
    sh(["experiments.slow_sweep", "--tag", "PRE_muon_b2048_lr0.001", "--optn", "Muon",
        "--beta", "0.95", "--batch", "2048", "--lr", "0.001", "--seed", "0",
        "--catapult_target", str(10**9), "--max_steps", "1500", "--warmup", str(10**9),
        "--stride", "1", "--out_dir", os.path.join(OUT, "preflight")])
    try:
        mm = json.load(open(os.path.join(OUT, "preflight", "PRE_muon_b2048_lr0.001", "meta.json")))
        if not mm.get("diverged"):
            run_pool([("L_muon_b2048_s0", "Muon", 0.95, 2048, 0.001, 16000, 4000, 0)])
    except Exception as e:
        print(f"[muon] {e}", flush=True)
    print("=== OVERNIGHT-2 COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
