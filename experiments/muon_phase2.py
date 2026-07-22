"""
Muon program on mlp_s (ADDENDUM 10 part B), data only:
  1. L_muon_b2048_s1/s2 -- replication of the single existing cell (same protocol:
     lr 0.001, momentum 0.95, b2048, 16000 steps, u0_at 4000) for reproducibility of the
     kappa_spec reading and the stationarity flag.
  2. kspec tables for the new cells.
  3. Onset brackets on s0/s1, wide grid (1.05, 1.15, 1.3, 1.5) -- frame-free ground truth,
     the PRIMARY Muon validation per the registered frame decision.
  4. cv2h pools for s0/s1 so the Muon margin row lands on the Tier-2 X axis.
Runs against the default mlp_s out dir (results/kspec). Do NOT import arch_battery here --
it force-sets EOSS_MODEL=mlp_l at import. Resume-safe.
"""
import os, sys, json, time, subprocess

os.environ.pop("EOSS_MODEL", None)
os.environ["EOSS_KSPEC_OUT"] = "kspec"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec")
MS = os.path.join(OUT, "ms")
PY = sys.executable

import experiments.ms_cocycle as MC
from experiments.ms_bracket import run_bracket
from experiments.kspec_estimator import analyze_cell

MC.CKPT_STEP["L_muon_b2048"] = 5000
CELLS = [(f"L_muon_b2048_s{s}", s) for s in (1, 2)]


def is_done(tag):
    try:
        return json.load(open(os.path.join(OUT, tag, "meta.json"))).get("status") in ("done", "diverged")
    except Exception:
        return False


def run_cells():
    procs = {}
    for tag, seed in CELLS:
        if is_done(tag):
            continue
        cmd = [PY, "-m", "experiments.slow_sweep", "--tag", tag, "--optn", "Muon",
               "--beta", "0.95", "--batch", "2048", "--lr", "0.001", "--seed", str(seed),
               "--catapult_target", str(10**9), "--max_steps", "16000", "--warmup", str(10**9),
               "--stride", "1", "--u0_at", "4000", "--out_dir", OUT]
        p = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, tag + ".log"), "w"),
                             stderr=subprocess.STDOUT, env=dict(os.environ))
        procs[p] = tag
        print(f"  launch {tag}", flush=True)
    while procs:
        time.sleep(10)
        for p in list(procs):
            if p.poll() is not None:
                print(f"  finished {procs.pop(p)} rc={p.returncode}", flush=True)


def tables():
    for tag, _ in CELLS:
        o = os.path.join(MS, f"{tag}_kspec.json")
        if os.path.exists(o) or not os.path.exists(os.path.join(OUT, tag, "dense.npz")):
            continue
        try:
            r = analyze_cell(os.path.join(OUT, tag))
            json.dump(r, open(o, "w"), indent=1)
            print(f"[kspec] {tag}: kspec={r.get('kappa_spec', float('nan')):.3f} "
                  f"gain={r.get('gain', float('nan')):.4f} r1={r.get('r1_dxu', float('nan')):+.2f} "
                  f"stationary={r.get('stationary')}", flush=True)
        except Exception as e:
            print(f"[kspec] {tag} ERROR {e}", flush=True)


def brackets(tags, cs=(1.05, 1.15, 1.3, 1.5), steps=2000):
    bp = os.path.join(MS, "bracket.json")
    done = {(r["tag"], r["c"]) for r in json.load(open(bp))} if os.path.exists(bp) else set()
    for tag in tags:
        if not is_done(tag):
            continue
        try:
            if not os.path.exists(os.path.join(MS, f"{tag}_ckpt.pt")):
                MC.replay(tag, MC.CKPT_STEP["L_muon_b2048"])
            for c in cs:
                if (tag, c) not in done:
                    run_bracket(tag, c, steps=steps)
        except Exception as e:
            print(f"[br] {tag} ERROR {e}", flush=True)


def pools(tags):
    for tag in tags:
        if not os.path.exists(os.path.join(MS, f"{tag}_ckpt.pt")):
            continue
        if os.path.exists(os.path.join(MS, f"{tag}_pool.npz")):
            continue
        try:
            MC.build_pool(tag)
        except Exception as e:
            print(f"[pool] {tag} ERROR {e}", flush=True)


def main():
    run_cells()
    tables()
    br_tags = ["L_muon_b2048_s0", "L_muon_b2048_s1"]
    brackets(br_tags)
    pools(br_tags)
    print("=== MUON PHASE2 COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
