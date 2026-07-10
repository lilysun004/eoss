"""
Paired dense-logging sweep driver: SGD-Momentum {beta 0.6,0.9,0.99} + matched SGD control at
every (batch, lr). Shared per-batch lr grid => exact SGD/SGDM pairing at matched batch-noise
(turns every confounded passive stat into a valid column-ratio).

Barbell design (per the plan): baseline grid at min(25 catapults, 40k steps); ~6 endpoint/
archetype cells at 3-4x depth (target ~90 catapults, 80k cap, 3 seeds). Trim beta breadth
(no 0.3/0.95) not lr/batch. Priority ordering so a night that dies at 60% still answers the
phase question: endpoints -> SGD twins -> SGDM canonical-lr -> extra lrs.

  python -m experiments.slow_sweep_driver --smoke        # time 1 b8 + 1 b2048, bisect b2048 lr, estimate, DON'T run
  python -m experiments.slow_sweep_driver --run [--concurrency 3]
"""
import os, sys, json, time, subprocess, argparse
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "slow_sweep")
os.makedirs(OUT, exist_ok=True)

BATCHES = [8, 32, 128, 512, 2048]
BETAS = [0.0, 0.6, 0.9, 0.99]                      # 0.0 = the paired SGD control column
# Per-batch shared lr grids on the VALIDATED live-momentum span (from the old sweep's live cells);
# momentum's stochastic edge sits BELOW SGD's, so the grid is anchored on the (lower) momentum-live
# range -- SGD cells there are sub-edge-but-live (valid matched-noise controls) and the top lr
# drives SGD toward its own edge. Shared per batch => exact SGD/SGDM pairing at each (B, lr).
PER_BATCH_LR = {
    8:    [0.0015, 0.0025, 0.0040, 0.0065],
    32:   [0.0015, 0.0028, 0.0050, 0.0090],
    128:  [0.0015, 0.0030, 0.0060, 0.0110],
    512:  [0.0020, 0.0040, 0.0080, 0.0140],
    2048: [0.0040, 0.0065, 0.0100, 0.0170],
}
SEEDS = {8: 3, 32: 3, 128: 2, 512: 1, 2048: 1}
STRIDE = {8: 1, 32: 1, 128: 1, 512: 2, 2048: 2}    # cost control on the big-batch HVP

# The 6 barbell depth cells, explicit VALIDATED-live lrs (from old-sweep live cells + archetypes),
# 3-4x catapult budget + 3 seeds. Marginal-large-memory is beta0.6 (beta0.9 b2048 has no live
# window -- reach-edge-then-diverge above, converge-dead below).
DEPTH_CELLS = [
    ("SGD",          0.0,  8,    0.0065, "marginal-small"),
    ("SGD-Momentum", 0.9,  8,    0.0040, "metastable"),
    ("SGD-Momentum", 0.99, 8,    0.0015, "metastable-deep"),
    ("SGD",          0.0,  2048, 0.0170, "marginal-large"),
    ("SGD-Momentum", 0.6,  2048, 0.0065, "marginal-large-memory"),
    ("SGD-Momentum", 0.9,  128,  0.0060, "crossover"),
]


def optn_of(beta):
    return ("SGD", "SGD") if beta == 0.0 else ("SGD-Momentum", "SGDM")


def build_cells():
    cells = []
    # depth cells first (prio 0): explicit lrs, 3 seeds, deep catapult budget
    for optn, beta, B, lr, label in DEPTH_CELLS:
        short = optn_of(beta)[1]
        for s in range(3):
            cells.append(dict(tag=f"DEPTH_{short}_b{B}_beta{beta}_{label}_s{s}", optn=optn,
                              beta=beta, batch=B, lr=lr, seed=s, catapult_target=90,
                              max_steps=80000, warmup=8000, stride=STRIDE[B], prio=0, label=label))
    # regular grid
    for B in BATCHES:
        grid = PER_BATCH_LR[B]
        canon = len(grid) - 2      # headline operating point per (B,beta)
        for beta in BETAS:
            optn, short = optn_of(beta)
            for lri, lr in enumerate(grid):
                prio = 1 if beta == 0.0 else (2 if lri == canon else 3)   # SGD twins first
                for s in range(SEEDS[B]):
                    cells.append(dict(tag=f"{short}_b{B}_beta{beta}_lr{lri}_s{s}", optn=optn,
                                      beta=beta, batch=B, lr=lr, seed=s, catapult_target=25,
                                      max_steps=40000, warmup=6000, stride=STRIDE[B], prio=prio,
                                      label=""))
    cells.sort(key=lambda c: (c["prio"], c["batch"], c["beta"], c["lr"], c["seed"]))
    return cells


def cell_cmd(c):
    return [sys.executable, "-m", "experiments.slow_sweep",
            "--tag", c["tag"], "--optn", c["optn"], "--beta", str(c["beta"]),
            "--batch", str(c["batch"]), "--lr", str(c["lr"]), "--seed", str(c["seed"]),
            "--catapult_target", str(c["catapult_target"]), "--max_steps", str(c["max_steps"]),
            "--warmup", str(c["warmup"]), "--stride", str(c["stride"]), "--out_dir", OUT]


def is_done(tag):
    mp = os.path.join(OUT, tag, "meta.json")
    if not os.path.exists(mp):
        return False
    try:
        return json.load(open(mp)).get("status") in ("done", "diverged")
    except Exception:
        return False


# ---------------------------------------------------------------- smoke: liveness of depth cells
def smoke():
    """Quick liveness check on the 6 depth cells at their validated lrs (1500 steps each): confirm
    none diverge and kappa reaches a sensible level, before committing the overnight run."""
    import experiments.slow_sweep as S
    print("[smoke] liveness of the 6 depth cells (1500 steps, validated lrs)...")
    for optn, beta, B, lr, label in DEPTH_CELLS:
        t0 = time.time()
        S.run_cell(f"SMOKE_{label}", optn, beta, B, lr, OUT, catapult_target=10 ** 9,
                   max_steps=1500, warmup=10 ** 9, stride=STRIDE[B], seed=0)
        m = json.load(open(os.path.join(OUT, f"SMOKE_{label}", "meta.json")))
        d = np.load(os.path.join(OUT, f"SMOKE_{label}", "dense.npz"))
        kap = float(np.nanmedian(d["kappa"][-30:])) if len(d["kappa"]) > 30 else float("nan")
        ms = 1000 * (time.time() - t0) / 1500
        print(f"  {label:22s} b{B} {optn[:4]} b{beta} lr{lr}: diverged={m.get('diverged')} "
              f"kappa~{kap:.2f}  ({ms:.1f} ms/step)", flush=True)
    cells = build_cells()
    print(f"[smoke] full grid = {len(cells)} cells "
          f"(prio: {[sum(1 for c in cells if c['prio']==p) for p in range(4)]})")


def run(concurrency):
    cells = [c for c in build_cells() if not is_done(c["tag"])]
    print(f"[run] {len(cells)} cells to run (concurrency {concurrency}); priority order")
    procs = {}
    it = iter(cells)
    def launch(c):
        p = subprocess.Popen(cell_cmd(c), stdout=open(os.path.join(OUT, c["tag"] + ".log"), "w"),
                             stderr=subprocess.STDOUT)
        procs[p] = c; print(f"  launch {c['tag']} (prio {c['prio']}) pid={p.pid}", flush=True)
    for _ in range(concurrency):
        try:
            launch(next(it))
        except StopIteration:
            break
    while procs:
        time.sleep(5)
        for p in list(procs):
            if p.poll() is not None:
                c = procs.pop(p); print(f"  finished {c['tag']} (rc={p.returncode})", flush=True)
                try:
                    launch(next(it))
                except StopIteration:
                    pass
    print("[run] all cells complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    if a.smoke:
        smoke()
    elif a.run:
        run(a.concurrency)
    else:
        cells = build_cells()
        print(f"{len(cells)} cells. by priority:",
              {p: sum(1 for c in cells if c['prio'] == p) for p in range(4)})


if __name__ == "__main__":
    main()
