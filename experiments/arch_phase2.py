"""
Arch battery phase 2 (ADDENDUM 10 part A + completions), mlp_l, data only:
  1. A2_ extended-budget cells (30000 steps, u0_at 20000, SAME preflighted lrs) for the two
     pre-plateau Tier-1 cells (A_b2048_beta0.9, A_adam_b2048) -> kspec tables -> brackets.
  2. Hotter brackets (1.6/1.9/2.3) for the six bracket-censored sub-edge cells (b8 family).
     The pre-plateau A_ b2048 cells are NOT hotter-bracketed: their checkpoints are
     pre-plateau, so the A2 brackets carry the wall claim instead.
  3. cv2h pools (ms_cocycle.build_pool) for every cell with a ckpt -- fills the X column
     the battery left NaN so the ADDENDUM 9 margin-law branch comparison can run.
  4. Muon on mlp_l (ADDENDUM 10 B3): preflight-bisect lr from 0.001, A_muon_b2048 s0/s1,
     table, brackets on the wide grid (no prior wall location for Muon).
Resume-safe like arch_battery: done cells/tables/brackets/pools all skip.
"""
import os, sys, json
import numpy as np

os.environ["EOSS_MODEL"] = "mlp_l"
os.environ["EOSS_KSPEC_OUT"] = "kspec_arch"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from experiments.arch_battery import run_pool, kspec_tables, brackets, probe, OUT, PF
import experiments.ms_cocycle as MC

MC.CKPT_STEP["A2_b2048_beta0.9"] = 20000
MC.CKPT_STEP["A2_adam_b2048"] = 20000
MC.CKPT_STEP["A_muon_b2048"] = 5000

HOT = [f"A_{n}_s{s}" for n in ("b8_beta0.9", "adam_b8", "b8_beta0.99") for s in (0, 1)]


def pools():
    ms = os.path.join(OUT, "ms")
    for f in sorted(os.listdir(ms)):
        if not f.endswith("_ckpt.pt"):
            continue
        tag = f[: -len("_ckpt.pt")]
        if os.path.exists(os.path.join(ms, f"{tag}_pool.npz")):
            continue
        try:
            MC.build_pool(tag)
        except Exception as e:
            print(f"[pool] {tag} ERROR {e}", flush=True)


def muon_lr():
    pf = json.load(open(PF))
    if pf.get("muon_b2048", {}).get("lr"):
        return pf["muon_b2048"]["lr"]
    lr, hi, lo, acc, hist = 0.001, None, None, None, []
    for it in range(6):
        v, k = probe("muon_b2048", "Muon", 0.95, 2048, lr, 1200)
        hist.append(dict(lr=lr, verdict=v, kappa=None if k != k else k))
        print(f"[apre] muon_b2048 probe{it+1}: lr={lr:.6g} -> {v} (kappa {k:.3f})", flush=True)
        if v == "live":
            acc = lr
            break
        if v == "diverged":
            hi = lr
            lr = float(np.sqrt(lo * hi)) if lo else lr / 2
        else:
            lo = lr
            lr = float(np.sqrt(lo * hi)) if hi else lr * 1.5
    if acc is None:
        nond = [h for h in hist if h["verdict"] != "diverged"]
        acc = max(nond, key=lambda h: h["lr"])["lr"] if nond else None
    pf["muon_b2048"] = dict(lr=acc, history=hist)
    json.dump(pf, open(PF, "w"), indent=1)
    return acc


def main():
    a2 = ([(f"A2_b2048_beta0.9_s{s}", "SGD-Momentum", 0.9, 2048, 0.0065, 30000, 20000, s) for s in (0, 1)]
          + [(f"A2_adam_b2048_s{s}", "Adam", 0.9, 2048, 0.001, 30000, 20000, s) for s in (0, 1)])
    run_pool(a2, conc=3)
    kspec_tables([c[0] for c in a2])
    brackets([c[0] for c in a2])
    brackets(HOT, cs=(1.6, 1.9, 2.3), steps=3000)
    pools()
    lr = muon_lr()
    if lr:
        mu = [(f"A_muon_b2048_s{s}", "Muon", 0.95, 2048, lr, 16000, 5000, s) for s in (0, 1)]
        run_pool(mu, conc=2)
        kspec_tables([c[0] for c in mu])
        brackets([c[0] for c in mu], cs=(1.05, 1.15, 1.3, 1.5))
        pools()
    else:
        print("[muon-l] no clean lr found; mlp_l Muon cells skipped", flush=True)
    print("=== ARCH PHASE2 COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
