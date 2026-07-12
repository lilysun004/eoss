"""
kappa_spec report: orchestrates GATE (blind, persisted first) -> ESTIMATOR -> merged table ->
pre-registered pass criteria from KSPEC_DESIGN.md, applied as written:
  PASS iff on GATED (at-edge) cells: median(kappa_spec) in [1.8, 2.2]
       AND CV(kappa_spec) < 0.5 * CV(kappa_raw)     (both computed over the same gated set)
  Two-sided: kappa_spec < 2 expected on sub-edge cells.
Also prints: in-frame vs fixed-frame agreement, split-half stability, robustness variants,
and the whole-ladder CVs (raw vs spec) for the collapse claim.

Usage: python -m experiments.kspec_report [--dirs results/kspec/L_*]
"""
import os, sys, glob, json, argparse
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec")
from experiments.kspec_estimator import analyze_cell
import experiments.kspec_gate as G


def cv(x):
    x = np.asarray(x, float)
    return float(np.std(x) / np.abs(np.mean(x))) if len(x) and np.mean(x) != 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None)
    a = ap.parse_args()
    dirs = a.dirs or sorted(d for d in glob.glob(os.path.join(OUT, "L_*")) if os.path.isdir(d))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "dense.npz"))]

    # 1. gate FIRST (blind; no-op if already persisted)
    G.main(dirs)
    gates = {g["cell"]: g for g in json.load(open(os.path.join(OUT, "gates.json")))}

    # 2. estimator
    rows = []
    for d in dirs:
        r = analyze_cell(d)
        if not r.get("ok"):
            print(f"  SKIP {r['cell']}: {r.get('why')}"); continue
        r["gate"] = gates.get(r["cell"], {}).get("gate", "?")
        r["regime"] = gates.get(r["cell"], {}).get("regime", "?")
        rows.append(r)

    # 3. table
    hdr = (f"{'cell':26s} {'b':>5} {'beta':>5} {'N':>6} {'k_raw':>7} {'r1':>6} {'w*/pi':>6} "
           f"{'gain':>7} {'kspec':>7} {'fixed':>7} {'coh':>7} {'dxw':>7} {'h1':>6} {'h2':>6} "
           f"{'regime':>9} {'gate':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["beta"], r["batch"], r["cell"])):
        print(f"{r['cell']:26s} {r['batch']:>5} {r['beta']:>5} {r['n_window']:>6} "
              f"{r['kappa_raw']:>7.3f} {r['r1_dxu']:>+6.2f} {r['omega_star_over_pi']:>6.2f} "
              f"{r['gain']:>7.2f} {r['kappa_spec']:>7.3f} {r['kappa_spec_fixed']:>7.3f} "
              f"{r['kappa_spec_coh']:>7.3f} {r['kappa_spec_dxw']:>7.3f} "
              f"{r['kappa_spec_h1']:>6.2f} {r['kappa_spec_h2']:>6.2f} "
              f"{r['regime']:>9} {r['gate']:>8}")

    # 4. pre-registered criteria
    gated = [r for r in rows if r["gate"] == "at-edge"]
    sub = [r for r in rows if r["gate"] == "sub-edge"]
    print(f"\n[criteria] gated (at-edge) cells: {len(gated)}, sub-edge: {len(sub)}")
    if gated:
        ks = [r["kappa_spec"] for r in gated]; kr = [r["kappa_raw"] for r in gated]
        med = float(np.median(ks))
        c1 = 1.8 <= med <= 2.2
        c2 = cv(ks) < 0.5 * cv(kr)
        print(f"  median(kappa_spec) on gated = {med:.3f}  in [1.8,2.2]: {c1}")
        print(f"  CV(kappa_spec)={cv(ks):.3f} vs 0.5*CV(kappa_raw)={0.5*cv(kr):.3f}: {c2}")
        print(f"  PRE-REGISTERED VERDICT: {'PASS' if (c1 and c2) else 'FAIL'}")
    if sub:
        print(f"  two-sided check -- kappa_spec on sub-edge cells: "
              f"{[round(r['kappa_spec'], 2) for r in sub]} (prediction: < 2)")
    ks_all = [r["kappa_spec"] for r in rows]; kr_all = [r["kappa_raw"] for r in rows]
    print(f"  whole-ladder collapse: CV(kappa_raw)={cv(kr_all):.3f} -> CV(kappa_spec)={cv(ks_all):.3f}")
    json.dump(rows, open(os.path.join(OUT, "kspec_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
