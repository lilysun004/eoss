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

    # Nesterov cells are ONLY analyzable after the paper-threshold anchor passed (registered
    # ruling 3): refuse them otherwise -- no kappa_spec claims on unreplicated cells.
    if any("nest_" in d for d in dirs):
        ap_ = os.path.join(OUT, "trio_anchor.json")
        ok = os.path.exists(ap_) and json.load(open(ap_)).get("anchor_pass")
        if not ok:
            sys.exit("[report] REFUSING Nesterov cells: trio anchor not passed "
                     "(run python -m experiments.kspec_ladder --anchor first).")

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

    # 3. ESTIMATOR SMOKE FIRST (registered): fixed-frame vs in-frame agreement at large batch,
    #    where u_B is static so the two frames MUST agree; disagreement = estimator bug, stop.
    big = [r for r in rows if r["batch"] >= 512]
    for r in big:
        agree = abs(r["kappa_spec_fixed"] / r["kappa_spec"] - 1) if r["kappa_spec"] else float("inf")
        r["frame_agree"] = agree
        print(f"[frame-smoke] {r['cell']:26s} in-frame {r['kappa_spec']:.3f} vs fixed "
              f"{r['kappa_spec_fixed']:.3f} -> {'OK' if agree < 0.1 else 'DISAGREE (estimator bug?)'}")

    # 4. table
    hdr = (f"{'cell':26s} {'b':>5} {'beta':>5} {'N':>6} {'k_raw':>7} {'drift':>6} {'GBS':>6} "
           f"{'r1':>6} {'w*/pi':>6} {'gain':>7} {'kspec':>7} {'CI':>13} {'fixed':>7} "
           f"{'h1':>6} {'h2':>6} {'res':>4} {'regime':>9} {'gate':>8} {'class':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["beta"], r["batch"], r["cell"])):
        # three-way registered class (KSPEC_PREREG_ANNOTATIONS.md): keyed on plateau-ness
        r["class3"] = "(i)" if r["stationary"] else "(ii)"
        if r["gate"] == "at-edge":
            r["class3"] += "+(iii)"
        print(f"{r['cell']:26s} {r['batch']:>5} {r['beta']:>5} {r['n_window']:>6} "
              f"{r['kappa_raw']:>7.3f} {r['kappa_drift']:>+6.2f} {r['gbs_med']:>6.2f} "
              f"{r['r1_dxu']:>+6.2f} {r['omega_star_over_pi']:>6.2f} {r['gain']:>7.2f} "
              f"{r['kappa_spec']:>7.3f} "
              f"[{r['kappa_spec_ci_lo']:>5.2f},{r['kappa_spec_ci_hi']:>5.2f}] "
              f"{r['kappa_spec_fixed']:>7.3f} {r['kappa_spec_h1']:>6.2f} {r['kappa_spec_h2']:>6.2f} "
              f"{'FLAG' if r['res_limited'] else 'ok':>4} {r['regime']:>9} {r['gate']:>8} "
              f"{r['class3']:>6}")

    # 5. LAYER 1 -- committed KSPEC_DESIGN criteria on gate=at-edge cells (unchanged)
    gated = [r for r in rows if r["gate"] == "at-edge"]
    sub = [r for r in rows if r["gate"] == "sub-edge"]
    print(f"\n[layer 1: committed gate] at-edge: {len(gated)}, sub-edge: {len(sub)}")
    if gated:
        ks = [r["kappa_spec"] for r in gated]; kr = [r["kappa_raw"] for r in gated]
        med = float(np.median(ks))
        c1 = 1.8 <= med <= 2.2
        c2 = cv(ks) < 0.5 * cv(kr)
        print(f"  median(kappa_spec) on gated = {med:.3f}  in [1.8,2.2]: {c1}")
        print(f"  CV(kappa_spec)={cv(ks):.3f} vs 0.5*CV(kappa_raw)={0.5*cv(kr):.3f}: {c2}")
        print(f"  COMMITTED VERDICT: {'PASS' if (c1 and c2) else 'FAIL'}")
    if sub:
        print(f"  gate-label two-sided read -- kappa_spec on sub-edge-labeled cells: "
              f"{[round(r['kappa_spec'], 2) for r in sub]}")

    # 6. LAYER 2 -- three-way prediction map (KSPEC_PREREG_ANNOTATIONS.md)
    cls_i = [r for r in rows if r["stationary"]]
    cls_ii = [r for r in rows if not r["stationary"]]
    print(f"\n[layer 2: three-way map] (i) plateaued: {len(cls_i)}, (ii) sub-plateau: {len(cls_ii)}")
    if cls_i:
        ks = [r["kappa_spec"] for r in cls_i]
        print(f"  (i) plateaued (predict ~2 at ALL omega): median={np.median(ks):.3f} "
              f"values={[round(k, 2) for k in sorted(ks)]}")
    if cls_ii:
        print(f"  (ii) sub-plateau (predict <2): "
              f"{[(r['cell'], round(r['kappa_spec'], 2)) for r in cls_ii]}")
    endp = [r for r in rows if r["gate"] == "at-edge"]
    for r in endp:
        print(f"  (iii) endpoint gain cross-check {r['cell']}: measured gain={r['gain']:.3f} "
              f"(compare on the anchor side to the known endpoint value for its regime)")

    # 7. secondary registered test: kappa_spec vs median GBS (independent instruments, slope~1)
    gb = [(r["gbs_med"], r["kappa_spec"]) for r in rows if np.isfinite(r["gbs_med"])]
    if len(gb) >= 3:
        g, k = np.array([x[0] for x in gb]), np.array([x[1] for x in gb])
        slope = float(np.sum(g * k) / np.sum(g * g))           # origin-anchored
        print(f"\n[GBS agreement] corr={np.corrcoef(g, k)[0, 1]:+.3f}, "
              f"origin-anchored slope={slope:.3f} (registered prediction ~1)")

    ks_all = [r["kappa_spec"] for r in rows]; kr_all = [r["kappa_raw"] for r in rows]
    print(f"\n[collapse] whole-ladder CV(kappa_raw)={cv(kr_all):.3f} -> CV(kappa_spec)={cv(ks_all):.3f}")
    json.dump(rows, open(os.path.join(OUT, "kspec_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
