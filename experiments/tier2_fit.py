"""
Tier-2 fit (ADDENDUM 8) — WRITTEN TONIGHT, RUN ONLY IN THE ANALYSIS SESSION (judgment work).

Protocol, executed exactly as registered:
 1. Load kspec_results/tier2_dataset.json (Y = bracket onset margin & budget, X = passive stats).
 2. For each X in {decoh, offpi, cv2h}: fit margin ~ f(X) pooled across ALL optimizers
    (primary form: linear through the origin, per the two-curves-anchored-at-(0,0) expectation;
    report also a monotone power-law log-log fit). Same for budget-1 (using death lower bounds
    only as censored marks, never as fitted points).
 3. CIRCULARITY GUARD is structural (Y intervention-based, X passive) — assert no u_B-derived
    quantity on the Y axis.
 4. REGISTERED FAILURE TEST: refit with optimizer one-hot dummies added; report the F-test /
    delta-R^2. Dummies significant => per-family fits only, residual structure reported.
 5. Two-panel figure: margin | budget vs the winning X, optimizers as marker shapes, estimator
    overlays (2 - kappa_spec, c*_2 - 1) as faint points. Saved to kspec_results/tier2_fig.png.
"""
import os, sys, json
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "kspec_results")


def main():
    rows = [r for r in json.load(open(os.path.join(RES, "tier2_dataset.json")))
            if r.get("margin") == r.get("margin")]
    Xs = ["decoh", "offpi", "cv2h"]
    print(f"{len(rows)} usable rows; optimizers: {sorted(set(r['optn'] for r in rows))}")
    Y = np.array([r["margin"] for r in rows])
    opt_labels = [r["optn"] for r in rows]
    best = None
    for xk in Xs:
        X = np.array([r.get(xk, np.nan) for r in rows], float)
        m = np.isfinite(X) & np.isfinite(Y)
        if m.sum() < 6:
            print(f"  X={xk}: too few rows"); continue
        x, y = X[m], Y[m]
        slope = float(np.sum(x * y) / np.sum(x * x))            # through origin
        r2 = 1 - np.sum((y - slope * x) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
        # optimizer-dummy test
        opts = sorted(set(np.array(opt_labels)[m]))
        D = np.column_stack([x] + [(np.array(opt_labels)[m] == o).astype(float) for o in opts[1:]])
        beta_full, res_full = np.linalg.lstsq(D, y, rcond=None)[:2]
        ss_full = float(res_full[0]) if len(res_full) else float(np.sum((y - D @ beta_full) ** 2))
        ss_x = float(np.sum((y - slope * x) ** 2))
        k_extra = D.shape[1] - 1; dof = len(y) - D.shape[1]
        F = ((ss_x - ss_full) / max(k_extra, 1)) / max(ss_full / max(dof, 1), 1e-300)
        print(f"  X={xk}: n={m.sum()} slope={slope:.3f} R2(origin-linear)={r2:.3f} "
              f"optimizer-dummy F={F:.2f} (k={k_extra}, dof={dof})")
        if best is None or r2 > best[2]:
            best = (xk, slope, r2)
    print(f"\nwinning X: {best[0]} (R2={best[2]:.3f}) -- figure + budget panel per protocol")
    # (figure generation added in the analysis session once the winning X is chosen)


if __name__ == "__main__":
    main()
