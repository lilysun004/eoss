"""
Open-loop / identity validation for the kappa_spec pipeline (KSPEC_DESIGN.md item 3).
Separate file from the estimator ON PURPOSE: this validation legitimately uses the optimizer
recursion (it simulates the optimizer); the estimator stays formula-free and grep-certified.

Two checks per cell:
 1. IN-FRAME IDENTITY (exact by construction, validates the logging): the runner computes
    s = compute_step_direction BEFORE the step, with the pre-update buffer m, all projected on the
    SAME u. For SGD-Momentum: su == -lr*(gu + beta*mu) per-step to float precision.
    (For SGD-Nesterov: su == -lr*((1+beta)*gu + beta^2*mu).)
 2. OPEN-LOOP TRANSFER (closed-loop-bias check): feed the recorded gu series through the momentum
    recursion offline (v_t = beta*v_{t-1} + gu_t; s_t = -lr*v_t) and compare to the logged su.
    EXACT only where u_B is static (large batch); the small-batch gap IS the decoherence (the
    buffer was accumulated in rotating frames) -- report R^2 as a diagnostic, not pass/fail.
"""
import os, sys, json
import numpy as np


def validate(cell_dir):
    z = np.load(os.path.join(cell_dir, "dense.npz"))
    meta = json.load(open(os.path.join(cell_dir, "meta.json")))
    beta, lr, optn = meta["beta"], meta["lr"], meta["optn"]
    gu, su, mu, gu0 = z["gu"], z["su"], z["mu"], z["gu0"]
    win = np.isfinite(gu0) & np.isfinite(gu) & np.isfinite(su) & np.isfinite(mu)
    gu_w, su_w, mu_w = gu[win], su[win], mu[win]

    # 1. in-frame identity
    if optn == "SGD-Nesterov":
        pred = -lr * ((1 + beta) * gu_w + beta * beta * mu_w)
    else:
        pred = -lr * (gu_w + beta * mu_w)
    rel = np.abs(pred - su_w) / (np.abs(su_w) + 1e-15)
    ident = dict(median_rel_err=float(np.median(rel)), p95_rel_err=float(np.percentile(rel, 95)))

    # 2. open-loop reconstruction over the full record (recursion needs history from t=0),
    #    scored on the plateau window
    m = np.isfinite(gu) & np.isfinite(su)
    g_full, s_full = np.where(m, gu, 0.0), su
    v = 0.0; s_ol = np.zeros(len(g_full))
    for t in range(len(g_full)):
        v = beta * v + g_full[t]
        if optn == "SGD-Nesterov":
            s_ol[t] = -lr * (g_full[t] + beta * v)
        else:
            s_ol[t] = -lr * v
    sw, ow = s_full[win], s_ol[win]
    ss_res = float(np.sum((sw - ow) ** 2)); ss_tot = float(np.sum((sw - np.mean(sw)) ** 2))
    ol = dict(R2=1 - ss_res / max(ss_tot, 1e-300), corr=float(np.corrcoef(sw, ow)[0, 1]))
    return dict(cell=os.path.basename(cell_dir), batch=meta["batch"], beta=beta,
                identity=ident, openloop=ol)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        r = validate(d)
        print(f"  {r['cell']:28s} identity rel-err med={r['identity']['median_rel_err']:.2e} "
              f"p95={r['identity']['p95_rel_err']:.2e} | open-loop R2={r['openloop']['R2']:+.3f} "
              f"corr={r['openloop']['corr']:+.3f}")
