"""
Paper-law side (MAY use beta formulas -- separate file from the formula-free ms_cocycle.py):
evaluate EoSS_Momentum.pdf Eq. 21 per cell from the SAME frozen-checkpoint pool, with the
model's actual h_t = u0^T H_B u0 = pool M[0,0] (fixed direction, fixed theta, fresh batches):
    eta_max = 2 a (1+beta)(1-beta) / ((1-beta) a^2 + (1+beta) sigma_b^2),  kappa* = eta_max * a
Compare kappa*_pred to the cell's measured plateau kappa (lr * median lam_batch, live window).
The lambda_B / along-step proxies already failed (wrong h_t); this is the registered clean read.
"""
import os, sys, json, glob
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "results", "kspec"); MS = os.path.join(OUT, "ms")

print(f"{'cell':24s} {'k_meas':>7} {'a(u0)':>7} {'cv2':>6} {'k*pred':>7} {'meas/pred':>9}")
rows = []
for f in sorted(glob.glob(os.path.join(MS, "*_pool.npz"))):
    tag = os.path.basename(f).replace("_pool.npz", "")
    z = np.load(f); beta = float(z["beta"]); lr = float(z["lr"])
    h = z["pool"][:, 0, 0]
    a = float(h.mean()); s2 = float(h.var()) if len(h) > 1 else 0.0
    eta_max = 2 * a * (1 + beta) * (1 - beta) / ((1 - beta) * a * a + (1 + beta) * s2)
    kpred = eta_max * a
    d = np.load(os.path.join(OUT, tag, "dense.npz"))
    win = np.isfinite(d["gu0"]) & np.isfinite(d["lam_batch"])
    gu = d["gu"][win]; lam = d["lam_batch"][win]
    alive = np.abs(gu) > 1e-3 * np.percentile(np.abs(gu), 90)
    kmeas = lr * float(np.median(lam[alive]))
    cv2 = s2 / a**2 if a else float("nan")
    rows.append(dict(tag=tag, kappa_meas=kmeas, a=a, cv2=cv2, kappa_pred=kpred,
                     ratio=kmeas / kpred))
    print(f"{tag:24s} {kmeas:>7.3f} {a:>7.0f} {cv2:>6.3f} {kpred:>7.3f} {kmeas/kpred:>9.2f}")
json.dump(rows, open(os.path.join(MS, "paperlaw.json"), "w"), indent=1)
