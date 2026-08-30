"""Path-computable, optimizer-agnostic 'one level deeper' metrics along the top mode, per cell.
  A     = Var(dxu)/Var(lr*gu)                 raw step/gradient-noise amplification (in-frame)
  c     : AR(1) restoring rate of x_t = cumsum(su0) (fixed frame): dx_t = -c x_t + e  ->  c = -cov(dx,x)/var(x)
  alpha : growth exponent of Var(sum_{L} dxu) vs L over L in [8, 512] (in-frame net motion)
Also prints GBS (health-masked), kappa_spec, r1, batch, optn. Uses kspec_results/healthmasked JSON for GBS/kspec.
"""
import os, sys, json, glob, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def metrics(z, lr, mask):
    idx = np.where(mask)[0]
    gu, dxu, su0 = z["gu"][idx], z["dxu"][idx], z["su0"][idx]
    A = float(np.var(dxu) / max(np.var(lr * gu), 1e-300))
    x = np.cumsum(su0); dx = np.diff(x); xx = x[:-1] - x[:-1].mean()
    c = float(-np.cov(dx, xx)[0, 1] / max(np.var(xx), 1e-300))
    Ls = [8, 16, 32, 64, 128, 256, 512]; V = []
    for L in Ls:
        k = len(dxu) // L
        if k < 8: break
        S = dxu[:k * L].reshape(k, L).sum(1); V.append(np.var(S))
    if len(V) >= 3:
        a = np.polyfit(np.log(Ls[:len(V)]), np.log(np.maximum(V, 1e-300)), 1)[0]
    else: a = float("nan")
    return A, c, float(a)
rows = []
for f in sorted(glob.glob(os.path.join(REPO, "kspec_results", "healthmasked", "*_kspec.json"))):
    k = json.load(open(f)); tag = k["cell"]
    d = next((os.path.join(REPO, "results", r, tag) for r in ("kspec", "kspec_arch") if os.path.exists(os.path.join(REPO, "results", r, tag, "dense.npz"))), None)
    if d is None or not k.get("ok"): continue
    z = np.load(os.path.join(d, "dense.npz"))
    m = np.isfinite(z["gu0"]) & np.isfinite(z["gu"]) & np.isfinite(z["su"]) & np.isfinite(z["dxu"]) & np.isfinite(z["su0"])
    if k.get("death_step") is not None: m &= z["step"] < k["death_step"]
    if m.sum() < 600: continue
    A, c, a = metrics(z, k["lr"], m)
    rows.append(dict(cell=tag, optn=k["optn"], beta=k["beta"], batch=k["batch"], gbs=k["gbs_med"], kspec=k["kappa_spec"], r1=k["r1_dxu"], A=A, c=c, alpha=a, kraw=k["kappa_raw"]))
rows.sort(key=lambda r: (r["optn"], r["batch"], r["cell"]))
print(f"{'cell':26s} {'b':>5} {'GBS':>5} {'kspec':>5} {'r1':>6} {'A':>8} {'c':>8} {'alpha':>6} {'k_raw':>6}")
for r in rows:
    print(f"{r['cell']:26s} {r['batch']:>5} {r['gbs']:5.2f} {r['kspec']:5.2f} {r['r1']:+6.2f} {r['A']:8.3f} {r['c']:8.4f} {r['alpha']:6.2f} {r['kraw']:6.2f}")
json.dump(rows, open(os.path.join(REPO, "analysis", "noise_amplification.json"), "w"), indent=1)
