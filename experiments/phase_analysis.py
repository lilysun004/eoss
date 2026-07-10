"""
Phase analysis on the dense slow-sweep series -- the tests the old sweep couldn't support.
Everything is on the SLOW variable kappa_t (a scalar, no rotating coordinate) or on the loss/a_t
series, over the plateau window (step >= warmup). Detrend kappa with a slow EMA so landscape
drift doesn't confound the timescale statistics.

  S1 burstiness : catapult inter-arrival (sigma-mu)/(sigma+mu). ~0 uniform (marginal, short-cycle
                  feedback); ->1 clustered bursts (metastable, long-cycle basin). + catapult rate.
  S2 tau_kappa  : AR(1) autocorrelation time of detrended kappa (steps). short = pinned/thermostat
                  (marginal); long = wanders between catapults (metastable).
  S3 restore    : slope of d kappa_{t+1} vs (kappa - mean). negative = restoring; report the
                  from-above vs from-below asymmetry (marginal thermostat shaves from above fast).
  S4 Elog_a     : E[log|a_t|] one-step multiplier (moving frame, rotation-proof). ~0 marginal, <0
                  metastable. + P(a_t<0) (overshoot fraction).
Prints a per-cell table and the marginal-vs-metastable endpoint contrast.
"""
import os, sys, json, glob, re
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.long_train_grid as L

SWEEP = os.path.join(_REPO, "results", "slow_sweep")


def ema_detrend(x, hl=400):
    a = 1 - 0.5 ** (1.0 / hl); e = x[0]; out = np.empty_like(x)
    for i, v in enumerate(x):
        e = (1 - a) * e + a * v; out[i] = v - e
    return out


def ar1_tau(z):
    z = z[np.isfinite(z)]
    if len(z) < 50 or np.std(z) < 1e-12:
        return np.nan
    a1 = float(np.dot(z[:-1] - z.mean(), z[1:] - z.mean()) / (np.dot(z[:-1] - z.mean(), z[:-1] - z.mean()) + 1e-30))
    return -1.0 / np.log(a1) if 0 < a1 < 1 else np.nan


def restoring(kappa):
    k = kappa[np.isfinite(kappa)]
    if len(k) < 50:
        return np.nan, np.nan, np.nan
    x = k[:-1] - k.mean(); dy = np.diff(k)
    slope = float(np.dot(x, dy) / (np.dot(x, x) + 1e-30))
    up = x > 0; dn = x < 0
    s_up = float(np.dot(x[up], dy[up]) / (np.dot(x[up], x[up]) + 1e-30)) if up.sum() > 10 else np.nan
    s_dn = float(np.dot(x[dn], dy[dn]) / (np.dot(x[dn], x[dn]) + 1e-30)) if dn.sum() > 10 else np.nan
    return -slope, -s_up, -s_dn      # positive = restoring


def burstiness(loss):
    on = L.detect_catapults(np.asarray(loss, float))
    if len(on) < 4:
        return np.nan, len(on) / max(1, len(loss))
    iat = np.diff(on).astype(float); mu, sd = iat.mean(), iat.std()
    return float((sd - mu) / (sd + mu + 1e-30)), float(len(on) / len(loss))


def analyze(cell_dir):
    m = json.load(open(os.path.join(cell_dir, "meta.json")))
    if m.get("diverged") or m.get("steps", 0) < 2000:
        return None
    d = np.load(os.path.join(cell_dir, "dense.npz"))
    step = d["step"]; warm = 8000 if m["tag"].startswith("DEPTH") else 6000
    pl = step >= min(warm, step[len(step) // 2])       # plateau window
    kap = d["kappa"][pl]; loss = d["loss"][pl]; a = d["a_t"][pl]
    kap = kap[np.isfinite(kap)]
    if len(kap) < 100:
        return None
    B, rate = burstiness(loss)
    tau = ar1_tau(ema_detrend(kap))
    rs, rup, rdn = restoring(kap)
    av = a[np.isfinite(a)]
    # CLEAN slow variable: lam_full = lr*lambda_max(held-out 2048 batch), sparse but un-confounded
    # by per-step batch noise (per-batch kappa_t is noise-dominated at small batch). S2/S3 here are
    # the trustworthy thermostat stats.
    lf = d.get("lam_full"); lfs = d.get("lf_step")
    tau_f = rs_f = np.nan
    if lf is not None and len(lf) > 30:
        lf = np.asarray(lf, float); lfs = np.asarray(lfs, float)
        lf = lf[lfs >= min(warm, lfs[len(lfs) // 2])]; lf = lf[np.isfinite(lf)]
        if len(lf) > 20:
            tau_f = ar1_tau(ema_detrend(lf)); rs_f = restoring(lf)[0]
    return dict(tag=m["tag"], optn=m["optn"], beta=m["beta"], batch=m["batch"], lr=m["lr"],
                n=len(kap), kappa=float(np.median(kap)), gbs=float(np.nanmedian(d["gbs"][pl])),
                S1_burst=B, cat_rate=rate, S2_tau=tau, S3_restore=rs, S3_up=rup, S3_dn=rdn,
                S2_tau_full=tau_f, S3_restore_full=rs_f,
                S4_Elog_a=float(np.mean(np.log(np.abs(av) + 1e-30))), S4_pneg=float(np.mean(av < 0)),
                censored=m.get("censored"))


def main():
    rows = [r for cd in sorted(glob.glob(os.path.join(SWEEP, "*", "meta.json")))
            for r in [analyze(os.path.dirname(cd))] if r]
    print(f"[phase] {len(rows)} analyzable (plateaued, live) cells\n")
    hdr = f"{'tag':44s}{'kap':>5}{'gbs':>5}{'S1burst':>8}{'catrate':>8}{'S2tau':>7}{'S3rest':>8}{'S4Eloga':>9}"
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["batch"], r["beta"])):
        print(f"{r['tag'][:44]:44s}{r['kappa']:5.2f}{r['gbs']:5.2f}{r['S1_burst']:8.3f}"
              f"{r['cat_rate']:8.4f}{r['S2_tau']:7.1f}{r['S3_restore']:8.4f}{r['S4_Elog_a']:9.4f}")
    json.dump(rows, open(os.path.join(_REPO, "results", "phase_analysis.json"), "w"), indent=1, default=str)

    # endpoint contrast (marginal vs metastable) among DEPTH cells, seed-averaged
    def grp(sub):
        g = [r for r in rows if sub in r["tag"]]
        if not g:
            return None
        return {k: float(np.nanmean([r[k] for r in g])) for k in
                ("S1_burst", "cat_rate", "S2_tau", "S3_restore", "S4_Elog_a", "kappa", "gbs")}
    print("\n=== ENDPOINT CONTRAST (seed-averaged) ===")
    for lbl, sub in [("marginal-small SGD b8", "marginal-small"), ("metastable SGDM b8", "DEPTH_SGDM_b8_beta0.9_meta"),
                     ("marginal-large SGD b2048", "marginal-large_"), ("crossover b128", "crossover")]:
        g = grp(sub)
        if g:
            print(f"  {lbl:26s} kap={g['kappa']:.2f} gbs={g['gbs']:.2f} | S1burst={g['S1_burst']:.3f} "
                  f"S2tau={g['S2_tau']:.1f} S3restore={g['S3_restore']:.4f} S4Eloga={g['S4_Elog_a']:.4f}")
    print("\n marginal expected: low S1burst, short S2tau, strong S3restore, S4~0.")
    print(" metastable expected: high S1burst, long S2tau, weak S3restore, S4<0.")


if __name__ == "__main__":
    main()
