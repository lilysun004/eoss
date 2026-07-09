"""
Passive (no-intervention) regime discriminators from already-saved plateau series.

Rationale (fluctuation-dissipation): the minibatch noise already kicks the system in every
direction every step, so the natural trajectory record IS the response experiment -- and
time-averaged dimensionless statistics are robust to both stochasticity (it's the probe) and
slow landscape drift (detrend + ratios). The regimes differ in TIMESCALE structure (short-cycle
feedback = marginal vs long-cycle basin = metastable), so the discriminators are timescale
statistics, not amplitudes.

HONEST SCOPE: the sweep saved dense per-step series only for the u-projection x_t and the loss
(catapult.npz: x_detrended, loss, 4000 steps). It did NOT save dense kappa_t (metrics_traj is
every ~800 steps) or per-step multipliers a_t. So the kappa-based mean-reversion (S2/S3) and the
one-step multiplier (S1/S4) need a dense-logging rerun. What we compute here:
  - loss_t : CLEAN, geometry-free scalar -> the trustworthy passive channel.
  - x_t    : the fixed-u projection -> FAST coordinate, rotation-contaminated at small batch
             (same caveat as gamma_proj); reported but flagged.

Statistics per cell (on both channels, robustly standardized by MAD):
  burstiness  B = (sigma-mu)/(sigma+mu) of exceedance inter-arrival times.  ~0 Poisson/uniform
                 (marginal, short-cycle);  ->1 clustered bursts (metastable catapults).
  exc_frac    fraction of steps above the exceedance threshold.  higher = straddling boundary.
  tau_ar      AR(1) mean-reversion time (steps): fit z_{t+1}=a z_t -> tau=-1/ln|a|.  short =
                 pinned/fast feedback (marginal); long = wanders between catapults (metastable).
  cat_rate    exceedance rate.
Endpoint test: marginal (large-batch, low-beta) should read {B~0, high exc_frac, short tau};
metastable (small-batch high-beta) should read {B->1, low exc_frac, long tau}. If the archetype
endpoints separate, this passive suite maps the whole sweep for free.
"""
import os, sys, json, glob, re
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(_REPO, "results", "comprehensive_sweep")
OUT = os.path.join(_REPO, "results", "plots")
os.makedirs(OUT, exist_ok=True)


def _mad_std(a):
    return 1.4826 * np.median(np.abs(a - np.median(a))) + 1e-30


def burstiness_and_rate(z, thr=3.0):
    """z standardized; exceedances |z|>thr. Return (B, exc_frac, cat_rate)."""
    z = z[np.isfinite(z)]
    if len(z) < 50:
        return np.nan, np.nan, np.nan
    exc = np.where(np.abs(z) > thr)[0]
    frac = len(exc) / len(z)
    if len(exc) < 4:
        return np.nan, frac, len(exc) / len(z)
    iat = np.diff(exc).astype(float)
    mu, sd = iat.mean(), iat.std()
    B = (sd - mu) / (sd + mu + 1e-30)
    return float(B), float(frac), float(len(exc) / len(z))


def tau_ar1(z):
    """AR(1) mean-reversion time in steps: z_{t+1}=a z_t -> tau=-1/ln|a| (nan if a>=1)."""
    z = z[np.isfinite(z)]
    if len(z) < 50:
        return np.nan
    a = float(np.dot(z[:-1], z[1:]) / (np.dot(z[:-1], z[:-1]) + 1e-30))
    if a <= 0 or a >= 1:
        return np.nan
    return float(-1.0 / np.log(a))


def cell_stats(cat_path):
    d = np.load(cat_path)
    out = {}
    for chan, key in [("loss", "loss"), ("x", "x_detrended")]:
        s = np.asarray(d[key], float); s = s[np.isfinite(s)]
        if len(s) < 50:
            continue
        z = (s - np.median(s)) / _mad_std(s)
        B, frac, rate = burstiness_and_rate(z)
        out[f"B_{chan}"] = B; out[f"excfrac_{chan}"] = frac
        out[f"catrate_{chan}"] = rate; out[f"tau_{chan}"] = tau_ar1(z)
    return out


def load_all():
    rows = []
    for cat in sorted(glob.glob(os.path.join(SWEEP, "b*_beta*", "catapult.npz"))):
        d = os.path.dirname(cat); tag = os.path.basename(d)
        m = re.match(r"b(\d+)_beta([\d.]+)_lr(\d)", tag)
        if not m:
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        st = meta.get("stationarity") or {}
        rows.append(dict(tag=tag, B=int(m[1]), beta=float(m[2]), lr_index=int(m[3]),
                         R=meta.get("plateau_R"), stabilized=bool(st.get("stabilized")),
                         **cell_stats(cat)))
    return rows


def main():
    rows = load_all()
    print(f"[passive] {len(rows)} cells with catapult series\n")
    # archetype endpoints
    def pick(B, beta):
        c = [r for r in rows if r["B"] == B and abs(r["beta"] - beta) < 1e-6]
        return max(c, key=lambda r: r["lr_index"]) if c else None
    arche = [("marginal-large  b2048 b0.3", pick(2048, 0.3)),
             ("marginal-large  b2048 b0.6", pick(2048, 0.6)),
             ("marginal-small  b8 b0.3", pick(8, 0.3)),
             ("metastable      b8 b0.9", pick(8, 0.9)),
             ("metastable      b8 b0.99", pick(8, 0.99))]
    print(f"{'archetype':30s}{'R':>8s} | {'B_loss':>7s}{'tau_loss':>9s}{'excf_loss':>10s} |"
          f"{'B_x':>6s}{'tau_x':>7s}")
    for name, r in arche:
        if r is None:
            print(f"{name:30s}  (missing)"); continue
        print(f"{name:30s}{(r['R'] or 0):8.2f} | {r.get('B_loss',np.nan):7.2f}"
              f"{r.get('tau_loss',np.nan):9.2f}{r.get('excfrac_loss',np.nan):10.4f} |"
              f"{r.get('B_x',np.nan):6.2f}{r.get('tau_x',np.nan):7.2f}")
    json.dump(rows, open(os.path.join(_REPO, "results", "passive_stats.json"), "w"),
              indent=1, default=str)

    # scatter valid cells vs R
    import matplotlib
    if os.environ.get("EOSS_NO_SHOW") == "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r["stabilized"] and r.get("R") and r["R"] > 0]
    panels = [("B_loss", "loss burstiness (→1 = bursty/metastable)"),
              ("tau_loss", "loss AR(1) time (steps)"),
              ("excfrac_loss", "loss exceedance fraction"),
              ("B_x", "x burstiness (fixed-u; caveat)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, lbl) in zip(axes.flat, panels):
        ax.axvspan(1e-5, 1, color="#2ca02c", alpha=0.07); ax.axvline(1, color="0.4", ls="--", lw=1)
        for r in valid:
            y = r.get(key)
            if y is None or not np.isfinite(y):
                continue
            ax.scatter(r["R"], y, s=40, c=[np.log10(r["B"])], cmap="viridis",
                       vmin=0.9, vmax=3.3, edgecolors="k", linewidths=0.3)
        ax.set_xscale("log"); ax.set_xlabel("R"); ax.set_ylabel(lbl); ax.grid(alpha=0.15)
    fig.suptitle("Passive regime statistics vs R (color = log10 batch; loss = clean channel)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "passive_stats.png"), dpi=140)
    print(f"\n[passive] wrote {os.path.join(OUT, 'passive_stats.png')}")


if __name__ == "__main__":
    main()
