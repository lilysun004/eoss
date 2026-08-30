"""
Collapse figure: is there an optimizer-INDEPENDENT law y = F(x) with y = GBS_med (and kappa_spec for
comparison) and x a path-computable, optimizer-agnostic coordinate?

x candidates (all computable during training without knowing the optimizer):
  batch    -- batch size (control; the coordinate that previously did NOT collapse)
  r1       -- lag-1 autocorr of the increment along the top mode (coherence; outcome-side)
  step_top -- median fraction of step energy on the top mode, su^2 / |s|^2  (coupling)
  u_rot    -- 1 - median cos(u_t, u_{t+1})   (top-eigvec rotation rate; landscape/noise-side)
  cv_lam   -- CV of lam_batch over the window  (curvature noise)

Collapse statistic (descriptive, pre-declared): for each x, fit ONE pooled isotonic (monotone) curve
across all optimizers and one isotonic curve PER optimizer; report RMS residuals and the ratio
  C = RMS_pooled / RMS_per_opt .  C ~ 1 => one curve serves all optimizers (collapse);
  C >> 1 => optimizer-specific branches (no collapse).
Also: within-optimizer spread across batch (RMS about the optimizer mean) as the scale reference.

Usage: python analysis/collapse_figure.py [--dirs results/kspec results/kspec_arch] [--json-dirs ...]
Writes analysis/collapse_<y>.{png,pdf} and analysis/collapse_table.csv, prints the stats.
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pava(y):
    """pool-adjacent-violators, non-decreasing fit to y (already sorted by x)."""
    v = list(map(float, y)); w = [1.0] * len(v); i = 0
    blocks = [[v[j], 1, j, j] for j in range(len(v))]  # mean, weight, lo, hi
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            a, c = out.pop(), out.pop()
            out.append([(c[0]*c[1] + a[0]*a[1]) / (c[1]+a[1]), c[1]+a[1], c[2], a[3]])
    f = np.empty(len(v))
    for m, _, lo, hi in out: f[lo:hi+1] = m
    return f


class IsotonicRegression:
    """minimal drop-in: increasing='auto' picks the direction with lower RMS; predict by step interpolation."""
    def __init__(self, increasing="auto", out_of_bounds="clip"): pass
    def fit(self, x, y):
        o = np.argsort(x); xs, ys = np.asarray(x)[o], np.asarray(y)[o]
        up = _pava(ys); dn = -_pava(-ys)
        f = up if np.mean((ys-up)**2) <= np.mean((ys-dn)**2) else dn
        self.xs, self.fs = xs, f; return self
    def predict(self, x):
        return np.interp(np.asarray(x, float), self.xs, self.fs)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"SGD": "#2a78d6", "SGD-Momentum": "#eb6834", "SGD-Nesterov": "#1baf7a", "Adam": "#eda100", "Muon": "#e87ba4"}
MRK = {"SGD": "o", "SGD-Momentum": "s", "SGD-Nesterov": "^", "Adam": "D", "Muon": "P"}
ORDER = ["SGD", "SGD-Momentum", "SGD-Nesterov", "Adam", "Muon"]
XS = [("batch", "batch size (log)", True), ("r1", "r$_1$ (increment lag-1 autocorr along top mode)", False),
      ("step_top", "median step-energy fraction on top mode (log)", True),
      ("u_rot", "1 − median cos(u$_t$, u$_{t+1}$) (log)", True), ("cv_lam", "CV(λ$_B$) over window (log)", True),
      ("sg_mis", "1 − median cos(s, g): step–gradient misalignment (log)", True)]


def load_cells(json_dirs, run_dirs):
    rows = []
    for jd in json_dirs:
        for f in sorted(glob.glob(os.path.join(jd, "*_kspec.json"))):
            k = json.load(open(f))
            if not k.get("ok", True) or not np.isfinite(k.get("gbs_med", np.nan)):
                continue
            tag = k["cell"]
            dense = None
            for rd in run_dirs + [os.path.join(REPO, "results", "kspec_gold")]:
                p = os.path.join(rd, tag, "dense.npz")
                if os.path.exists(p):
                    dense = np.load(p); break
            if dense is None:
                print(f"[skip] no dense.npz for {tag}"); continue
            z = dense
            w = np.isfinite(z["gu0"]) & np.isfinite(z["gu"]) & np.isfinite(z["su"]) & np.isfinite(z["lam_batch"])
            su, sn, lam, cuu = z["su"][w], z["step_norm"][w], z["lam_batch"][w], z["cos_uu"][w]
            csg = z["cos_sg"][w] if "cos_sg" in z.files else np.full(w.sum(), np.nan)
            rows.append(dict(cell=tag, optn=k["optn"], beta=k["beta"], batch=int(k["batch"]),
                             arch="mlp_l" if tag.startswith("A") else "mlp_s", stationary=bool(k.get("stationary", True)),
                             gbs=float(k["gbs_med"]), kspec=float(k["kappa_spec"]), r1=float(k["r1_dxu"]),
                             step_top=float(np.nanmedian(su**2 / np.maximum(sn**2, 1e-30))),
                             u_rot=float(max(1 - np.nanmedian(np.abs(cuu)), 1e-6)),
                             cv_lam=float(np.nanstd(lam) / max(np.nanmean(lam), 1e-30)),
                             sg_mis=float(max(1 - np.nanmedian(csg), 1e-4))))
    return rows


def iso_rms(x, y, groups=None, log=False):
    xx = np.log10(x) if log else x
    if groups is None:
        ir = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(xx, y)
        return float(np.sqrt(np.mean((y - ir.predict(xx))**2)))
    res = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 3:
            res.extend((y[m] - y[m].mean()).tolist()); continue
        ir = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(xx[m], y[m])
        res.extend((y[m] - ir.predict(xx[m])).tolist())
    return float(np.sqrt(np.mean(np.square(res))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dirs", nargs="+", default=[os.path.join(REPO, "kspec_results"), os.path.join(REPO, "kspec_results", "arch")])
    ap.add_argument("--dirs", nargs="+", default=[os.path.join(REPO, "results", "kspec"), os.path.join(REPO, "results", "kspec_arch")])
    ap.add_argument("--out", default=os.path.join(REPO, "analysis"))
    ap.add_argument("--label", default="pre-gold: kspec + kspec_arch, 2026-08-30")
    ap.add_argument("--exclude-hb2048", action="store_true", help="sensitivity: drop heavy-ball SGDM b2048 cells (the GBS hole)")
    ap.add_argument("--arch", default=None, help="restrict to mlp_s or mlp_l")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    rows = load_cells(a.json_dirs, a.dirs)
    if a.exclude_hb2048:
        rows = [r for r in rows if not (r["optn"] == "SGD-Momentum" and r["batch"] == 2048)]
    if a.arch:
        rows = [r for r in rows if r["arch"] == a.arch]
    print(f"{len(rows)} cells")
    import csv
    with open(os.path.join(a.out, f"collapse_table{a.suffix}.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    opt = np.array([r["optn"] for r in rows])
    stats = {}
    for yname in ("gbs", "kspec"):
        y = np.array([r[yname] for r in rows])
        fig, axes = plt.subplots(1, len(XS), figsize=(3.9 * len(XS), 4.4), constrained_layout=True)
        for ax, (xk, xl, lg) in zip(axes, XS):
            x = np.array([r[xk] for r in rows])
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0 if lg else True)
            rp = iso_rms(x[ok], y[ok], log=lg); ro = iso_rms(x[ok], y[ok], groups=opt[ok], log=lg)
            wb = float(np.sqrt(np.mean(np.concatenate([(y[ok][opt[ok] == g] - y[ok][opt[ok] == g].mean()) for g in np.unique(opt[ok])])**2)))
            stats[(yname, xk)] = dict(rms_pooled=rp, rms_per_opt=ro, C=rp / ro if ro > 0 else np.nan, within_opt_spread=wb, n=int(ok.sum()))
            xx = np.log10(x[ok]) if lg else x[ok]
            ir = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(xx, y[ok])
            xs = np.linspace(xx.min(), xx.max(), 200)
            ax.plot(10**xs if lg else xs, ir.predict(xs), color="#777", lw=1.5, ls="--", label="pooled monotone fit", zorder=1)
            for o in ORDER:
                m = ok & (opt == o)
                if not m.any(): continue
                arch = np.array([r["arch"] for r in rows])[m]
                ax.scatter(x[m], y[m], s=46, marker=MRK[o], c=COL[o], edgecolors="white", linewidths=1.2,
                           alpha=np.where(arch == "mlp_s", 0.95, 0.55), label=o, zorder=3)
            ax.axhline(2.0, color="#bbb", lw=1, zorder=0)
            if lg: ax.set_xscale("log")
            ax.set_xlabel(xl); ax.grid(alpha=0.25); ax.set_ylim(-0.1, 2.7)
            ax.set_title(f"C = {rp/ro:.2f}   (pooled {rp:.2f} / per-opt {ro:.2f})", fontsize=10)
        axes[0].set_ylabel({"gbs": "GBS median = sᵀH_B s / (−gᵀs)", "kspec": "κ_spec"}[yname])
        h, l = axes[1].get_legend_handles_labels(); axes[1].legend(h, l, fontsize=8, loc="upper right" if yname == "gbs" else "lower left")
        fig.suptitle(f"Collapse test, y = {yname}: filled = mlp_s, faded = mlp_l   [{a.label}]   C≈1 ⇒ one curve for all optimizers", fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"collapse_{yname}{a.suffix}.{ext}"), dpi=150)
        plt.close(fig)
    print(f"{'y':6s} {'x':9s} {'n':>3} {'RMS_pooled':>10} {'RMS_peropt':>10} {'C':>6} {'within-opt spread':>18}")
    for (yn, xk), s in stats.items():
        print(f"{yn:6s} {xk:9s} {s['n']:>3} {s['rms_pooled']:>10.3f} {s['rms_per_opt']:>10.3f} {s['C']:>6.2f} {s['within_opt_spread']:>18.3f}")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in stats.items()}, open(os.path.join(a.out, f"collapse_stats{a.suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
