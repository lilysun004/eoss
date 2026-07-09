"""
Regime map: live cells only, parameterized by R (and, honestly, by (memory, tau_rot)).

Organizing claim: R = (state memory 1/(1-beta)) / (unstable-direction rotation time tau_rot)
decides MARGINAL (R<~1, at edge, GBS=2) vs METASTABLE (R>>1, sub-edge damped basin). But the
regime question is only *posed* for cells whose dynamics are ALIVE. This script therefore
classifies every cell 4 ways and plots only the valid (live) ones, at the canonical operating
point (largest stable lr per (B,beta)).

CLASSIFICATION (the gate is ACTIVITY, not stationarity):
  - diverged      : loss blew up.
  - not_plateaued : GBS never stabilized (typically lowest-lr cells still descending toward EoS).
  - converged_dead: GBS "stable" but the DYNAMICS STOPPED -- the iterate quit moving (step-norm
                    collapsed vs early training). These converged to a minimum without ever going
                    unstable (e.g. high-beta full-batch: permissive edge 2(1+beta)/eta, small lr).
                    They're neither marginal nor metastable; their R~1e-5 is an ARTIFACT (no motion
                    -> u_B frozen -> tau_rot->inf -> R->0), so they must be excluded, not plotted.
  - valid         : plateaued AND alive (persistent motion). Metastable b8 passes (noise, catapults);
                    converged full-batch cells fail.

PANELS (headline, canonical-lr, valid-only): GBS, kappa, cos(buffer,u_B), gamma_proj, Hill alpha.
Dropped from headline: gamma_full (full-space, chaotic bulk, positive everywhere -- not the u-mode
response), sharpen (interpolation-confounded), kurtosis & p99/p50 (fragile; Hill alpha is the tail
signal). CAVEAT on gamma_proj: it is fit to <dv, u_FIXED> (kick-time eigenvector); at high R the
coordinate rotates out from under dv, so gamma_proj is artifact-prone there -- see the archetype
rerun (experiments/archetype_gamma.py) with rotation-robust coordinates.

6th panel: the 2D (memory, tau_rot) plane colored by GBS, with the diagonal memory=tau_rot (R=1).
If R is exactly right, the marginal/metastable boundary is that diagonal; a different boundary
shape would be a sharper law than the ratio.

Usage: run the `# %%` cells inline (each figure shows + saves to results/plots/), or `python
plot.py` (set EOSS_NO_SHOW=1 for headless). Override dirs with EOSS_SWEEP_DIR / EOSS_PLOT_OUT.
"""
import os, json, glob
import numpy as np
import matplotlib
SHOW = os.environ.get("EOSS_NO_SHOW") != "1"
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO = os.path.dirname(os.path.abspath(__file__))
DEAD_ACT_THRESH = 0.05   # step-norm plateau/early ratio below this => dynamics dead


def _f(x):
    try:
        return float(x) if x is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _plateau_mean(a, frac=0.4):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.mean(a[int(len(a) * (1 - frac)):])) if len(a) else np.nan


def _activity_ratio(z):
    """median(step_norm over late plateau) / median(step_norm over early training).

    ~1 (or growing) for a live iterate; ->0 when the run converged and the iterate stopped
    moving. Rotation-robust and scale-free (self-normalized within the run)."""
    try:
        sn = np.asarray(z["step_norm"], float); sn = sn[np.isfinite(sn)]
        if len(sn) < 4:
            return np.nan
        early = np.median(sn[:max(1, int(len(sn) * 0.3))])
        late = np.median(sn[int(len(sn) * 0.5):])
        return float(late / (early + 1e-30))
    except (KeyError, ValueError):
        return np.nan


def classify(sweep_dir):
    """Return (valid_rows, classes). `classes` has one entry per cell with its klass; valid_rows
    is only the live+plateaued cells, with the fields the panels need."""
    valid, classes = [], []
    for mp in sorted(glob.glob(os.path.join(sweep_dir, "b*_beta*", "meta.json"))):
        try:
            m = json.load(open(mp))
        except (json.JSONDecodeError, OSError):
            continue
        tag = m.get("tag"); B = int(m["B"]); beta = float(m["beta"])
        rec_meta = dict(tag=tag, B=B, beta=beta, lr_index=m.get("lr_index"))
        if m.get("diverged"):
            classes.append(dict(klass="diverged", **rec_meta)); continue
        st = m.get("stationarity") or {}
        if not st.get("stabilized"):
            classes.append(dict(klass="not_plateaued", **rec_meta)); continue
        # activity gate
        act = np.nan; tau_rot = np.nan
        try:
            z = np.load(mp.replace("meta.json", "metrics_traj.npz"))
            act = _activity_ratio(z)
            tau_rot = _plateau_mean(z["tau_rot"]) if "tau_rot" in z else np.nan
        except (OSError, ValueError):
            pass
        gbs = _f(m.get("plateau_gbs"))
        # Dead = the top unstable mode is not being excited. Two flavors:
        #  (A) iterate stopped moving -> step-norm collapsed (act < thresh).
        #  (B) high-beta full-batch "coasting on momentum": step-norm stays up (act high) but u_B is
        #      frozen (tau_rot->inf) AND the cell never reached its edge (GBS~0). The GBS<0.5 clause
        #      is essential: a LIVE marginal full-batch cell ALSO has frozen u_B (clean period-2
        #      oscillation along a fixed direction) but sits at GBS~2 -- it must be kept.
        frozen = (not np.isfinite(tau_rot)) or tau_rot > 1e3
        dead = (np.isfinite(act) and act < DEAD_ACT_THRESH) or (frozen and np.isfinite(gbs) and gbs < 0.5)
        if dead:
            classes.append(dict(klass="converged_dead", act=act, tau_rot=tau_rot, **rec_meta)); continue
        classes.append(dict(klass="valid", act=act, **rec_meta))
        valid.append(dict(
            tag=tag, B=B, beta=beta, lr=_f(m.get("lr")), lr_index=m.get("lr_index"),
            R=_f(m.get("plateau_R")), act=act,
            memory=1.0 / (1.0 - beta) if beta < 1 else np.inf, tau_rot=tau_rot,
            gbs=_f(m.get("plateau_gbs")), kappa=_f(m.get("plateau_kappa")),
            cos_buf_uB=_f(m.get("plateau_cos_buf_uB")),
            gamma_proj=_f(m.get("perturb_gamma_proj_mean")),
            gamma_proj_std=_f(m.get("perturb_gamma_proj_std")),
            hill=_f(m.get("catapult_hill")),
        ))
    return valid, classes


def canonical(rows):
    """One row per (B, beta): the largest-lr valid cell = the canonical operating point (where
    'does it reach its edge' is actually asked). Sub-maximal lrs go to the appendix."""
    best = {}
    for r in rows:
        k = (r["B"], round(r["beta"], 4))
        if k not in best or (r["lr"] > best[k]["lr"]):
            best[k] = r
    return list(best.values())


def print_classification(classes):
    order = ["valid", "converged_dead", "not_plateaued", "diverged"]
    counts = {k: [c for c in classes if c["klass"] == k] for k in order}
    print(f"[classify] {len(classes)} cells:  " +
          "  ".join(f"{k}={len(counts[k])}" for k in order))
    if counts["converged_dead"]:
        print("  converged-dead (excluded; dynamics stopped, R artifactually ~0):")
        for c in sorted(counts["converged_dead"], key=lambda c: (c["B"], c["beta"])):
            tr = c.get("tau_rot", float("nan"))
            frozen = " [u_B frozen]" if (not np.isfinite(tr) or tr > 1e3) else ""
            print(f"    {c['tag']:26s} step-norm late/early={c['act']:.4f}{frozen}")
    if counts["not_plateaued"]:
        tags = sorted(c["tag"] for c in counts["not_plateaued"])
        print(f"  not-plateaued (excluded): {len(tags)}  e.g. {', '.join(tags[:6])}")


# ---- panels (headline): key, label, href, ylim ---------------------------------------------
PANELS = [
    ("gbs",        "GBS  (edge = 2)",                      2.0, (-0.3, 2.7)),
    ("kappa",      r"$\kappa=\eta\lambda$  (edge frac.)",  None, None),
    ("cos_buf_uB", r"cos(buffer, $u_B$)  (tracking)",      None, None),
    ("gamma_proj", r"$\gamma_{proj}$ (fixed-$u$; caveat)", 0.0, None),
    ("hill",       r"catapult Hill $\alpha$  (m.s.=2)",    2.0, None),
]
BETA_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _shade(ax, xlo, xhi):
    ax.axvspan(xlo, 1.0, color="#2ca02c", alpha=0.07, zorder=0)
    ax.axvspan(1.0, xhi, color="#ff7f0e", alpha=0.07, zorder=0)
    ax.axvline(1.0, color="0.4", ls="--", lw=1, zorder=1)


def fig_regime_map(rows, out_path, title_extra=""):
    if not rows:
        print("[plot] no valid cells to plot"); return None
    batches = sorted({r["B"] for r in rows}); betas = sorted({r["beta"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(batches)))
    bcolor = {b: cmap[i] for i, b in enumerate(batches)}
    bmark = {be: BETA_MARKERS[i % len(BETA_MARKERS)] for i, be in enumerate(betas)}
    Rall = np.array([r["R"] for r in rows if np.isfinite(r["R"]) and r["R"] > 0])
    xlo, xhi = Rall.min() * 0.6, Rall.max() * 1.7

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (key, ylabel, href, ylim) in zip(axes.flat[:5], PANELS):
        _shade(ax, xlo, xhi); noff = 0
        for r in rows:
            y = r.get(key)
            if y is None or not np.isfinite(y) or not (np.isfinite(r["R"]) and r["R"] > 0):
                continue
            if ylim and not (ylim[0] <= y <= ylim[1]):
                noff += 1
            ax.scatter(r["R"], y, s=60, marker=bmark[r["beta"]], color=bcolor[r["B"]],
                       edgecolors="k", linewidths=0.5, alpha=0.9, zorder=3)
            if key == "gamma_proj" and np.isfinite(r.get("gamma_proj_std", np.nan)):
                ax.errorbar(r["R"], y, yerr=r["gamma_proj_std"], fmt="none",
                            ecolor=bcolor[r["B"]], alpha=0.35, capsize=2, zorder=2)
        if href is not None:
            ax.axhline(href, color="crimson", ls=":", lw=1.2)
        ax.set_xscale("log")
        if ylim:
            ax.set_ylim(*ylim)
            if noff:
                ax.text(0.98, 0.02, f"{noff} off-scale", transform=ax.transAxes,
                        fontsize=7, color="0.4", ha="right", va="bottom")
        ax.set_xlim(xlo, xhi); ax.set_xlabel("R = memory / rotation time")
        ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=0.15)
    axes.flat[0].text(np.sqrt(xlo), axes.flat[0].get_ylim()[1] * 0.96, "MARGINAL",
                      color="#1b6b1b", fontweight="bold", ha="center", va="top", fontsize=10)
    axes.flat[0].text(np.sqrt(xhi), axes.flat[0].get_ylim()[1] * 0.96, "METASTABLE",
                      color="#b35900", fontweight="bold", ha="center", va="top", fontsize=10)

    # 6th panel: (memory, tau_rot) plane colored by GBS, diagonal = R=1
    _plane(fig, axes.flat[5], rows)

    bh = [Line2D([0], [0], marker="o", ls="none", mfc=bcolor[b], mec="k", ms=9, label=f"b{b}")
          for b in batches]
    mh = [Line2D([0], [0], marker=bmark[be], ls="none", mfc="0.4", mec="0.4", ms=8,
                 label=fr"$\beta$={be}") for be in betas]
    fig.legend(handles=bh, title="batch", loc="upper left", bbox_to_anchor=(0.005, 0.995),
               fontsize=8, ncol=len(batches))
    fig.legend(handles=mh, title=r"$\beta$", loc="upper left", bbox_to_anchor=(0.30, 0.995),
               fontsize=8, ncol=len(betas))
    fig.suptitle(f"Regime map — live cells, canonical lr, vs R{title_extra}", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out_path}  ({len(rows)} cells)")
    plt.show() if SHOW else plt.close(fig)
    return fig


def _plane(fig, ax, rows):
    """(memory, tau_rot) plane, color = GBS. R-hypothesis => boundary is the diagonal mem=tau_rot."""
    xs = [r["memory"] for r in rows if np.isfinite(r.get("tau_rot", np.nan))]
    ys = [r["tau_rot"] for r in rows if np.isfinite(r.get("tau_rot", np.nan))]
    cs = [r["gbs"] for r in rows if np.isfinite(r.get("tau_rot", np.nan))]
    if not xs:
        ax.set_visible(False); return
    lo = min(min(xs), min(ys)) * 0.7; hi = max(max(xs), max(ys)) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="memory = $\\tau_{rot}$  (R=1)")
    sc = ax.scatter(xs, ys, c=cs, cmap="RdYlBu", vmin=0, vmax=2.4, s=80, edgecolors="k",
                    linewidths=0.5, zorder=3)
    fig.colorbar(sc, ax=ax, label="GBS", fraction=0.046, pad=0.04)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"state memory  $1/(1-\beta)$"); ax.set_ylabel(r"rotation time  $\tau_{rot}$")
    ax.set_title("is R exactly right?  (boundary = diagonal ⇔ ratio)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, which="both", alpha=0.15)


def fig_appendix(rows, out_path):
    """All valid lrs (not just canonical), colored by lr fraction, to show sub-maximal cells drift
    toward 'not yet at edge'. Two panels: GBS and kappa vs R."""
    if not rows:
        return None
    # lr fraction within each (B,beta)
    mx = {}
    for r in rows:
        k = (r["B"], round(r["beta"], 4)); mx[k] = max(mx.get(k, 0), r["lr"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Rall = np.array([r["R"] for r in rows if np.isfinite(r["R"]) and r["R"] > 0])
    xlo, xhi = Rall.min() * 0.6, Rall.max() * 1.7
    for ax, key, lbl, href in [(axes[0], "gbs", "GBS", 2.0), (axes[1], "kappa", r"$\kappa$", None)]:
        _shade(ax, xlo, xhi)
        for r in rows:
            y = r.get(key)
            if not np.isfinite(y) or not (r["R"] > 0):
                continue
            frac = r["lr"] / (mx[(r["B"], round(r["beta"], 4))] + 1e-30)
            sc = ax.scatter(r["R"], y, c=[frac], cmap="plasma", vmin=0.2, vmax=1.0, s=45,
                            edgecolors="k", linewidths=0.3, zorder=3)
        if href:
            ax.axhline(href, color="crimson", ls=":", lw=1.2)
        ax.set_xscale("log"); ax.set_xlim(xlo, xhi); ax.set_xlabel("R"); ax.set_ylabel(lbl)
        ax.grid(True, which="both", alpha=0.15)
    fig.colorbar(sc, ax=axes, label="lr / lr_max within (B,$\\beta$)", fraction=0.04)
    fig.suptitle("Appendix: all valid lrs (sub-maximal lr → not-yet-at-edge, expected)", fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    plt.show() if SHOW else plt.close(fig)
    return fig


def fig_frozen_cocycle(out_path):
    jp = os.path.join(_REPO, "results", "frozen_cocycle_v3", "frozen_cocycle_v3.json")
    if not os.path.exists(jp):
        print("[plot] no frozen_cocycle_v3.json; skipping"); return None
    cells = json.load(open(jp))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0.0, color="0.3", lw=1); ax.axvline(1.0, color="0.6", ls="--", lw=1, label="c=1")
    for cell, col in zip(cells, plt.cm.plasma(np.linspace(0, 0.85, len(cells)))):
        c = np.array(cell.get("c_grid", []), float); g = np.array(cell.get("gammaK_raw", []), float)
        if len(c) != len(g) or not len(c):
            continue
        ax.plot(c, g, "-o", color=col, ms=4, label=cell.get("tag", "?"))
        idx = np.where(np.diff(np.sign(g)) != 0)[0]
        for i in idx:
            cstar = c[i] - g[i] * (c[i + 1] - c[i]) / (g[i + 1] - g[i])
            ax.plot(cstar, 0, "*", color=col, ms=14, mec="k", mew=0.5, zorder=5)
    ax.set_xlabel("lr multiplier c"); ax.set_ylabel(r"$\gamma_K(c)$")
    ax.set_title("Frozen-cocycle certificate: $\\gamma_K(c^*)=0$ at $c^*\\approx1$ ⇒ marginal\n"
                 "(large-batch = clean; small-batch oscillation-confounded)", fontsize=11)
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(out_path, dpi=140); print(f"[plot] wrote {out_path}")
    plt.show() if SHOW else plt.close(fig)
    return fig


def dump_table(rows, out_path):
    if not rows:
        return
    cols = ["tag", "B", "beta", "lr", "R", "memory", "tau_rot", "act", "gbs", "kappa",
            "cos_buf_uB", "gamma_proj", "hill"]
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: r["R"]):
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[plot] wrote {out_path}")


# %% [setup] — load + classify (run first)
SWEEP_DIR = os.environ.get("EOSS_SWEEP_DIR", os.path.join(_REPO, "results", "comprehensive_sweep"))
OUT_DIR = os.environ.get("EOSS_PLOT_OUT", os.path.join(_REPO, "results", "plots"))
os.makedirs(OUT_DIR, exist_ok=True)
valid, classes = classify(SWEEP_DIR)
print_classification(classes)
canon = canonical(valid)
print(f"[plot] {len(valid)} valid cells -> {len(canon)} canonical (B,beta) operating points")

# %% [Figure 1] — headline regime map (canonical lr, live only)
fig1 = fig_regime_map(canon, os.path.join(OUT_DIR, "regime_map.png"))

# %% [Figure 2] — appendix: all valid lrs colored by lr fraction
fig2 = fig_appendix(valid, os.path.join(OUT_DIR, "regime_map_all_lr.png"))

# %% [Figure 3] — frozen-cocycle certificate
fig3 = fig_frozen_cocycle(os.path.join(OUT_DIR, "frozen_cocycle.png"))

# %% [table]
dump_table(valid, os.path.join(OUT_DIR, "regime_table.csv"))
