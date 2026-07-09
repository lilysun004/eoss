"""
Regime map: everything parameterized by R.

The organizing claim of this project is that a single dimensionless control parameter
    R = (optimizer state-memory 1/(1-beta)) / (unstable-direction rotation time tau_rot)
decides whether a cell is MARGINAL (at its edge, R<~1) or METASTABLE (sub-edge damped
basin, R>>1). If that's right, then every per-cell diagnostic should collapse onto a
function of R alone -- regardless of whether R was moved by changing batch size or beta.

This script reads the comprehensive SGD-Momentum sweep (results/comprehensive_sweep/<cell>/
meta.json, one cell per (batch, beta, lr)) and scatters each diagnostic against R, with the
marginal/metastable regions shaded and the R=1 crossover marked. Color = batch, marker = beta,
so you can eyeball the collapse: do points at the same R land at the same y no matter how they
got there?

Figure 1 (regime_map.png): 3x3 grid of diagnostics vs R.
Figure 2 (frozen_cocycle.png): the frozen-cocycle marginality certificate gamma_K(c) -- where
    the closed-loop Lyapunov crosses zero (c*~1 <=> marginal). Separate, sparser run; small-batch
    cells are oscillation-confounded (gamma<0 even for marginal SGD_b8), so this is a large-batch
    certificate, complementary to the per-cell perturb-relax gamma in Figure 1.

Usage:
    - Inline (Cursor/VS Code): run the `# %%` cells below one at a time; each figure is
      displayed with plt.show() AND saved to results/plots/.
    - Batch: `python plot.py` runs top-to-bottom, saves the PNGs; set EOSS_NO_SHOW=1 to
      force the headless Agg backend and skip the (blocking) window pop-ups.
    - Override dirs with env vars EOSS_SWEEP_DIR / EOSS_PLOT_OUT.
"""

#%%
import os, json, glob
import numpy as np
import matplotlib
# Only force the non-interactive backend when explicitly headless; otherwise keep whatever
# interactive backend the inline (#%%) kernel provides so plt.show() actually renders.
SHOW = os.environ.get("EOSS_NO_SHOW") != "1"
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO = os.path.dirname(os.path.abspath(__file__))


def _f(x):
    """None/missing -> nan, else float."""
    try:
        return float(x) if x is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_cells(sweep_dir, require_plateau=True):
    """Read every cell's meta.json into a list of flat dicts.

    VALIDITY GATE: a cell's diagnostics are only meaningful if the run reached a stationary
    plateau (whatever value). We drop (a) diverged cells and (b) cells whose GBS trajectory
    never stabilized -- typically the lowest-lr cells that burned the full step budget still
    descending toward EoS. Set require_plateau=False to inspect the rejects. Returns
    (rows, excluded) where `excluded` lists (tag, reason) for transparency.
    """
    rows, excluded = [], []
    for meta_path in sorted(glob.glob(os.path.join(sweep_dir, "b*_beta*", "meta.json"))):
        try:
            m = json.load(open(meta_path))
        except (json.JSONDecodeError, OSError):
            continue
        tag = m.get("tag")
        if m.get("diverged"):
            excluded.append((tag, "diverged")); continue
        st = m.get("stationarity") or {}
        stabilized = bool(st.get("stabilized"))
        R = _f(m.get("plateau_R"))
        if not np.isfinite(R) or R <= 0:
            excluded.append((tag, "no R")); continue
        if require_plateau and not stabilized:
            excluded.append((tag, f"GBS not plateaued (drift={_f(st.get('drift_over_window')):.3f}, "
                                  f"{m.get('steps_trained')}/{m.get('steps_max')} steps)"))
            continue
        # Validity = GBS reached a stationary plateau (stabilized), at WHATEVER value. Loss is
        # irrelevant: at EoS the loss keeps decreasing while lambda/GBS plateaus, so loss->0 is
        # normal, not a disqualifier. `stationarity` in meta is computed on the GBS trajectory
        # (plateau_mean == plateau_gbs), so `stabilized` is exactly the right gate.
        rows.append(dict(
            tag=tag, B=int(m["B"]), beta=float(m["beta"]),
            lr=_f(m.get("lr")), lr_index=m.get("lr_index"),
            R=R, stabilized=stabilized,
            drift=_f(st.get("drift_over_window")),
            gbs=_f(m.get("plateau_gbs")),
            kappa=_f(m.get("plateau_kappa")),
            cos_buf_uB=_f(m.get("plateau_cos_buf_uB")),
            gamma_proj=_f(m.get("perturb_gamma_proj_mean")),
            gamma_proj_std=_f(m.get("perturb_gamma_proj_std")),
            gamma_full=_f(m.get("perturb_gamma_full_mean")),
            sharpen=_f(m.get("sharpen_net_rise_mean")),
            sharpen_std=_f(m.get("sharpen_net_rise_std")),
            kurt=_f(m.get("catapult_kurtosis")),
            p99_p50=_f(m.get("catapult_p99_p50")),
            hill=_f(m.get("catapult_hill")),
        ))
    return rows, excluded


# (key, y-label, horizontal reference line or None, yscale, ylim)
PANELS = [
    ("gbs",        "GBS  (edge = 2)",                    2.0,  "linear", (-0.5, 3.2)),
    ("kappa",      r"$\kappa=\eta\lambda$  (edge frac.)", None, "linear", None),
    ("cos_buf_uB", r"cos(buffer, $u_B$)  (tracking)",    None, "linear", None),
    ("gamma_proj", r"$\gamma_{relax}$ proj  (kick decay)", 0.0, "linear", None),
    ("gamma_full", r"$\gamma_{relax}$ full-space",        0.0, "linear", None),
    ("sharpen",    "sharpening net rise\n(boundary binding?)", 0.0, "linear", None),
    ("kurt",       "catapult kurtosis",                  None, "symlog", None),
    ("p99_p50",    "catapult p99/p50",                   None, "symlog", None),
    ("hill",       r"catapult Hill $\alpha$ (m.s. bd = 2)", 2.0, "linear", None),
]

BETA_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def shade_regimes(ax, xlo, xhi):
    """Green = marginal (R<1), orange = metastable (R>1), dashed line at R=1."""
    ax.axvspan(xlo, 1.0, color="#2ca02c", alpha=0.07, zorder=0)
    ax.axvspan(1.0, xhi, color="#ff7f0e", alpha=0.07, zorder=0)
    ax.axvline(1.0, color="0.4", ls="--", lw=1, zorder=1)


def fig_regime_map(rows, out_path):
    if not rows:
        print("[plot] no cells to plot yet"); return
    batches = sorted({r["B"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(batches)))
    bcolor = {b: cmap[i] for i, b in enumerate(batches)}
    bmarker = {be: BETA_MARKERS[i % len(BETA_MARKERS)] for i, be in enumerate(betas)}

    Rall = np.array([r["R"] for r in rows])
    xlo, xhi = Rall.min() * 0.6, Rall.max() * 1.7

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (key, ylabel, href, yscale, ylim) in zip(axes.flat, PANELS):
        shade_regimes(ax, xlo, xhi)
        n_clip = 0
        for r in rows:
            y = r[key]
            if not np.isfinite(y):
                continue
            if ylim is not None and not (ylim[0] <= y <= ylim[1]):
                n_clip += 1  # counted, drawn clipped at the axis edge below
            ax.scatter(r["R"], y, s=46, marker=bmarker[r["beta"]],
                       facecolors=bcolor[r["B"]], edgecolors=bcolor[r["B"]], linewidths=1.3,
                       alpha=0.9, zorder=3)
            if key == "gamma_proj" and np.isfinite(r["gamma_proj_std"]):
                ax.errorbar(r["R"], y, yerr=r["gamma_proj_std"], fmt="none",
                            ecolor=bcolor[r["B"]], alpha=0.35, capsize=2, zorder=2)
        if href is not None:
            ax.axhline(href, color="crimson", ls=":", lw=1.2, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlim(xlo, xhi)
        if ylim is not None:
            ax.set_ylim(*ylim)
            if n_clip:
                ax.text(0.98, 0.02, f"{n_clip} off-scale", transform=ax.transAxes,
                        fontsize=7, color="0.4", ha="right", va="bottom")
        ax.set_xlabel("R  =  buffer memory / rotation time")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.15)

    # regime labels on the first panel
    a0 = axes.flat[0]
    ylim = a0.get_ylim()
    yt = ylim[1] - 0.06 * (ylim[1] - ylim[0])
    a0.text(np.sqrt(xlo * 1.0), yt, "MARGINAL", color="#1b6b1b", fontsize=10,
            fontweight="bold", ha="center", va="top")
    a0.text(np.sqrt(1.0 * xhi), yt, "METASTABLE", color="#b35900", fontsize=10,
            fontweight="bold", ha="center", va="top")

    # legends: batch (color) + beta (marker)
    batch_handles = [Line2D([0], [0], marker="o", ls="none", mfc=bcolor[b], mec=bcolor[b],
                            ms=9, label=f"b{b}") for b in batches]
    beta_handles = [Line2D([0], [0], marker=bmarker[be], ls="none", mfc="0.35", mec="0.35",
                           ms=8, label=fr"$\beta$={be}") for be in betas]
    leg1 = fig.legend(handles=batch_handles, title="batch (color)", loc="upper left",
                      bbox_to_anchor=(0.005, 0.995), fontsize=9, framealpha=0.9)
    fig.add_artist(leg1)
    fig.legend(handles=beta_handles, title=r"$\beta$ (marker)", loc="upper right",
               bbox_to_anchor=(0.995, 0.995), fontsize=9, framealpha=0.9)

    fig.suptitle("Regime map — every diagnostic vs the control parameter R\n"
                 f"({len(rows)} cells; color=batch, marker=$\\beta$; collapse onto R "
                 "⇒ R is the control parameter)", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out_path}  ({len(rows)} cells)")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return fig


def fig_frozen_cocycle(out_path):
    """Companion: the frozen-cocycle certificate gamma_K(c). c*~1 (zero crossing) <=> marginal."""
    jp = os.path.join(_REPO, "results", "frozen_cocycle_v3", "frozen_cocycle_v3.json")
    if not os.path.exists(jp):
        print("[plot] no frozen_cocycle_v3.json; skipping Fig 2"); return
    cells = json.load(open(jp))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0.0, color="0.3", lw=1)
    ax.axvline(1.0, color="0.6", ls="--", lw=1, label="c=1 (operating point)")
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(cells)))
    for cell, col in zip(cells, colors):
        c = np.array(cell.get("c_grid", []), float)
        g = np.array(cell.get("gammaK_raw", []), float)
        if len(c) != len(g) or len(c) == 0:
            continue
        ax.plot(c, g, "-o", color=col, ms=4, label=cell.get("tag", "?"))
        # mark zero-crossing c* if one exists
        s = np.sign(g)
        idx = np.where(np.diff(s) != 0)[0]
        for i in idx:
            cstar = c[i] - g[i] * (c[i + 1] - c[i]) / (g[i + 1] - g[i])
            ax.plot(cstar, 0, "*", color=col, ms=14, mec="k", mew=0.5, zorder=5)
    ax.set_xlabel("lr multiplier  c")
    ax.set_ylabel(r"frozen-cocycle top Lyapunov  $\gamma_K(c)$")
    ax.set_title("Marginality certificate: $\\gamma_K(c^\\ast)=0$ at $c^\\ast\\!\\approx\\!1$ ⇒ marginal\n"
                 "(★ = zero crossing; small-batch cells oscillation-confounded, "
                 "large-batch cells are the clean certificate)", fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[plot] wrote {out_path}  ({len(cells)} frozen-cocycle cells)")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return fig


def dump_table(rows, out_path):
    """CSV of the aggregated per-cell scalars, sorted by R, for ad-hoc analysis."""
    if not rows:
        return
    cols = ["tag", "B", "beta", "lr", "R", "stabilized", "drift", "gbs", "kappa", "cos_buf_uB",
            "gamma_proj", "gamma_full", "sharpen", "kurt", "p99_p50", "hill"]
    rows = sorted(rows, key=lambda r: r["R"])
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[plot] wrote {out_path}")


# %% [setup] — run this cell first (loads the sweep)
SWEEP_DIR = os.environ.get("EOSS_SWEEP_DIR", os.path.join(_REPO, "results", "comprehensive_sweep"))
OUT_DIR = os.environ.get("EOSS_PLOT_OUT", os.path.join(_REPO, "results", "plots"))
os.makedirs(OUT_DIR, exist_ok=True)
rows, excluded = load_cells(SWEEP_DIR)   # require_plateau=True: only GBS-stationary runs are valid
print(f"[plot] {len(rows)} valid (GBS-plateaued) cells | {len(excluded)} excluded")
_reasons = {}
for _tag, _why in excluded:
    _reasons.setdefault(_why.split(" (")[0], []).append(_tag)
for _why, _tags in sorted(_reasons.items(), key=lambda kv: -len(kv[1])):
    print(f"    excluded [{_why}]: {len(_tags)}  e.g. {', '.join(_tags[:4])}")

# %% [Figure 1] — regime map: every diagnostic vs R (displays inline + saves PNG)
fig1 = fig_regime_map(rows, os.path.join(OUT_DIR, "regime_map.png"))

# %% [Figure 2] — frozen-cocycle marginality certificate (displays inline + saves PNG)
fig2 = fig_frozen_cocycle(os.path.join(OUT_DIR, "frozen_cocycle.png"))

# %% [table] — dump the per-cell scalars to CSV
dump_table(rows, os.path.join(OUT_DIR, "regime_table.csv"))
