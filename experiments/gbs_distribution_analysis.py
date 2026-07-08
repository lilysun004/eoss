"""Step-level GBS_t distribution analysis (experiment #2 from suggested.txt's ranked
list) plus the (sigma^2, mu)-plane stretch goal from Direction 2's "top pick".

For every run folder under RESULTS/gbs_distributions/<run>/results.txt this script:
  1. Parses the run's (optimizer_name, optimizer_params, lr, batch_size) from the
     folder name (see utils/storage.py::initialize_folders / parse_folder_name for
     the naming convention: YYYYMMDD_HHMM_SS_<opt>_lr<lr>_b<batch>[_<param>-<val>...]).
  2. Loads results.txt (2 leading '#' comment lines, then a CSV header row -- see
     utils/storage.py::RESULTS_COLUMNS / get_welcome_string).
  3. Restricts to a "plateau" window (default: last `--plateau_frac` fraction of
     recorded steps) where the EoS ratio lr*lmax/2 has presumably stopped trending,
     and extracts the GBS column, dropping NaNs (rows where the probe-GBS
     measurement wasn't taken that step).
  4. Computes, on that windowed GBS_t series:
       - mean(GBS_t), median(GBS_t)
       - geometric mean = exp(mean(log(GBS_t))), restricted to GBS_t > 0
         (non-positive values are reported as a count and excluded, since log is
         undefined there -- this treatment is noted in the printed table)
       - mu = mean(log(GBS_t)), sigma2 = var(log(GBS_t), ddof=1)   [same positive
         subset as the geometric mean]
  5. Prints one row per (optimizer, batch) cell, plus a plain-text/CSV dump under
     RESULTS/gbs_distributions/analysis/.
  6. STRETCH GOAL: plots every (optimizer, batch) cell as a point (sigma2, mu) in
     the log-GBS plane [treating GBS_t as a cheap, imperfect proxy for the true
     per-step growth factor a_t -- see suggested.txt Direction 2, which explicitly
     flags this substitution as an acceptable starting point, not the more
     principled tangent-propagation a_t], and overlays the three candidate
     stability lines:
         mu = 0            (a.s. / log-stability)
         mu = -sigma2/2    (mean stability -- log-normal mean = 1 <=> GBS-type mean = 2 baseline)
         mu = -sigma2      (mean-square stability)

Usage:
    python experiments/gbs_distribution_analysis.py \
        [--results_dir RESULTS/gbs_distributions] [--plateau_frac 0.5]

No existing tracked files are modified; this is a new, standalone file.
"""
import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_COLUMNS = [
    "epoch", "step", "batch_loss", "full_loss",
    "lmax", "step_sharpness", "batch_sharpness",
    "A", "B", "GBS", "SBS",
    "A_u", "B_u", "GBS_u",
    "A_g", "B_g", "GBS_g",
    "A_gfull", "B_gfull", "GBS_gfull",
    "A_ufull", "B_ufull", "GBS_ufull",
    "A_cos_sBgB", "GBS_cos_sBgB",
    "A_cos_sBgfull", "GBS_cos_sBgfull",
    "A_cos_gBgfull", "GBS_cos_gBgfull",
    "full_accuracy",
]

FOLDER_RE = re.compile(
    r"^\d{8}_\d{4}_\d{2}_(?P<opt>[A-Za-z0-9\-]+)_lr(?P<lr>[^_]+)_b(?P<batch>\d+)(?P<rest>.*)$"
)


def parse_run_folder(name):
    """Parse optimizer/lr/batch (and any extra optimizer params) out of a run
    folder name, following the convention documented in utils/storage.py."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    opt = m.group("opt")
    lr = float(m.group("lr"))
    batch = int(m.group("batch"))
    rest = m.group("rest")
    params = {}
    for part in rest.split("_"):
        if not part:
            continue
        if "-" in part:
            k, v = part.split("-", 1)
            try:
                v = float(v)
            except ValueError:
                pass
            params[k] = v
    return {"optimizer_name": opt, "lr": lr, "batch_size": batch, "optimizer_params": params}


def load_results_txt(path):
    """Load a results.txt (2 comment lines + CSV header + CSV rows) into a dict
    of column_name -> np.ndarray (float, NaN for missing/blank)."""
    with open(path) as f:
        lines = f.readlines()
    # Skip '#'-prefixed comment lines to find the header row.
    data_start = None
    header = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            continue
        header = line.strip().split(",")
        data_start = i + 1
        break
    if header is None:
        return None
    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        rows.append(parts)
    if not rows:
        return None
    arr = np.array(rows, dtype=object)
    cols = {}
    for j, name in enumerate(header):
        col = arr[:, j]
        vals = np.full(len(col), np.nan)
        for i, v in enumerate(col):
            try:
                vals[i] = float(v)
            except (ValueError, TypeError):
                vals[i] = np.nan
        cols[name] = vals
    return cols


def plateau_window(cols, plateau_frac):
    """Return boolean mask selecting the last `plateau_frac` fraction of rows
    (by row index / step ordering), used as the "stabilized" window."""
    n = len(cols["step"])
    start_idx = int(math.floor(n * (1 - plateau_frac)))
    mask = np.zeros(n, dtype=bool)
    mask[start_idx:] = True
    return mask


def analyze_run(run_dir, plateau_frac):
    results_path = run_dir / "results.txt"
    if not results_path.exists():
        return None
    cols = load_results_txt(results_path)
    if cols is None or "GBS" not in cols:
        return None

    mask = plateau_window(cols, plateau_frac)
    gbs_window = cols["GBS"][mask]
    gbs_valid = gbs_window[~np.isnan(gbs_window)]

    n_total = len(gbs_window)
    n_valid = len(gbs_valid)
    n_nonpos = int(np.sum(gbs_valid <= 0)) if n_valid else 0
    gbs_pos = gbs_valid[gbs_valid > 0]

    out = {
        "run_dir": run_dir.name,
        "n_plateau_rows": n_total,
        "n_valid_gbs": n_valid,
        "n_nonpositive_gbs": n_nonpos,
        "n_used_for_log": len(gbs_pos),
    }

    if n_valid == 0:
        out.update({k: float("nan") for k in
                    ["mean_gbs", "median_gbs", "geomean_gbs", "mu_log_gbs", "sigma2_log_gbs"]})
        return out

    out["mean_gbs"] = float(np.mean(gbs_valid))
    out["median_gbs"] = float(np.median(gbs_valid))

    if len(gbs_pos) >= 2:
        log_gbs = np.log(gbs_pos)
        mu = float(np.mean(log_gbs))
        sigma2 = float(np.var(log_gbs, ddof=1))
        out["mu_log_gbs"] = mu
        out["sigma2_log_gbs"] = sigma2
        out["geomean_gbs"] = float(math.exp(mu))
    elif len(gbs_pos) == 1:
        out["mu_log_gbs"] = float(np.log(gbs_pos[0]))
        out["sigma2_log_gbs"] = float("nan")
        out["geomean_gbs"] = float(gbs_pos[0])
    else:
        out["mu_log_gbs"] = float("nan")
        out["sigma2_log_gbs"] = float("nan")
        out["geomean_gbs"] = float("nan")

    # EoS sanity check: mean lr*lmax/2 over the plateau window (should be ~1 if
    # at EoS -- flagged separately in the report, not required to equal 1 for
    # every optimizer since some stabilize below/above depending on batch).
    lmax_window = cols["lmax"][mask]
    lr = None  # filled in by caller from folder parse; ratio computed there instead
    out["mean_lmax_plateau"] = float(np.nanmean(lmax_window)) if len(lmax_window) else float("nan")
    out["full_loss_first_half_mean"] = float(np.nanmean(cols["full_loss"][~mask])) if np.any(~mask) else float("nan")
    out["full_loss_plateau_mean"] = float(np.nanmean(cols["full_loss"][mask])) if len(cols["full_loss"][mask]) else float("nan")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str,
                     default=str(REPO_ROOT / "results" / "gbs_distributions"))
    ap.add_argument("--plateau_frac", type=float, default=0.5,
                     help="Fraction of the tail of the run (by recorded rows) "
                          "considered the stabilized/plateau window.")
    ap.add_argument("--out_dir", type=str,
                     default=str(REPO_ROOT / "results" / "gbs_distributions" / "analysis"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in results_dir.iterdir()
                        if p.is_dir() and p.name not in ("logs", "analysis")])

    rows = []
    for run_dir in run_dirs:
        parsed = parse_run_folder(run_dir.name)
        if parsed is None:
            print(f"[skip] could not parse folder name: {run_dir.name}")
            continue
        # Prefer args.json if present, for an authoritative optimizer_params/lr.
        args_json_path = run_dir / "args.json"
        if args_json_path.exists():
            try:
                with open(args_json_path) as f:
                    saved_args = json.load(f)
                parsed["lr"] = float(saved_args.get("lr", parsed["lr"]))
                parsed["batch_size"] = int(saved_args.get("batch_size", parsed["batch_size"]))
                parsed["optimizer_params"] = saved_args.get("optimizer_params", parsed["optimizer_params"])
            except Exception:
                pass

        stats = analyze_run(run_dir, args.plateau_frac)
        if stats is None:
            print(f"[skip] no usable results.txt / GBS column: {run_dir.name}")
            continue

        ratio = float("nan")
        if not math.isnan(stats.get("mean_lmax_plateau", float("nan"))):
            ratio = parsed["lr"] * stats["mean_lmax_plateau"] / 2.0

        row = {
            "optimizer": parsed["optimizer_name"],
            "batch_size": parsed["batch_size"],
            "lr": parsed["lr"],
            "optimizer_params": parsed["optimizer_params"],
            **stats,
            "mean_lr_lmax_over_2_plateau": ratio,
        }
        rows.append(row)

    # Sort for a nice report: optimizer, then batch size ascending.
    opt_order = {"SGD": 0, "SGD-Momentum": 1, "Adam": 2, "Muon": 3}
    rows.sort(key=lambda r: (opt_order.get(r["optimizer"], 99), r["batch_size"]))

    # ---- Write CSV ----
    csv_path = out_dir / "gbs_distribution_table.csv"
    fieldnames = ["optimizer", "batch_size", "lr", "optimizer_params",
                  "n_plateau_rows", "n_valid_gbs", "n_nonpositive_gbs", "n_used_for_log",
                  "mean_gbs", "median_gbs", "geomean_gbs",
                  "mu_log_gbs", "sigma2_log_gbs",
                  "mean_lr_lmax_over_2_plateau",
                  "full_loss_first_half_mean", "full_loss_plateau_mean",
                  "run_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # ---- Print table ----
    print("\n" + "=" * 130)
    print("GBS_t plateau-window distribution table  (plateau_frac=%.2f)" % args.plateau_frac)
    print("=" * 130)
    hdr = f"{'optimizer':14s} {'batch':>6s} {'lr':>10s} {'n':>4s} {'mean':>8s} {'median':>8s} {'geomean':>8s} {'mu(logGBS)':>11s} {'sigma2':>8s} {'lr*lmax/2':>10s}"
    print(hdr)
    print("-" * 130)
    for r in rows:
        print(f"{r['optimizer']:14s} {r['batch_size']:6d} {r['lr']:10.6g} "
              f"{r['n_valid_gbs']:4d} {r['mean_gbs']:8.3f} {r['median_gbs']:8.3f} "
              f"{r['geomean_gbs']:8.3f} {r['mu_log_gbs']:11.4f} {r['sigma2_log_gbs']:8.4f} "
              f"{r['mean_lr_lmax_over_2_plateau']:10.3f}   [{r['run_dir']}]")
    print("=" * 130)
    print(f"Full CSV written to {csv_path}")

    # ---- Stretch goal: (sigma2, mu) plane plot ----
    try:
        make_sigma_mu_plot(rows, out_dir)
    except Exception as e:
        print(f"[stretch goal plot skipped: {e}]")


def make_sigma_mu_plot(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"SGD": "tab:blue", "SGD-Momentum": "tab:orange",
              "Adam": "tab:green", "Muon": "tab:red"}
    markers = {8: "o", 128: "s", 2048: "^"}

    sigma2_max = 0.0
    for r in rows:
        s2 = r.get("sigma2_log_gbs", float("nan"))
        mu = r.get("mu_log_gbs", float("nan"))
        if math.isnan(s2) or math.isnan(mu):
            continue
        sigma2_max = max(sigma2_max, s2)
        c = colors.get(r["optimizer"], "gray")
        m = markers.get(r["batch_size"], "x")
        ax.scatter(s2, mu, color=c, marker=m, s=90, edgecolor="black", linewidth=0.5,
                   label=f"{r['optimizer']} b={r['batch_size']}")

    sigma2_max = max(sigma2_max, 0.05) * 1.2
    xs = np.linspace(0, sigma2_max, 100)
    ax.plot(xs, 0 * xs, "k--", label="mu=0 (a.s. / log-stability)")
    ax.plot(xs, -xs / 2, "k-.", label="mu=-sigma2/2 (mean stability)")
    ax.plot(xs, -xs, "k:", label="mu=-sigma2 (mean-square stability)")

    ax.set_xlabel("sigma^2 = Var[log GBS_t]  (plateau window)")
    ax.set_ylabel("mu = E[log GBS_t]  (plateau window)")
    ax.set_title("(sigma^2, mu) plane: GBS_t as proxy for growth factor a_t\n"
                  "(GBS_t is NOT the principled tangent-propagated a_t -- see suggested.txt Direction 2)")
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)

    # De-duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, fontsize=7, loc="best")

    fig.tight_layout()
    plot_path = out_dir / "sigma2_mu_plane.png"
    fig.savefig(plot_path, dpi=150)
    print(f"(sigma^2, mu)-plane plot written to {plot_path}")


if __name__ == "__main__":
    main()
