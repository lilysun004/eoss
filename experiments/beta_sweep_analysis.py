"""Beta-sweep analysis (experiment #4 from suggested.txt's ranked list).

Fixes a small batch size and sweeps momentum beta (SGD-Momentum) / beta1 (Adam),
loads the per-step GBS time series each run writes to results.txt, restricts to
the stabilized plateau window (last `--plateau_frac` fraction of recorded
steps), and tabulates mean(GBS) / deficit (2 - mean(GBS)) vs beta.

This directly tests the "Mechanism A" (step-noise coupling) prediction from
instructions.tex / suggested.txt: at beta=0 the deficit should vanish
(GBS -> 2, matching plain SGD, continuously), and the deficit should grow
monotonically as beta -> 1 (more history-averaging, less current-batch
coupling).

Usage:
    python experiments/beta_sweep_analysis.py \
        [--results_dir RESULTS/beta_sweep] [--plateau_frac 0.6]

No existing tracked file is modified; this is a new, standalone file mirroring
the loader conventions in experiments/gbs_distribution_analysis.py.
"""
import argparse
import math
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
    """Load a results.txt (comment lines starting with '#', then a CSV header,
    then CSV rows) into a dict of column_name -> np.ndarray."""
    with open(path) as f:
        lines = f.readlines()
    header = None
    data_start = None
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
    lmax_window = cols["lmax"][mask]
    lmax_valid = lmax_window[~np.isnan(lmax_window)]
    loss_window = cols["batch_loss"][mask]

    meta = parse_run_folder(run_dir.name)
    beta = None
    if meta is not None:
        beta = meta["optimizer_params"].get("beta", meta["optimizer_params"].get("beta1"))

    out = {
        "run_dir": run_dir.name,
        "beta": beta,
        "n_steps_total": len(cols["step"]),
        "n_plateau_rows": len(gbs_window),
        "n_valid_gbs": len(gbs_valid),
        "mean_gbs": float(np.mean(gbs_valid)) if len(gbs_valid) else float("nan"),
        "median_gbs": float(np.median(gbs_valid)) if len(gbs_valid) else float("nan"),
        "std_gbs": float(np.std(gbs_valid)) if len(gbs_valid) else float("nan"),
        "mean_lmax_plateau": float(np.mean(lmax_valid)) if len(lmax_valid) else float("nan"),
        "first_lmax": float(cols["lmax"][~np.isnan(cols["lmax"])][0]) if np.any(~np.isnan(cols["lmax"])) else float("nan"),
        "last_lmax": float(cols["lmax"][~np.isnan(cols["lmax"])][-1]) if np.any(~np.isnan(cols["lmax"])) else float("nan"),
        "mean_loss_plateau": float(np.mean(loss_window)) if len(loss_window) else float("nan"),
        "first_loss": float(cols["batch_loss"][0]),
        "last_loss": float(cols["batch_loss"][-1]),
    }
    out["deficit"] = 2.0 - out["mean_gbs"] if not math.isnan(out["mean_gbs"]) else float("nan")
    return out


def eos_diagnostics(run_dir_name, lr, out):
    """Quick 'did this run actually reach EoS' sanity numbers."""
    beta = out["beta"] or 0.0
    ratio_mom = (lr * out["last_lmax"]) / (2 * (1 + beta)) if out["last_lmax"] == out["last_lmax"] else float("nan")
    lmax_still_climbing = None
    if out["first_lmax"] == out["first_lmax"] and out["last_lmax"] == out["last_lmax"] and out["first_lmax"] > 0:
        lmax_still_climbing = (out["last_lmax"] / out["first_lmax"] - 1.0)
    loss_still_dropping = None
    if out["first_loss"] > 0:
        loss_still_dropping = (out["last_loss"] - out["mean_loss_plateau"])
    return ratio_mom, lmax_still_climbing, loss_still_dropping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=None,
                     help="Folder containing run subdirs (default: $RESULTS/beta_sweep or ./results/beta_sweep)")
    ap.add_argument("--plateau_frac", type=float, default=0.6)
    args = ap.parse_args()

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        import os
        env_results = os.environ.get("RESULTS")
        if env_results:
            results_dir = Path(env_results) / "beta_sweep"
        else:
            results_dir = REPO_ROOT / "results" / "beta_sweep"

    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and (d / "results.txt").exists()])
    if not run_dirs:
        print(f"No run folders with results.txt found under {results_dir}")
        return

    rows = []
    for run_dir in run_dirs:
        out = analyze_run(run_dir, args.plateau_frac)
        if out is None:
            continue
        meta = parse_run_folder(run_dir.name)
        lr = meta["lr"] if meta else float("nan")
        ratio_mom, lmax_climb, loss_drop = eos_diagnostics(run_dir.name, lr, out)
        out["lr"] = lr
        out["ratio_lr_lmax_over_2_1pbeta"] = ratio_mom
        out["lmax_frac_growth_over_run"] = lmax_climb
        out["loss_minus_plateau_mean"] = loss_drop
        rows.append(out)

    rows.sort(key=lambda r: (r["beta"] if r["beta"] is not None else -1))

    print(f"\n{'='*130}")
    print(f"Beta sweep GBS-plateau analysis  (plateau = last {args.plateau_frac:.0%} of steps)")
    print(f"{'='*130}")
    header = (f"{'run':45s} {'beta':>6s} {'lr':>9s} {'mean_GBS':>9s} {'deficit(2-GBS)':>14s} "
              f"{'median_GBS':>10s} {'std_GBS':>8s} {'last_lmax':>10s} {'lr*lmax/2(1+b)':>14s} "
              f"{'lmax_grow%':>10s} {'n_valid':>7s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        beta_s = f"{r['beta']:.2f}" if r['beta'] is not None else "?"
        lmax_grow_s = f"{r['lmax_frac_growth_over_run']*100:.0f}%" if r['lmax_frac_growth_over_run'] is not None else "?"
        print(f"{r['run_dir']:45s} {beta_s:>6s} {r['lr']:>9.5f} {r['mean_gbs']:>9.4f} "
              f"{r['deficit']:>14.4f} {r['median_gbs']:>10.4f} {r['std_gbs']:>8.4f} "
              f"{r['last_lmax']:>10.2f} {r['ratio_lr_lmax_over_2_1pbeta']:>14.3f} "
              f"{lmax_grow_s:>10s} {r['n_valid_gbs']:>7d}")
    print(f"{'='*130}\n")

    # Monotonicity / shape check on beta -> deficit
    valid_rows = [r for r in rows if r["beta"] is not None and not math.isnan(r["deficit"])]
    if len(valid_rows) >= 2:
        betas = np.array([r["beta"] for r in valid_rows])
        deficits = np.array([r["deficit"] for r in valid_rows])
        order = np.argsort(betas)
        betas, deficits = betas[order], deficits[order]
        diffs = np.diff(deficits)
        monotonic = bool(np.all(diffs >= -1e-9))
        print(f"betas:    {betas}")
        print(f"deficits: {np.round(deficits, 4)}")
        print(f"monotonic non-decreasing in beta: {monotonic}")
        # Fits: linear in beta, linear in (1-(1-beta)) forms, quadratic
        try:
            lin_beta = np.polyfit(betas, deficits, 1)
            resid_lin = deficits - np.polyval(lin_beta, betas)
            print(f"linear fit deficit ~ a*beta + b: a={lin_beta[0]:.4f} b={lin_beta[1]:.4f} "
                  f"ss_resid={np.sum(resid_lin**2):.5f}")
        except Exception as e:
            print(f"linear fit failed: {e}")
        # deficit ~ c * beta/(1-beta) (diverges as beta->1) -- only for beta<1
        mask_lt1 = betas < 0.999
        if np.sum(mask_lt1) >= 2:
            x = betas[mask_lt1] / (1 - betas[mask_lt1])
            y = deficits[mask_lt1]
            c = np.sum(x * y) / np.sum(x * x) if np.sum(x * x) > 0 else float("nan")
            resid = y - c * x
            print(f"fit deficit ~ c*beta/(1-beta): c={c:.5f} ss_resid={np.sum(resid**2):.5f}")


if __name__ == "__main__":
    main()
