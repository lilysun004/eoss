"""
Render bimodality histograms for all SST_opt_batch_sweep runs and place them
under bimodality_sweep/MSE/SSTTransformer/<optimizer>/, mirroring the existing
MSE/CNN and CE/MLP structure. Appends new rows to manifest.csv.

Run from repo root:
    python marc_files/curvature_results/add_sst_to_bimodality_sweep.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/SST_opt_batch_sweep")
BIM_SWEEP   = Path("/n/home06/mwalden/eoss/marc_files/curvature_results/bimodality_sweep")
MANIFEST    = BIM_SWEEP / "manifest.csv"
PYTHON      = "/n/home06/mwalden/.conda/envs/eoss/bin/python"
PLOT_SCRIPT = "/n/home06/mwalden/eoss/plot_histograms.py"

PRECOND_OPTS = {"Adam", "RMSProp"}

def parse_folder(name: str):
    ts   = re.match(r'^(\d{8}_\d{4}_\d{2})_', name)
    opt  = re.search(r'_(SGD(?:-Momentum|-Nesterov)?|Adam|RMSProp|Muon)_', name)
    lr   = re.search(r'_lr([\d.eE+-]+)', name)
    b    = re.search(r'_b(\d+)', name)
    return (
        opt.group(1)  if opt  else None,
        lr.group(1)   if lr   else None,
        b.group(1)    if b    else None,
    )

def config_suffix(folder_name: str) -> str:
    """Build the human-readable config part of the filename."""
    opt, lr, batch = parse_folder(folder_name)
    # grab any trailing _key-val pairs after the batch (beta, momentum, etc.)
    tail = re.sub(r'^.*_b\d+', '', folder_name)  # everything after _b<N>
    return f"{opt}_lr{lr}_b{batch}{tail}"

# Load existing manifest source_folders to skip already-done rows
existing_sources = set()
if MANIFEST.exists():
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            existing_sources.add(row.get("source_folder", ""))

run_folders = sorted(RESULTS_DIR.iterdir())
new_rows = []

for run_dir in run_folders:
    if not run_dir.is_dir():
        continue
    if not (run_dir / "projections.npz").exists():
        print(f"  SKIP (no projections): {run_dir.name}")
        continue

    source_str = str(run_dir)
    if source_str in existing_sources:
        print(f"  SKIP (already in manifest): {run_dir.name}")
        continue

    opt, lr, batch = parse_folder(run_dir.name)
    if not opt:
        print(f"  SKIP (parse failed): {run_dir.name}")
        continue

    dest_dir = BIM_SWEEP / "MSE" / "SSTTransformer" / opt
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix   = config_suffix(run_dir.name)
    bim_png  = dest_dir / f"histograms_{suffix}.png"
    prec_png = dest_dir / f"histograms_{suffix}_precond.png"

    # Render main histogram
    if not bim_png.exists():
        print(f"  Rendering: {bim_png.name}")
        r = subprocess.run(
            [PYTHON, PLOT_SCRIPT, str(run_dir), "--out", str(bim_png)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"    ERROR: {r.stderr[-400:]}")
            continue
    else:
        print(f"  EXISTS: {bim_png.name}")

    # Get max step from results.txt
    max_step = ""
    try:
        import pandas as pd
        df = pd.read_csv(run_dir / "results.txt", comment="#")
        max_step = int(df["step"].max())
    except Exception:
        pass

    new_rows.append({
        "loss": "MSE",
        "arch": "SSTTransformer",
        "optimizer": opt,
        "batch_size": batch,
        "lr": lr,
        "max_step": max_step,
        "source_folder": source_str,
        "curvature_figure": "",
        "bimodality_figure": str(bim_png),
        "bimodality_precond_figure": str(prec_png) if opt in PRECOND_OPTS else "",
        "drift_figure": "",
    })

# Append to manifest
if new_rows:
    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        fieldnames = ["loss","arch","optimizer","batch_size","lr","max_step",
                      "source_folder","curvature_figure","bimodality_figure",
                      "bimodality_precond_figure","drift_figure"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"\nAdded {len(new_rows)} rows to manifest.csv")
else:
    print("\nNo new rows to add.")
