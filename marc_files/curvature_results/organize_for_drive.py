#!/usr/bin/env python
"""Organize curvature-failure runs into clean trees for sharing (Google Drive).

Three SEPARATE trees, each mirroring marc_files/results_histograms/CNN_sweep
but with a top-level split between MSE and CE loss:

    curvature_failure_sweep/MSE|CE/<arch>/<optimizer>/curvature_<config>.png
    bimodality_sweep/       MSE|CE/<arch>/<optimizer>/histograms_<config>.png
    drift_sweep/            MSE|CE/<arch>/<optimizer>/tangent_drift_<config>.png

All three trees are rendered from the SAME deduplicated run folders, so a
curvature plot, its bimodality plot, and its tangent-drift plot live at the
identical relative path (loss/arch/optimizer/<config>) and the shared manifest
records the exact source run — you can always pair the three plot types for the
very same run. The trees are kept separate (not crammed together). The
canonical standalone drift study still lives in marc_files/drift_results/.

For each (loss, arch, optimizer, batch) cell we keep only the latest-timestamp
run folder (kempner_requeue preemption leaves duplicates).

Run from anywhere:  python marc_files/curvature_results/organize_for_drive.py
"""
import csv
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # /n/home06/mwalden/eoss
OUT = HERE / "curvature_failure_sweep"
OUT_BIMODAL = HERE / "bimodality_sweep"
OUT_DRIFT = HERE / "drift_sweep"
DRIFT_SCRIPT = REPO / "marc_files" / "drift_results" / "plot_tangent_drift.py"

# (loss, arch, source-subfolder list). All curvature runs with a real cell live
# in these folders; CE probe folders without curvature data are ignored.
SOURCES = [
    ("MSE", "CNN", ["curvature_failure_cnn"]),
    ("MSE", "MLP", ["curvature_failure_mlp"]),
    ("MSE", "ViT", ["curvature_failure_vit"]),
    ("CE",  "MLP", ["curvature_failure_mlp_ce_bsweep"]),
]

TS_RE = re.compile(r"^(\d{8}_\d{4}_\d{2})_(.+)$")  # timestamp_, config


def load_make_plot():
    spec = importlib.util.spec_from_file_location(
        "plot_curvature_failure", HERE / "plot_curvature_failure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_plot


def has_curv(run: Path) -> bool:
    return (run / "curvature_segment.npz").exists() or \
           (run / "data" / "curvature_segment.npz").exists()


def optimizer_of(config: str) -> str:
    return config.split("_lr")[0]


def max_step(run: Path) -> str:
    res = run / "results.txt"
    if not res.exists():
        res = run / "data" / "results.txt"
    if not res.exists():
        return ""
    last = ""
    with open(res) as f:
        for line in f:
            if line.startswith("#") or line.startswith("epoch"):
                continue
            last = line
    if not last:
        return ""
    parts = last.split(",")
    return parts[1] if len(parts) > 1 else ""


def render_histograms(run: Path, dest_main: Path) -> Path | None:
    """Render bimodality histograms for `run` to dest_main via plot_histograms.py.

    Returns the precond figure path if one was produced, else None.
    """
    dest_main.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(REPO / "plot_histograms.py"),
         str(run), "--out", str(dest_main)],
        check=True, cwd=str(REPO),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    precond = dest_main.with_name(dest_main.stem + "_precond" + dest_main.suffix)
    return precond if precond.exists() else None


def render_drift(run: Path, dest: Path) -> None:
    """Render the tangent-drift plot for `run` to dest via plot_tangent_drift.py."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(DRIFT_SCRIPT), str(run), "--out", str(dest)],
        check=True, cwd=str(REPO),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    make_plot = load_make_plot()
    # Idempotent: curvature PNGs are reused if already rendered in the run
    # folder; bimodality PNGs are skipped if already present in the tree. This
    # makes reruns cheap (just copies) and never re-renders existing figures.
    rows = []

    for loss, arch, subfolders in SOURCES:
        # cell -> (timestamp, run_path); keep the latest timestamp per config.
        latest: dict[str, tuple[str, Path]] = {}
        for sub in subfolders:
            for run in sorted((HERE / sub).glob("*/")):
                if not has_curv(run):
                    continue
                m = TS_RE.match(run.name)
                if not m:
                    continue
                ts, config = m.group(1), m.group(2)
                if config not in latest or ts > latest[config][0]:
                    latest[config] = (ts, run)

        for config, (ts, run) in sorted(latest.items()):
            opt = optimizer_of(config)
            batch = re.search(r"_b(\d+)", config)
            batch = batch.group(1) if batch else ""
            lr = re.search(r"_lr([\d.eE+-]+?)_b", config)
            lr = lr.group(1) if lr else ""

            # Curvature: reuse the PNG already rendered in the run folder; only
            # render if it's missing. Then copy into the clean tree.
            png_src = run / "curvature_failure.png"
            if not png_src.exists():
                try:
                    make_plot(run)
                except SystemExit as e:
                    print(f"  SKIP {run.name}: {e}")
                    continue

            dest_dir = OUT / loss / arch / opt
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"curvature_{config}.png"
            shutil.copy2(png_src, dest)

            # Same run -> bimodality histogram at the mirrored path. Skip the
            # (slow) render if it already exists in the tree.
            bim_dest = OUT_BIMODAL / loss / arch / opt / f"histograms_{config}.png"
            bim_precond = bim_dest.with_name(bim_dest.stem + "_precond" + bim_dest.suffix)
            if bim_dest.exists():
                bim_precond = bim_precond if bim_precond.exists() else None
            else:
                try:
                    bim_precond = render_histograms(run, bim_dest)
                except subprocess.CalledProcessError as e:
                    print(f"  WARN histogram failed for {run.name}: {e}")
                    bim_dest, bim_precond = None, None

            # Same run -> tangent-drift plot at the mirrored path.
            drift_dest = OUT_DRIFT / loss / arch / opt / f"tangent_drift_{config}.png"
            if not drift_dest.exists():
                try:
                    render_drift(run, drift_dest)
                except subprocess.CalledProcessError as e:
                    print(f"  WARN drift failed for {run.name}: {e}")
                    drift_dest = None

            rows.append({
                "loss": loss, "arch": arch, "optimizer": opt,
                "batch_size": batch, "lr": lr, "max_step": max_step(run),
                "source_folder": str(run),
                "curvature_figure": str(dest),
                "bimodality_figure": str(bim_dest) if bim_dest else "",
                "bimodality_precond_figure": str(bim_precond) if bim_precond else "",
                "drift_figure": str(drift_dest) if drift_dest else "",
            })
            print(f"  {loss}/{arch}/{opt}/{config}")

    # Same manifest in both trees so each is self-contained for upload; the
    # source_folder column lets you pair a curvature plot with the bimodality
    # plot of the very same run.
    fields = ["loss", "arch", "optimizer", "batch_size", "lr", "max_step",
              "source_folder", "curvature_figure", "bimodality_figure",
              "bimodality_precond_figure", "drift_figure"]
    for tree in (OUT, OUT_BIMODAL, OUT_DRIFT):
        tree.mkdir(parents=True, exist_ok=True)
        with open(tree / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print(f"\nWrote {len(rows)} cells.")
    print(f"  curvature : {OUT}")
    print(f"  bimodality: {OUT_BIMODAL}")
    print(f"  drift     : {OUT_DRIFT}")


if __name__ == "__main__":
    main()
