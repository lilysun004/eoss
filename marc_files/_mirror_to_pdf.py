"""One-off helper: regenerate every PNG under results_histograms/{CNN_sweep,vit_sweep}/
as a PDF under results_histograms_pdf/, using the new plot_histograms.py.

For each main PNG (skipping the auto-generated _precond.png siblings), the script:
  1. parses (optimizer, lr, batch, beta-style) from the filename,
  2. finds the latest matching run folder in $RESULTS/<source_subfolder>/,
  3. invokes plot_histograms.py with --out pointing into results_histograms_pdf/.

Adam/RMSProp runs produce a _precond.pdf automatically.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PYTHON = "/n/home06/mwalden/.conda/envs/eoss/bin/python"
PLOT_SCRIPT = "/n/home06/mwalden/eoss/plot_histograms.py"
SRC_ROOT = Path("/n/home06/mwalden/eoss/marc_files/results_histograms")
DST_ROOT = Path("/n/home06/mwalden/eoss/marc_files/results_histograms_pdf")
RESULTS_ROOT = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results")

SOURCE_SUBFOLDERS = {
    "CNN_sweep": ["marc_cnn_sweep_fixed_u", "marc_cnn_sweep_fixed_u_n16384"],
    "vit_sweep": ["marc_vit_sweep"],
}


def parse_filename(fname: str) -> dict | None:
    """Return dict with keys optimizer, lr (or None), batch, beta_chunk (or '')."""
    m = re.match(
        r"histograms_"
        r"(?P<opt>SGD-Momentum|SGD-Nesterov|SGD|Adam|RMSProp|Muon)"
        r"_+(?:lr(?P<lr>[\d.eE+-]+)_+)?"
        r"b(?P<batch>\d+)"
        r"(?P<rest>(?:_+[a-z0-9-]+-[\d.eE+-]+)*)"
        r"\.png$",
        fname,
    )
    if not m:
        return None
    return {
        "opt": m.group("opt"),
        "lr": m.group("lr"),
        "batch": int(m.group("batch")),
        "rest": m.group("rest") or "",
    }


def find_run(parsed: dict, sub_dirs: list[str]) -> Path | None:
    """Pick the latest run folder matching this (optimizer, lr, batch, [betas])."""
    opt = parsed["opt"]
    batch = parsed["batch"]
    candidates: list[Path] = []
    for sub in sub_dirs:
        base = RESULTS_ROOT / sub
        if not base.exists():
            continue
        # Run folders: <ts>_<opt>_lr<lr>_b<batch>[_beta...]
        for run in base.iterdir():
            if not run.is_dir():
                continue
            name = run.name
            # Extract optimizer + batch from the run-folder name (canonical convention)
            rm = re.match(
                r"\d{8}_\d{4}_\d{2}_(?P<opt>SGD-Momentum|SGD-Nesterov|SGD|Adam|RMSProp|Muon)"
                r"_lr(?P<lr>[\d.eE+-]+)_b(?P<batch>\d+)",
                name,
            )
            if not rm:
                continue
            if rm.group("opt") != opt or int(rm.group("batch")) != batch:
                continue
            if parsed["lr"] is not None and rm.group("lr") != parsed["lr"]:
                continue
            # Need projections.npz to plot
            if not (run / "projections.npz").exists():
                continue
            candidates.append(run)

    if not candidates:
        return None
    # Latest by name (timestamp prefix sorts chronologically)
    return sorted(candidates, key=lambda p: p.name)[-1]


def main() -> None:
    failures: list[tuple[str, str]] = []
    successes: list[Path] = []

    for sweep_name, src_subs in SOURCE_SUBFOLDERS.items():
        sweep_src = SRC_ROOT / sweep_name
        sweep_dst = DST_ROOT / sweep_name
        if not sweep_src.exists():
            print(f"[skip] {sweep_src} does not exist")
            continue
        for opt_dir in sorted(sweep_src.iterdir()):
            if not opt_dir.is_dir():
                continue
            dst_opt = sweep_dst / opt_dir.name
            dst_opt.mkdir(parents=True, exist_ok=True)
            for png in sorted(opt_dir.glob("histograms_*.png")):
                if png.name.endswith("_precond.png"):
                    continue  # produced automatically by plot_histograms.py
                parsed = parse_filename(png.name)
                if parsed is None:
                    failures.append((str(png), "filename parse failed"))
                    continue
                run = find_run(parsed, src_subs)
                if run is None:
                    failures.append(
                        (str(png),
                         f"no run for opt={parsed['opt']} b={parsed['batch']} lr={parsed['lr']}")
                    )
                    continue
                pdf_path = dst_opt / png.with_suffix(".pdf").name
                cmd = [PYTHON, PLOT_SCRIPT, str(run), "--out", str(pdf_path)]
                print(f"[run] {png.relative_to(SRC_ROOT)} <- {run.name}")
                rc = subprocess.run(cmd, capture_output=True, text=True)
                if rc.returncode != 0:
                    failures.append((str(png), f"plot exit {rc.returncode}: {rc.stderr.strip()[:300]}"))
                    continue
                successes.append(pdf_path)

    print(f"\n=== Done. {len(successes)} successes, {len(failures)} failures.")
    for png, why in failures:
        print(f"  FAIL  {png}: {why}")


if __name__ == "__main__":
    main()
