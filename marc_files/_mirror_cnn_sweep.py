"""One-off helper: produce PNG + PDF histograms for every CNN_sweep cell.

CNN runs are split across two source subfolders (per the original sweep launch):
  $RESULTS/marc_cnn_sweep_fixed_u/         — base sweep
  $RESULTS/marc_cnn_sweep_fixed_u_n16384/  — extended-data variants for some cells

For each unique (optimizer, lr, batch, params) cell across BOTH source roots, this
picks the latest run folder (by timestamp prefix), then calls plot_histograms.py
twice — once for PNG (under results_histograms/CNN_sweep/<opt>/), once for PDF
(under results_histograms_pdf/CNN_sweep/<opt>/).

Adam/RMSProp produce auto _precond.{png,pdf} siblings.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

PYTHON = "/n/home06/mwalden/.conda/envs/eoss/bin/python"
PLOT_SCRIPT = "/n/home06/mwalden/eoss/plot_histograms.py"
RESULTS_ROOTS = [
    Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/marc_cnn_sweep_fixed_u"),
    Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results/marc_cnn_sweep_fixed_u_n16384"),
]
PNG_ROOT = Path("/n/home06/mwalden/eoss/marc_files/results_histograms/CNN_sweep")
PDF_ROOT = Path("/n/home06/mwalden/eoss/marc_files/results_histograms_pdf/CNN_sweep")

RUN_RE = re.compile(
    r"\d{8}_\d{4}_\d{2}_(?P<opt>SGD-Momentum|SGD-Nesterov|SGD|Adam|RMSProp|Muon)"
    r"_lr(?P<lr>[\d.eE+-]+)_b(?P<batch>\d+)(?P<rest>.*)$"
)


def main() -> None:
    cells: dict[tuple, list[Path]] = defaultdict(list)
    for root in RESULTS_ROOTS:
        if not root.exists():
            continue
        for run in root.iterdir():
            if not run.is_dir():
                continue
            m = RUN_RE.match(run.name)
            if not m:
                continue
            if not (run / "projections.npz").exists():
                continue
            key = (m.group("opt"), m.group("lr"), m.group("batch"), m.group("rest"))
            cells[key].append(run)

    failures: list[tuple[str, str]] = []
    successes: list[str] = []

    for (opt, lr, batch, rest), runs in sorted(cells.items()):
        run = sorted(runs, key=lambda p: p.name)[-1]
        stem = f"histograms_{opt}_lr{lr}_b{batch}{rest}"

        for ext, root in (("png", PNG_ROOT), ("pdf", PDF_ROOT)):
            dst_opt = root / opt
            dst_opt.mkdir(parents=True, exist_ok=True)
            out_path = dst_opt / f"{stem}.{ext}"
            cmd = [PYTHON, PLOT_SCRIPT, str(run), "--out", str(out_path)]
            print(f"[{ext}] {opt} b{batch}{rest}  <- {run.name}")
            rc = subprocess.run(cmd, capture_output=True, text=True)
            if rc.returncode != 0:
                failures.append((str(out_path),
                                 f"exit {rc.returncode}: {rc.stderr.strip()[:300]}"))
                continue
            successes.append(str(out_path))

    print(f"\n=== Done. {len(successes)} files written, {len(failures)} failures.")
    for path, why in failures:
        print(f"  FAIL  {path}: {why}")


if __name__ == "__main__":
    main()
