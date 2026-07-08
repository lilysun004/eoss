"""
Mechanism-B batch-size gap fill: run the subspace-GBS probe at intermediate
batch sizes {16,32,64} for all four optimizers, to see whether GBS_top ramps
smoothly from ~0.3-0.5 (b=8, momentum optimizers) up to ~2 (b=128), or jumps.

Reuses subspace_gbs_v2's train/probe/summarize. lr per (optimizer,new-batch) is
read from results/calib2/winners_gap.json (produced by first running
    python experiments/calibrate_grid.py experiments/calib_jobs_gap.json
which writes results/calib2/winners.json -- copy/point OUT_DIR there).

Improvement over the base run: probes at MULTIPLE checkpoints in the back half
(reduces the heavy single-checkpoint sampling noise) and averages.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("EOSS_SKIP_CHECKSUM", "1")
os.environ.setdefault("DATASETS", str(_REPO.parent.parent / "datasets"))

import experiments.subspace_gbs_v2 as sg   # noqa: E402
from utils.optimizer import create_optimizer  # noqa: E402

OUT_DIR = _REPO / "results" / "subspace_gbs_v2"
WINNERS = _REPO / "results" / "calib2" / "winners.json"

# gap cells: (cell_name in winners.json, optimizer, params, batch)
GAP = [
    ("SGD_b16", "SGD", {}, 16), ("SGD_b32", "SGD", {}, 32), ("SGD_b64", "SGD", {}, 64),
    ("SGDM09_b16", "SGD-Momentum", {"beta": 0.9}, 16), ("SGDM09_b32", "SGD-Momentum", {"beta": 0.9}, 32), ("SGDM09_b64", "SGD-Momentum", {"beta": 0.9}, 64),
    ("Adam_b16", "Adam", {"beta1": 0.9, "beta2": 0.99}, 16), ("Adam_b32", "Adam", {"beta1": 0.9, "beta2": 0.99}, 32), ("Adam_b64", "Adam", {"beta1": 0.9, "beta2": 0.99}, 64),
    ("Muon_b16", "Muon", {"momentum": 0.9}, 16), ("Muon_b32", "Muon", {"momentum": 0.9}, 32), ("Muon_b64", "Muon", {"momentum": 0.9}, 64),
]
CALIB_STEPS = {16: 3000, 32: 2600, 64: 2400}
N_CHECKPOINTS = 3
CHUNK_MARGIN = 1.3


def main():
    with open(WINNERS) as f:
        winners = json.load(f)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_txt = OUT_DIR / f"{ts}_subspace_gbs_gap.txt"
    out_json = OUT_DIR / f"{ts}_subspace_gbs_gap.json"
    results = []
    with open(out_txt, "w") as fh:
        sg.log(fh, f"mechB gap-fill ts={ts}  batches=16/32/64  N_CHECKPOINTS={N_CHECKPOINTS}")
        for cell_name, optn, params, batch in GAP:
            w = winners.get(cell_name)
            if not w or w.get("lr") is None:
                sg.log(fh, f"{cell_name}: NO CALIBRATED LR (run calibrate_grid on calib_jobs_gap.json first) -- skip")
                continue
            lr = w["lr"]
            total_steps = int(round(CALIB_STEPS[batch] * CHUNK_MARGIN))
            sg.log(fh, f"\n{'='*70}\nCELL {cell_name} {optn} b={batch} lr={lr} steps={total_steps}\n{'='*70}")
            torch.manual_seed(1000)
            net, X, Y, loss_fn = sg.build_net_and_data()
            opt = create_optimizer(optn, net, lr, params)
            # train to the start of the back half, then probe at N checkpoints
            first = total_steps // 2
            info = sg.train_fixed_lr(net, opt, X, Y, loss_fn, batch, first,
                                     monitor_every=max(1, first // 8), log_fh=fh, tag=cell_name)
            if info["diverged"]:
                sg.log(fh, f"  {cell_name}: DIVERGED in warmup -- skip")
                results.append(dict(cell=cell_name, optimizer=optn, batch=batch, lr=lr, diverged=True))
                continue
            gap = max(1, (total_steps - first) // N_CHECKPOINTS)
            all_recs = []
            for ck in range(N_CHECKPOINTS):
                recs = sg.run_probe(net, opt, X, Y, loss_fn, batch, sg.N_PROBE)
                all_recs.extend(recs)
                s = sg.summarize(recs)
                sg.log(fh, f"  checkpoint {ck}: GBS_top={s['GBS_top_outside_mean']:.3f} "
                           f"GBS_bulk={s['GBS_bulk_outside_mean']:.3f} GBS_total={s['GBS_total_outside_mean']:.3f}")
                if ck < N_CHECKPOINTS - 1:
                    sg.train_fixed_lr(net, opt, X, Y, loss_fn, batch, gap,
                                      monitor_every=gap, log_fh=fh, tag=f"{cell_name}_ck{ck}")
            summ = sg.summarize(all_recs)
            sg.log(fh, f"  {cell_name} AGG over {N_CHECKPOINTS} ckpts: "
                       f"GBS_top(out/in)={summ['GBS_top_outside_mean']:.3f}/{summ['GBS_top_inside']:.3f}  "
                       f"GBS_bulk={summ['GBS_bulk_outside_mean']:.3f}  GBS_total={summ['GBS_total_outside_mean']:.3f}  "
                       f"cross/Btot={summ['cross_over_Btotal']:.4f}")
            results.append(dict(cell=cell_name, optimizer=optn, batch=batch, lr=lr,
                                diverged=False, summary=summ))
            with open(out_json, "w") as jf:
                json.dump(results, jf, indent=2)

        sg.log(fh, "\n\nGAP SUMMARY TABLE")
        sg.log(fh, f"{'cell':14s} {'opt':14s} {'batch':>5s} {'GBS_top':>8s} {'GBS_bulk':>9s} {'GBS_total':>9s}")
        for r in results:
            if r.get("diverged"):
                sg.log(fh, f"{r['cell']:14s} {r['optimizer']:14s} {r['batch']:5d}  DIVERGED"); continue
            a = r["summary"]
            sg.log(fh, f"{r['cell']:14s} {r['optimizer']:14s} {r['batch']:5d} "
                       f"{a['GBS_top_outside_mean']:8.3f} {a['GBS_bulk_outside_mean']:9.3f} {a['GBS_total_outside_mean']:9.3f}")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
