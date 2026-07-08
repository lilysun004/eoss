"""
experiments/subspace_gbs_v2.py

Clean rerun of the Mechanism-B (mode mixture) subspace-GBS test, using the
pre-validated (lr, calib_steps) grid in results/calib2/FINAL_GRID.json instead
of any in-script lr tuning/auto-tune/restart logic. See instructions.tex /
suggested.txt Direction 3 / CLAUDE.md for full background.

Background: GBS = E_B[s_B^T H_B s_B / (-g_B^T s_B)] stabilizes at ~2 for ALL
optimizers at full batch, but below 2 for non-SGD optimizers at small batch.
Mechanism B says this is a weighting artifact: GBS is an energy-weighted
average over H_B's eigenmodes -- the top-K eigenspace should read ~2 for
every optimizer at every batch size, while the orthogonal "bulk" is <2 and
optimizer/batch-dependent; total GBS is the energy-weighted mixture.

Algorithm per probe batch B at a checkpoint theta (near EoS, i.e. after
calib_steps * ~1.3 steps of training at the grid's lr):
  1. Forward+backward on B -> loss, flat grads g_B (create_graph=True for HVP).
  2. s_B = optimizer_wrapper.compute_step_direction(g_B, params).
  3. Top-K eigenpairs (lambda_i, v_i), K=5, of H_B via compute_eigenvalues
     (LOBPCG, use_power_iteration=False).
  4. c_i = s_B . v_i ; s_top = sum_i c_i v_i ; s_bulk = s_B - s_top.
  5. Hs_top = hvp(s_top), Hs_total = hvp(s_B) (one extra HVP; reuse HVP object).
       B_total = s_B . Hs_total ; B_top = s_top . Hs_top ; B_bulk = B_total - B_top
       (algebraic split -- cross term s_top^T H_B s_bulk measured directly and
        reported relative to B_total)
       A_total = g_B . s_B ; A_top = g_B . s_top ; A_bulk = A_total - A_top (exact)
  6. GBS_x = B_x / (-A_x) for x in {top, bulk, total}, averaged over n_probe
     fresh probe batches (outside = mean of ratios, inside = mean(B)/-mean(A)).

Grid: 12 base_grid cells from FINAL_GRID.json (4 optimizers x batch {8,128,2048}),
CIFAR-10 MLP ('mlp_s'), num_data=2048, mse loss, dataset_seed=888, init_seed=8888,
init_scale=0.2, CPU only. lr and calib_steps come directly from the grid file
(with a +30% step margin) -- NOT re-tuned here.

Usage:
    source /Users/xq/Desktop/moonshot/eoss/.venv/bin/activate
    cd /Users/xq/Desktop/moonshot/eoss/.claude/worktrees/gbs-search
    export DATASETS=/Users/xq/Desktop/moonshot/eoss/datasets
    export EOSS_SKIP_CHECKSUM=1
    python experiments/subspace_gbs_v2.py                # full 12-cell grid
    python experiments/subspace_gbs_v2.py --only SGD_b8   # single cell (debug)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("EOSS_SKIP_CHECKSUM", "1")
os.environ.setdefault("DATASETS", str(REPO_ROOT.parent.parent / "datasets"))

if os.environ.get("EOSS_SKIP_CHECKSUM"):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

from utils.data import prepare_dataset, get_dataset_presets          # noqa: E402
from utils.nets import get_model_presets, prepare_net, initialize_net, SquaredLoss  # noqa: E402
from utils.optimizer import create_optimizer                          # noqa: E402
from utils.measure import (                                           # noqa: E402
    compute_eigenvalues,
    create_hessian_vector_product,
    flatt,
)

torch.set_num_threads(max(1, os.cpu_count() or 1))

DATASET_FOLDER = Path(os.environ["DATASETS"]).expanduser()
GRID_PATH = REPO_ROOT / "results" / "calib2" / "FINAL_GRID.json"
OUT_DIR = REPO_ROOT / "results" / "subspace_gbs_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "cifar10"
NUM_DATA = 2048
MODEL_PRESET = "mlp_s"
LOSS_TYPE = "mse"
DATASET_SEED = 888
INIT_SCALE = 0.2
INIT_SEED = 8888
K_TOP = 5
N_PROBE = 20        # within instructed 16-24 range
STEP_MARGIN = 1.3   # "modest safety margin ... e.g. +30%"


def log(fh, msg):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def safe_compute_eigenvalues(loss, net, k, max_iterations=40, reltol=1e-2,
                              init_vectors=None, grads=None, max_retries=4):
    """LOBPCG occasionally fails to converge (ill-conditioned / repeated
    eigenvalues on tiny/degenerate batch Hessians); retry with a fresh random
    init subspace (drop warm start) when that happens."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return compute_eigenvalues(
                loss, net, k=k, max_iterations=max_iterations, reltol=reltol,
                init_vectors=init_vectors if attempt == 0 else None,
                return_eigenvectors=True, use_power_iteration=False, grads=grads,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            init_vectors = None
    raise last_err


def build_net_and_data(seed_offset=0):
    X, Y, _, _ = prepare_dataset(DATASET, DATASET_FOLDER, NUM_DATA, [], DATASET_SEED, loss_type=LOSS_TYPE)
    presets = get_model_presets()
    dpresets = get_dataset_presets()
    params = dict(presets[MODEL_PRESET]["params"])
    params["input_dim"] = dpresets[DATASET]["input_dim"]
    params["output_dim"] = dpresets[DATASET]["output_dim"]
    net = prepare_net(model_type=presets[MODEL_PRESET]["type"], params=params)
    initialize_net(net, scale=INIT_SCALE, seed=INIT_SEED)
    loss_fn = SquaredLoss()
    return net, X, Y, loss_fn


def train_fixed_lr(net, opt, X, Y, loss_fn, batch_size, steps, monitor_every, log_fh, tag):
    """No auto-tuning, no restarts -- fixed lr from the validated grid.
    Aborts early (reporting diverged=True) if loss ever goes non-finite."""
    n = len(X)
    full_batch = batch_size >= n
    pbar = tqdm(range(steps), desc=f"train {tag}", leave=False)
    diverged = False
    last_loss = float("nan")
    last_lmax = float("nan")
    eig_cache_vec = None
    for step in pbar:
        if full_batch:
            Xb, Yb = X, Y
        else:
            idx = torch.randperm(n)[:batch_size]
            Xb, Yb = X[idx], Y[idx]

        opt.zero_grad()
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        loss.backward()
        opt.step()

        loss_val = loss.item()
        last_loss = loss_val
        if not np.isfinite(loss_val) or abs(loss_val) > 1e8:
            diverged = True
            log(log_fh, f"  [DIVERGED @ step {step}] loss={loss_val}")
            break

        if step % monitor_every == 0 or step == steps - 1:
            try:
                preds2 = net(Xb).squeeze(-1)
                loss2 = loss_fn(preds2, Yb)
                lam = compute_eigenvalues(
                    loss2, net, k=1, max_iterations=25, reltol=0.03,
                    return_eigenvectors=False, use_power_iteration=False,
                ).item()
                last_lmax = lam
            except Exception:
                lam = float("nan")
            pbar.set_postfix(loss=f"{loss_val:.4g}", lam=f"{lam:.3g}")
            log(log_fh, f"  step {step:5d}  loss={loss_val:.6g}  lambda~{lam:.4g}")

    return dict(diverged=diverged, final_loss=last_loss, final_lmax=last_lmax, steps_done=step + 1)


def run_probe(net, opt, X, Y, loss_fn, batch_size, n_probe, k=K_TOP,
              max_iterations=40, reltol=1e-2):
    n = len(X)
    params = [p for p in net.parameters() if p.requires_grad]
    full_batch = batch_size >= n
    eigvec_init = None
    records = []
    n_actual = 1 if full_batch else n_probe  # full batch has no batch-randomness to average over

    for probe_i in range(n_actual):
        if full_batch:
            Xb, Yb = X, Y
        else:
            idx = torch.randperm(n)[:batch_size]
            Xb, Yb = X[idx], Y[idx]

        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = flatt(grads)

        s_b = opt.compute_step_direction(g_flat, params).detach()

        try:
            eigvals, eigvecs = safe_compute_eigenvalues(
                loss, net, k=k, max_iterations=max_iterations, reltol=reltol,
                init_vectors=eigvec_init, grads=grads,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    [probe {probe_i}] eigensolve failed after retries, skipping: {e}")
            eigvec_init = None
            continue
        eigvecs = eigvecs.detach()
        eigvec_init = eigvecs.clone()

        c = torch.matmul(eigvecs.t(), s_b)          # [k]
        s_top = eigvecs @ c
        s_bulk = s_b - s_top

        hvp = create_hessian_vector_product(loss, net, params=params, grads=grads, flat_grads=g_flat)
        try:
            Hs_top = hvp(s_top, retain_graph_override=True)
            Hs_total = hvp(s_b, retain_graph_override=False)
        finally:
            hvp.free_memory()

        B_top = torch.dot(s_top, Hs_top).item()
        B_total = torch.dot(s_b, Hs_total).item()
        B_bulk = B_total - B_top
        # cross term s_top^T H_B s_bulk = s_top . (Hs_total - Hs_top)
        cross = torch.dot(s_top, Hs_total - Hs_top).item()

        g_det = g_flat.detach()
        A_total = torch.dot(g_det, s_b).item()
        A_top = torch.dot(g_det, s_top).item()
        A_bulk = A_total - A_top

        def ratio(bv, av):
            return bv / (-av) if abs(av) > 1e-12 else float("nan")

        records.append(dict(
            A_top=A_top, B_top=B_top, GBS_top=ratio(B_top, A_top),
            A_bulk=A_bulk, B_bulk=B_bulk, GBS_bulk=ratio(B_bulk, A_bulk),
            A_total=A_total, B_total=B_total, GBS_total=ratio(B_total, A_total),
            cross=cross,
            lambda_top1=float(eigvals[0]) if hasattr(eigvals, "__len__") else float(eigvals),
        ))

    return records


def summarize(records):
    keys3 = ("top", "bulk", "total")
    if not records:
        out = {f"GBS_{x}_outside_mean": float("nan") for x in keys3}
        out.update({f"GBS_{x}_outside_std": float("nan") for x in keys3})
        out.update({f"GBS_{x}_inside": float("nan") for x in keys3})
        out.update(cross_abs_mean=float("nan"), B_total_abs_mean=float("nan"),
                    cross_over_Btotal=float("nan"), lambda_top1_mean=float("nan"), n_probe=0)
        return out
    arr = {key: np.array([r[key] for r in records], dtype=float) for key in records[0].keys()}
    out = {}
    for x in keys3:
        gbs = arr[f"GBS_{x}"]
        gbs_f = gbs[np.isfinite(gbs)]
        out[f"GBS_{x}_outside_mean"] = float(np.mean(gbs_f)) if len(gbs_f) else float("nan")
        out[f"GBS_{x}_outside_std"] = float(np.std(gbs_f)) if len(gbs_f) else float("nan")
        meanA = float(np.mean(arr[f"A_{x}"]))
        meanB = float(np.mean(arr[f"B_{x}"]))
        out[f"GBS_{x}_inside"] = (meanB / -meanA) if abs(meanA) > 1e-12 else float("nan")
    out["cross_abs_mean"] = float(np.mean(np.abs(arr["cross"])))
    out["B_total_abs_mean"] = float(np.mean(np.abs(arr["B_total"])))
    out["cross_over_Btotal"] = (out["cross_abs_mean"] / out["B_total_abs_mean"]
                                 if out["B_total_abs_mean"] > 1e-12 else float("nan"))
    out["lambda_top1_mean"] = float(np.mean(arr["lambda_top1"]))
    out["n_probe"] = len(records)
    return out


def run_cell(cell_name, cell, log_fh):
    optimizer_name = cell["optimizer"]
    optimizer_params = cell.get("params", {})
    batch_size = cell["batch"]
    lr = cell["lr"]
    steps = int(round(cell["calib_steps"] * STEP_MARGIN))

    log(log_fh, f"\n{'='*80}\nCELL {cell_name}  optimizer={optimizer_name} params={optimizer_params} "
                 f"batch={batch_size} lr={lr} steps={steps} (calib_steps={cell['calib_steps']} x{STEP_MARGIN})\n{'='*80}")

    torch.manual_seed(1000)
    net, X, Y, loss_fn = build_net_and_data()
    opt = create_optimizer(optimizer_name, net, lr, optimizer_params)

    t0 = time.time()
    train_info = train_fixed_lr(net, opt, X, Y, loss_fn, batch_size, steps,
                                 monitor_every=max(1, steps // 15), log_fh=log_fh, tag=cell_name)
    log(log_fh, f"  training done in {time.time()-t0:.1f}s  diverged={train_info['diverged']} "
                 f"final_loss={train_info['final_loss']:.6g} final_lmax={train_info['final_lmax']:.4g} "
                 f"steps_done={train_info['steps_done']}/{steps}")

    t1 = time.time()
    records = run_probe(net, opt, X, Y, loss_fn, batch_size, N_PROBE, k=K_TOP)
    summ = summarize(records)
    log(log_fh, f"  probe done in {time.time()-t1:.1f}s  n_probe={summ['n_probe']}  "
                 f"GBS_top(out/in)={summ['GBS_top_outside_mean']:.3f}/{summ['GBS_top_inside']:.3f}  "
                 f"GBS_bulk(out/in)={summ['GBS_bulk_outside_mean']:.3f}/{summ['GBS_bulk_inside']:.3f}  "
                 f"GBS_total(out/in)={summ['GBS_total_outside_mean']:.3f}/{summ['GBS_total_inside']:.3f}  "
                 f"cross/|B_total|={summ['cross_over_Btotal']:.4f}  lam1~{summ['lambda_top1_mean']:.3f}")
    for r in records[:5]:
        log(log_fh, f"      raw probe: {json.dumps(r)}")

    return dict(
        cell=cell_name, optimizer=optimizer_name, optimizer_params=optimizer_params,
        batch_size=batch_size, lr=lr, steps=steps, calib_steps=cell["calib_steps"],
        calib_gbs=cell.get("calib_gbs"), train_info=train_info,
        summary=summ, records=records,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="comma list of cell names to run (debug)")
    args = ap.parse_args()

    with open(GRID_PATH) as f:
        grid_data = json.load(f)
    base_grid = grid_data["base_grid"]

    cell_names = list(base_grid.keys())
    if args.only:
        wanted = set(args.only.split(","))
        cell_names = [c for c in cell_names if c in wanted]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_txt = OUT_DIR / f"{ts}_subspace_gbs_v2.txt"
    out_json = OUT_DIR / f"{ts}_subspace_gbs_v2.json"
    all_results = []

    with open(out_txt, "w") as log_fh:
        log(log_fh, f"subspace_gbs_v2 run ts={ts} cells={cell_names} K_TOP={K_TOP} "
                     f"N_PROBE={N_PROBE} STEP_MARGIN={STEP_MARGIN} num_data={NUM_DATA} model={MODEL_PRESET}")
        t_start = time.time()
        for cell_name in cell_names:
            cell = base_grid[cell_name]
            result = run_cell(cell_name, cell, log_fh)
            all_results.append(result)
            with open(out_json, "w") as jf:
                json.dump(all_results, jf, indent=2)

        log(log_fh, f"\nTOTAL WALL CLOCK: {time.time()-t_start:.1f}s")
        log(log_fh, "\n\nSUMMARY TABLE")
        log(log_fh, f"{'cell':16s} {'opt':14s} {'batch':>6s} {'calib_gbs':>9s} "
                     f"{'GBS_top(out)':>12s} {'GBS_top(in)':>11s} "
                     f"{'GBS_bulk(out)':>13s} {'GBS_bulk(in)':>12s} "
                     f"{'GBS_total(out)':>14s} {'GBS_total(in)':>13s} {'cross/Btot':>10s}")
        for cell_result in all_results:
            a = cell_result["summary"]
            log(log_fh, f"{cell_result['cell']:16s} {cell_result['optimizer']:14s} {cell_result['batch_size']:6d} "
                         f"{cell_result['calib_gbs']:9.3f} "
                         f"{a['GBS_top_outside_mean']:12.4f} {a['GBS_top_inside']:11.4f} "
                         f"{a['GBS_bulk_outside_mean']:13.4f} {a['GBS_bulk_inside']:12.4f} "
                         f"{a['GBS_total_outside_mean']:14.4f} {a['GBS_total_inside']:13.4f} "
                         f"{a['cross_over_Btotal']:10.4f}")

    print(f"\nWrote log to {out_txt}\nWrote json to {out_json}")


if __name__ == "__main__":
    main()
