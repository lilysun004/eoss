"""
experiments/subspace_gbs.py

Tests "Mechanism B" (mode mixture) from suggested.txt Direction 3 — ranked #1
("cheapest, most decisive") in the ranked experiment list.

Background (see instructions.tex / suggested.txt / CLAUDE.md for full context):
  GBS = E_B[ s_B^T H_B s_B / (-g_B^T s_B) ]  stabilizes at 2 for ALL optimizers at
  large/full batch, but at small batch it stabilizes BELOW 2 for every optimizer
  except SGD. Mechanism B says this is a weighting artifact: GBS is an
  energy-weighted average of per-mode ratios; modes at the edge (top eigenspace
  of H_B) contribute ~2, stable bulk modes are genuinely descending and
  contribute <2. If true, GBS restricted to just the top-K eigenspace of H_B
  should be ~2 for ALL optimizers at ALL batch sizes, while the bulk-restricted
  GBS is <2 and optimizer/batch dependent.

Algorithm per probe batch B at iterate theta (near EoS):
  1. loss, g_B = forward/backward on B (create_graph=True for HVPs).
  2. s_B = optimizer_wrapper.compute_step_direction(g_B, params)  (detached)
  3. (lambda_i, v_i), i=1..K = top-K eigenpairs of H_B (LOBPCG, K=5).
  4. c_i = s_B . v_i ; s_top = sum_i c_i v_i ; s_bulk = s_B - s_top.
  5. Hs_top = H_B s_top ; Hs_total = H_B s_B  (two HVP calls; H_B s_bulk inferred
     algebraically as Hs_total - Hs_top, so the top/bulk curvature split
     assumes the cross term s_top^T H_B s_bulk is small -- this holds well once
     LOBPCG has converged, since s_top lies (almost) exactly in the top-K
     eigenspace which is (almost) exactly H_B-invariant. We measure this cross
     term directly on every probe and report its typical size relative to
     B_total.)
       B_total = s_B . Hs_total ; B_top = s_top . Hs_top ; B_bulk = B_total - B_top
       A_total = g_B . s_B ; A_top = g_B . s_top ; A_bulk = A_total - A_top  (EXACT
       split, bilinearity of the dot product, no approximation)
  6. GBS_x = B_x / (-A_x) for x in {top, bulk, total}.
  7. Average over n_probe fresh probe batches. Report BOTH the "outside"
     mean-of-ratios (mean of per-probe GBS_x) AND the "inside" placement
     (mean(B_x) / -mean(A_x)) -- these differ by Jensen gaps.

Grid: {SGD, SGD-Momentum(beta=0.9), Adam(beta1=0.9,beta2=0.99), Muon(momentum=0.9)}
      x batch size in {8, 128, full(=num_data)}, CIFAR-10 MLP (preset 'mlp_s'),
      CPU only, num_data=2048.

Usage:
    source /Users/xq/Desktop/moonshot/eoss/.venv/bin/activate
    cd /Users/xq/Desktop/moonshot/eoss/.claude/worktrees/gbs-search
    export DATASETS=/Users/xq/Desktop/moonshot/eoss/datasets
    export EOSS_SKIP_CHECKSUM=1
    export RESULTS=/Users/xq/Desktop/moonshot/eoss/.claude/worktrees/gbs-search/results
    python experiments/subspace_gbs.py                 # full grid
    python experiments/subspace_gbs.py --quick          # fast 1-cell smoke test
    python experiments/subspace_gbs.py --only SGD,Adam --batches 8,128
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

if os.environ.get("EOSS_SKIP_CHECKSUM"):
    # Local CPU smoke-testing: our CIFAR-10 copy comes from a mirror other than
    # the official host, so torchvision's hardcoded MD5s legitimately differ.
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
RESULTS_FOLDER = Path(os.environ.get("RESULTS", str(REPO_ROOT / "results"))).expanduser()
OUT_DIR = RESULTS_FOLDER / "subspace_gbs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "cifar10"
NUM_DATA = 2048
MODEL_PRESET = "mlp_s"          # hidden_dim=256, n_layers=1 -- small & fast on CPU
LOSS_TYPE = "mse"
DATASET_SEED = 888
INIT_SCALE = 0.2
INIT_SEED = 8888
K_TOP = 5

# (name, optimizer_params, reference_lr_at_num_data_8192)
OPTIMIZER_GRID = [
    ("SGD", {}, 0.02),
    ("SGD-Momentum", {"beta": 0.9}, 0.002),
    ("Adam", {"beta1": 0.9, "beta2": 0.99}, 0.00003),
    ("Muon", {"momentum": 0.9}, 0.001),
]

BATCH_LABELS = {"8": 8, "128": 128, "full": NUM_DATA}


def log(fh, msg):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def safe_compute_eigenvalues(loss, net, k, max_iterations, reltol, init_vectors=None,
                               grads=None, return_eigenvectors=True, max_retries=4):
    """compute_eigenvalues wrapper that retries on LOBPCG's occasional
    eigh-fails-to-converge error (ill-conditioned / repeated eigenvalues on
    some tiny/degenerate batch Hessians). Retries drop the warm start and
    re-roll a fresh random init subspace, which reliably clears the issue."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return compute_eigenvalues(
                loss, net, k=k, max_iterations=max_iterations, reltol=reltol,
                init_vectors=init_vectors if attempt == 0 else None,
                return_eigenvectors=return_eigenvectors, use_power_iteration=False,
                grads=grads,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            init_vectors = None
    raise last_err


def measure_lambda(net, loss_fn, X, Y, subset_size, max_iterations=40, reltol=1e-2):
    """Cheap top-1 eigenvalue probe (fresh random subset) for EoS monitoring."""
    n = len(X)
    subset_size = min(subset_size, n)
    idx = torch.randperm(n)[:subset_size]
    Xb, Yb = X[idx], Y[idx]
    loss = loss_fn(net(Xb).squeeze(-1), Yb)
    try:
        lam = safe_compute_eigenvalues(
            loss, net, k=1, max_iterations=max_iterations, reltol=reltol,
            return_eigenvectors=False,
        )
    except Exception:
        return float("nan")
    return float(lam.detach())


def build_net_and_data():
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


def train_to_eos(optimizer_name, optimizer_params, lr, batch_size, steps, monitor_every, log_fh,
                   max_restarts=4, pretune_target_ratio=0.8):
    n = NUM_DATA
    full_batch = batch_size >= n
    # Monitor sharpness at the *actual* training batch size (no artificial floor):
    # batch sharpness self-stabilization is batch-size-specific, so calibrating
    # against a mismatched (larger) subset systematically under-estimates the
    # lr needed to reach EoS at small batch.
    monitor_subset = min(batch_size, n)

    for restart in range(max_restarts + 1):
        torch.manual_seed(1000 + restart)
        net, X, Y, loss_fn = build_net_and_data()
        cur_lr = lr * (0.5 ** restart)  # halve lr on every restart caused by divergence

        # ---- Pre-tune: rescale lr against the *initial* curvature so we start
        # a bit below the naive edge threshold (progressive sharpening during
        # training will carry the run up into EoS; starting already above 1x
        # is what caused immediate NaN blowups at num_data=2048, where the
        # untrained-network curvature is higher than at num_data=8192). ----
        lam0 = measure_lambda(net, loss_fn, X, Y, subset_size=monitor_subset)
        if np.isfinite(lam0) and lam0 > 1e-8:
            ratio0 = cur_lr * lam0 / 2.0
            if ratio0 > 1e-8:
                factor0 = pretune_target_ratio / ratio0
                pre_lr = cur_lr * factor0
                log(log_fh, f"  [pre-tune] lambda0~{lam0:.4f}  lr {cur_lr:.6g} -> {pre_lr:.6g} "
                            f"(ratio0={ratio0:.3f} -> target {pretune_target_ratio})")
                cur_lr = pre_lr

        opt = create_optimizer(optimizer_name, net, cur_lr, optimizer_params)

        n_tunes = 0
        max_tunes = 3
        tune_window = (int(0.15 * steps), int(0.85 * steps))

        pbar = tqdm(range(steps), desc=f"train {optimizer_name} b={batch_size} (try{restart})", leave=False)
        last_ratio = float("nan")
        diverged = False
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
            if not np.isfinite(loss_val):
                log(log_fh, f"  [DIVERGED @ step {step}] loss is non-finite ({loss_val}); "
                            f"will restart with halved base lr" if restart < max_restarts
                            else f"  [DIVERGED @ step {step}] out of restarts, keeping partial run")
                diverged = True
                break

            if step % monitor_every == 0 or step == steps - 1:
                lam = measure_lambda(net, loss_fn, X, Y, subset_size=monitor_subset)
                cur_opt_lr = opt.param_groups[0]["lr"]
                ratio = cur_opt_lr * lam / 2.0
                last_ratio = ratio
                pbar.set_postfix(loss=f"{loss_val:.4f}", lam=f"{lam:.2f}", lr_lam_2=f"{ratio:.3f}")
                log(log_fh, f"  step {step:5d}  loss={loss_val:.6f}  lambda~{lam:.4f}  "
                            f"lr={cur_opt_lr:.6g}  lr*lambda/2={ratio:.4f}")

                if (n_tunes < max_tunes) and tune_window[0] <= step <= tune_window[1] and np.isfinite(ratio) and ratio > 1e-8:
                    factor = max(0.4, min(2.5, 1.0 / max(ratio, 1e-3)))
                    if ratio > 1.4 or ratio < 0.65:
                        new_lr = cur_opt_lr * factor
                        for g in opt.param_groups:
                            g["lr"] = new_lr
                        log(log_fh, f"  [auto-tune @ step {step}] lr {cur_opt_lr:.6g} -> {new_lr:.6g} "
                                    f"(factor {factor:.3f}, target lr*lambda/2~1)")
                        n_tunes += 1

        if not diverged or restart == max_restarts:
            return net, opt, X, Y, loss_fn, last_ratio

    # unreachable
    return net, opt, X, Y, loss_fn, last_ratio


def run_subspace_gbs_probe(net, opt, X, Y, loss_fn, batch_size, n_probe, k=K_TOP,
                             max_iterations=40, reltol=1e-2):
    n = len(X)
    params = [p for p in net.parameters() if p.requires_grad]
    full_batch = batch_size >= n
    eigvec_init = None  # warm-start [P, k] tensor across probes

    records = []
    for probe_i in range(n_probe):
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
                init_vectors=eigvec_init, return_eigenvectors=True, grads=grads,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    [probe {probe_i}] eigensolve failed after retries, skipping probe: {e}")
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
        # cross term s_top^T H_B s_bulk = s_top . (Hs_total - Hs_top) = s_top.Hs_bulk
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

        if full_batch:
            # deterministic given theta -- more probes would just repeat this record.
            break

    return records


def summarize(records):
    if not records:
        keys = ["A_top", "B_top", "GBS_top", "A_bulk", "B_bulk", "GBS_bulk",
                "A_total", "B_total", "GBS_total", "cross", "lambda_top1"]
        out = {f"GBS_{x}_outside_mean": float("nan") for x in ("top", "bulk", "total")}
        out.update({f"GBS_{x}_outside_median": float("nan") for x in ("top", "bulk", "total")})
        out.update({f"GBS_{x}_outside_std": float("nan") for x in ("top", "bulk", "total")})
        out.update({f"GBS_{x}_inside": float("nan") for x in ("top", "bulk", "total")})
        out.update(cross_mean=float("nan"), cross_abs_mean=float("nan"),
                    B_total_abs_mean=float("nan"), cross_over_Btotal=float("nan"),
                    lambda_top1_mean=float("nan"), n_probe=0)
        return out
    arr = {key: np.array([r[key] for r in records], dtype=float) for key in records[0].keys()}
    out = {}
    for x in ("top", "bulk", "total"):
        gbs = arr[f"GBS_{x}"]
        gbs_finite = gbs[np.isfinite(gbs)]
        out[f"GBS_{x}_outside_mean"] = float(np.mean(gbs_finite)) if len(gbs_finite) else float("nan")
        out[f"GBS_{x}_outside_median"] = float(np.median(gbs_finite)) if len(gbs_finite) else float("nan")
        out[f"GBS_{x}_outside_std"] = float(np.std(gbs_finite)) if len(gbs_finite) else float("nan")
        meanA = float(np.mean(arr[f"A_{x}"]))
        meanB = float(np.mean(arr[f"B_{x}"]))
        out[f"GBS_{x}_inside"] = (meanB / -meanA) if abs(meanA) > 1e-12 else float("nan")
    out["cross_mean"] = float(np.mean(arr["cross"]))
    out["cross_abs_mean"] = float(np.mean(np.abs(arr["cross"])))
    out["B_total_abs_mean"] = float(np.mean(np.abs(arr["B_total"])))
    out["cross_over_Btotal"] = (out["cross_abs_mean"] / out["B_total_abs_mean"]
                                  if out["B_total_abs_mean"] > 1e-12 else float("nan"))
    out["lambda_top1_mean"] = float(np.mean(arr["lambda_top1"]))
    out["n_probe"] = len(records)
    return out


def run_cell(optimizer_name, optimizer_params, lr, batch_size, steps, n_checkpoints,
             n_probe, monitor_every, log_fh):
    log(log_fh, f"\n{'='*80}\nCELL optimizer={optimizer_name} params={optimizer_params} "
                 f"lr={lr} batch_size={batch_size}\n{'='*80}")
    t0 = time.time()
    net, opt, X, Y, loss_fn, last_ratio = train_to_eos(
        optimizer_name, optimizer_params, lr, batch_size, steps, monitor_every, log_fh,
    )
    log(log_fh, f"  training done in {time.time()-t0:.1f}s, final lr*lambda/2~{last_ratio:.4f}")

    checkpoint_summaries = []
    ckpt_pbar = tqdm(range(n_checkpoints), desc=f"probe {optimizer_name} b={batch_size}", leave=False)
    for ci in ckpt_pbar:
        t1 = time.time()
        # take one extra optimizer step between checkpoints so successive
        # checkpoints sample slightly different points along the EoS oscillation.
        if ci > 0:
            n = len(X)
            if batch_size >= n:
                Xb, Yb = X, Y
            else:
                idx = torch.randperm(n)[:batch_size]
                Xb, Yb = X[idx], Y[idx]
            opt.zero_grad()
            preds = net(Xb).squeeze(-1)
            loss = loss_fn(preds, Yb)
            loss.backward()
            opt.step()

        records = run_subspace_gbs_probe(net, opt, X, Y, loss_fn, batch_size, n_probe)
        summ = summarize(records)
        checkpoint_summaries.append(summ)
        dt = time.time() - t1
        log(log_fh, f"  checkpoint {ci}: n_probe={summ['n_probe']} dt={dt:.1f}s  "
                     f"GBS_top(out/in)={summ['GBS_top_outside_mean']:.3f}/{summ['GBS_top_inside']:.3f}  "
                     f"GBS_bulk(out/in)={summ['GBS_bulk_outside_mean']:.3f}/{summ['GBS_bulk_inside']:.3f}  "
                     f"GBS_total(out/in)={summ['GBS_total_outside_mean']:.3f}/{summ['GBS_total_inside']:.3f}  "
                     f"cross/|B_total|={summ['cross_over_Btotal']:.4f}  lam1~{summ['lambda_top1_mean']:.3f}")
        for r in records[:3]:
            log(log_fh, f"      raw probe: {json.dumps(r)}")

    agg = {}
    for key in checkpoint_summaries[0].keys():
        vals = [c[key] for c in checkpoint_summaries if np.isfinite(c[key])]
        agg[key] = float(np.mean(vals)) if vals else float("nan")
    log(log_fh, f"  CELL AGGREGATE (mean over {len(checkpoint_summaries)} checkpoints): "
                 f"GBS_top(out/in)={agg['GBS_top_outside_mean']:.3f}/{agg['GBS_top_inside']:.3f}  "
                 f"GBS_bulk(out/in)={agg['GBS_bulk_outside_mean']:.3f}/{agg['GBS_bulk_inside']:.3f}  "
                 f"GBS_total(out/in)={agg['GBS_total_outside_mean']:.3f}/{agg['GBS_total_inside']:.3f}  "
                 f"cross/|B_total|={agg['cross_over_Btotal']:.4f}")

    return dict(
        optimizer=optimizer_name, optimizer_params=optimizer_params, lr=lr,
        batch_size=batch_size, final_lr_lambda_2=last_ratio,
        checkpoints=checkpoint_summaries, aggregate=agg,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny 1-cell smoke test")
    ap.add_argument("--only", type=str, default=None, help="comma list of optimizer names to run")
    ap.add_argument("--batches", type=str, default=None, help="comma list of batch labels: 8,128,full")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--checkpoints", type=int, default=6)
    ap.add_argument("--n_probe", type=int, default=16)
    ap.add_argument("--monitor_every", type=int, default=200)
    args = ap.parse_args()

    if args.quick:
        args.steps = 150
        args.checkpoints = 2
        args.n_probe = 3
        args.monitor_every = 30

    grid = OPTIMIZER_GRID
    if args.only:
        wanted = set(args.only.split(","))
        grid = [g for g in grid if g[0] in wanted]

    batch_labels = list(BATCH_LABELS.keys())
    if args.batches:
        batch_labels = args.batches.split(",")
    if args.quick:
        grid = grid[:1]
        batch_labels = batch_labels[:1]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{ts}_subspace_gbs{'_quick' if args.quick else ''}.txt"
    all_results = []
    with open(out_path, "w") as log_fh:
        log(log_fh, f"subspace_gbs experiment  ts={ts}  steps={args.steps} "
                     f"checkpoints={args.checkpoints} n_probe={args.n_probe} "
                     f"num_data={NUM_DATA} model={MODEL_PRESET} K_TOP={K_TOP}")
        t_start = time.time()
        for optimizer_name, optimizer_params, ref_lr in grid:
            for blabel in batch_labels:
                batch_size = BATCH_LABELS[blabel]
                cell = run_cell(
                    optimizer_name, optimizer_params, ref_lr, batch_size,
                    args.steps, args.checkpoints, args.n_probe, args.monitor_every, log_fh,
                )
                all_results.append(cell)
                json_path = OUT_DIR / f"{ts}_subspace_gbs{'_quick' if args.quick else ''}.json"
                with open(json_path, "w") as jf:
                    json.dump(all_results, jf, indent=2)

        log(log_fh, f"\nTOTAL WALL CLOCK: {time.time()-t_start:.1f}s")
        log(log_fh, "\n\nSUMMARY TABLE (aggregate over checkpoints)")
        log(log_fh, f"{'optimizer':14s} {'batch':>6s} {'GBS_top(out)':>12s} {'GBS_top(in)':>11s} "
                     f"{'GBS_bulk(out)':>13s} {'GBS_bulk(in)':>12s} {'GBS_total(out)':>14s} "
                     f"{'GBS_total(in)':>13s} {'cross/Btot':>10s}")
        for cell in all_results:
            a = cell["aggregate"]
            log(log_fh, f"{cell['optimizer']:14s} {cell['batch_size']:6d} "
                         f"{a['GBS_top_outside_mean']:12.4f} {a['GBS_top_inside']:11.4f} "
                         f"{a['GBS_bulk_outside_mean']:13.4f} {a['GBS_bulk_inside']:12.4f} "
                         f"{a['GBS_total_outside_mean']:14.4f} {a['GBS_total_inside']:13.4f} "
                         f"{a['cross_over_Btotal']:10.4f}")

    print(f"\nWrote log to {out_path}")


if __name__ == "__main__":
    main()
