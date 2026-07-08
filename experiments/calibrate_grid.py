"""
Fast, no-frills calibration harness: find (optimizer, batch_size, optimizer_params, lr)
cells where training actually reaches a stabilized regime, checking ONLY:
  - loss doesn't diverge (no NaN/Inf, stays bounded) -- it need NOT stop decreasing
  - GBS (generalized batch sharpness) stabilizes (no strong monotonic trend)
  - BS (batch_sharpness, E[g^T H_B g / ||g||^2]) stabilizes

No power iteration for extra eigvecs, no subspace splits, no distributions -- just the
three scalars per measurement. This is the ground-truth grid to hand to the real
experiment agents afterward, so they don't waste time recalibrating.

Usage:
    python experiments/calibrate_grid.py '<json list of jobs>'
Each job: {"optimizer": "SGD", "params": {}, "batch": 8, "lr": 0.02, "steps": 1500,
           "tag": "sgd_b8_lr0.02"}
Writes results/calib2/<tag>.json and prints a one-line PASS/FAIL summary per job.
"""
import os, sys, time, json
import numpy as np
import torch as T

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')

if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.measure import (
    compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
    flatt, calculate_averaged_grad_H_grad_step,
)

T.set_num_threads(4)

DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = 2048
MODEL = 'mlp_s'
OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'calib2')
os.makedirs(OUT_DIR, exist_ok=True)

_DATA_CACHE = {}


def get_data():
    if 'xy' not in _DATA_CACHE:
        data = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], 888, loss_type='mse')
        X_train, Y_train, _, _ = data
        _DATA_CACHE['xy'] = (X_train, Y_train)
    return _DATA_CACHE['xy']


def quick_gbs(net, X, Y, loss_fn, opt_wrapper, batch_size, n_probe=4):
    vals = []
    N = len(X)
    for _ in range(n_probe):
        idx = T.randperm(N)[:batch_size]
        Xb, Yb = X[idx], Y[idx]
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        params = [p for p in net.parameters() if p.requires_grad]
        grads = T.autograd.grad(loss, params, create_graph=True)
        g_flat = flatt(grads)
        s = opt_wrapper.compute_step_direction(g_flat, params)
        hvp = create_hessian_vector_product(loss, net, params=params, grads=grads, flat_grads=g_flat)
        try:
            Hs = hvp(s, retain_graph_override=False)
            A = T.dot(g_flat.detach(), s).item()
            B = T.dot(s, Hs).item()
        finally:
            hvp.free_memory()
        if abs(A) > 1e-12 and np.isfinite(A) and np.isfinite(B):
            vals.append(B / (-A))
    return vals


def run_cell(optimizer_name, optimizer_params, batch_size, lr, steps,
             measure_every=75, n_probe=4, bs_estimates=4, tag=None):
    X_train, Y_train = get_data()
    presets = get_model_presets()
    ds_presets = get_dataset_presets()
    mparams = dict(presets[MODEL]['params'])
    mparams['input_dim'] = ds_presets['cifar10']['input_dim']
    mparams['output_dim'] = ds_presets['cifar10']['output_dim']
    net = prepare_net(model_type=presets[MODEL]['type'], params=mparams)
    initialize_net(net, scale=0.2, seed=8888)
    loss_fn = SquaredLoss()
    opt = create_optimizer(optimizer_name, net, lr, optimizer_params)

    eig_cache = EigenvectorCache(max_eigenvectors=1)
    log = []
    diverged = False
    t0 = time.time()
    step = -1
    for step in range(steps):
        idx = T.randperm(len(X_train))[:batch_size]
        Xb, Yb = X_train[idx], Y_train[idx]
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        lv = loss.item()
        if not np.isfinite(lv) or abs(lv) > 1e6:
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % measure_every == 0:
            preds2 = net(Xb).squeeze(-1)
            loss2 = loss_fn(preds2, Yb)
            try:
                lmax = compute_eigenvalues(loss2, net, k=1, max_iterations=25, reltol=0.03,
                                            eigenvector_cache=eig_cache,
                                            use_power_iteration=False).item()
            except Exception:
                lmax = float('nan')
            try:
                bs = calculate_averaged_grad_H_grad_step(
                    net, X_train, Y_train, loss_fn, batch_size=batch_size,
                    n_estimates=bs_estimates, min_estimates=bs_estimates, eps=0.5)
            except Exception:
                bs = float('nan')
            gbs_vals = quick_gbs(net, X_train, Y_train, loss_fn, opt, batch_size, n_probe=n_probe)
            gbs = float(np.mean(gbs_vals)) if gbs_vals else float('nan')
            log.append(dict(step=step, loss=lv, lmax=lmax, bs=bs, gbs=gbs))

    dt = time.time() - t0
    result = dict(optimizer=optimizer_name, params=optimizer_params, batch=batch_size, lr=lr,
                  steps_done=step + 1, steps_target=steps, diverged=diverged, wall=dt, log=log)
    verdict = check_stable(result)
    result['verdict'] = verdict
    if tag:
        with open(os.path.join(OUT_DIR, f"{tag}.json"), 'w') as f:
            json.dump(result, f, indent=1)
    return result


def check_stable(result):
    log = result['log']
    if result['diverged']:
        return dict(ok=False, reason='diverged')
    if len(log) < 6:
        return dict(ok=False, reason='too_short')
    n = len(log)
    window = log[n // 2:]

    def trend_ratio(key):
        vals = [r[key] for r in window if np.isfinite(r[key])]
        if len(vals) < 3:
            return float('nan'), []
        h = len(vals) // 2
        a, b = np.mean(vals[:h]) if h > 0 else vals[0], np.mean(vals[h:])
        r = (b / a) if abs(a) > 1e-9 else float('inf')
        return r, vals

    r_gbs, gbs_vals = trend_ratio('gbs')
    r_bs, bs_vals = trend_ratio('bs')
    r_lmax, lmax_vals = trend_ratio('lmax')
    loss_vals = [r['loss'] for r in log]
    loss_ok = all(np.isfinite(v) and abs(v) < 1e6 for v in loss_vals)

    ratios_ok = all(np.isfinite(r) and 0.5 < r < 2.0 for r in [r_gbs, r_bs, r_lmax])
    ok = loss_ok and ratios_ok and len(gbs_vals) >= 3 and len(bs_vals) >= 3

    return dict(
        ok=ok, reason=('stable' if ok else 'trending_or_missing'),
        gbs_mean=float(np.mean(gbs_vals)) if gbs_vals else None,
        bs_mean=float(np.mean(bs_vals)) if bs_vals else None,
        lmax_mean=float(np.mean(lmax_vals)) if lmax_vals else None,
        r_gbs=float(r_gbs) if np.isfinite(r_gbs) else None,
        r_bs=float(r_bs) if np.isfinite(r_bs) else None,
        r_lmax=float(r_lmax) if np.isfinite(r_lmax) else None,
        final_loss=loss_vals[-1] if loss_vals else None,
    )


def main():
    arg = sys.argv[1]
    if os.path.exists(arg):
        with open(arg) as f:
            jobs = json.load(f)
    else:
        jobs = json.loads(arg)
    winners = {}
    for job in jobs:
        cell_name = job.get('cell') or f"{job['optimizer']}_b{job['batch']}_{job.get('params',{})}"
        lrs = job['lr_candidates'] if 'lr_candidates' in job else [job['lr']]
        cell_pass = None
        for lr in lrs:
            tag = job.get('tag') or f"{job['optimizer']}_b{job['batch']}_lr{lr}"
            t0 = time.time()
            result = run_cell(
                job['optimizer'], job.get('params', {}), job['batch'], lr,
                job.get('steps', 1500), measure_every=job.get('measure_every', 75),
                n_probe=job.get('n_probe', 4), bs_estimates=job.get('bs_estimates', 4),
                tag=tag,
            )
            v = result['verdict']
            status = 'PASS' if v['ok'] else f"FAIL({v['reason']})"
            print(f"[{time.time()-t0:5.1f}s] {tag:45s} {status:20s} "
                  f"gbs={v.get('gbs_mean')} bs={v.get('bs_mean')} lmax={v.get('lmax_mean')} "
                  f"r_gbs={v.get('r_gbs')} r_bs={v.get('r_bs')} r_lmax={v.get('r_lmax')} "
                  f"final_loss={v.get('final_loss')} steps_done={result['steps_done']}/{result['steps_target']}",
                  flush=True)
            if v['ok']:
                cell_pass = dict(lr=lr, tag=tag, verdict=v)
                break
        winners[cell_name] = cell_pass if cell_pass else dict(lr=None, tag=None, verdict=None, note='NO_CANDIDATE_PASSED')
        print(f"  -> CELL {cell_name}: "
              f"{'winner lr=' + str(cell_pass['lr']) if cell_pass else 'NO CANDIDATE PASSED'}",
              flush=True)

    winners_path = os.path.join(OUT_DIR, 'winners.json')
    # Merge with any existing winners file so multiple invocations accumulate.
    existing = {}
    if os.path.exists(winners_path):
        try:
            with open(winners_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(winners)
    with open(winners_path, 'w') as f:
        json.dump(existing, f, indent=1)
    print(f"\nWinners written to {winners_path}")


if __name__ == '__main__':
    main()
