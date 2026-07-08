"""
Tangent-propagation growth-factor experiment (suggested.txt Direction 1 / exp #3,
"the principled version") + the (sigma^2, mu)-plane plot (Direction 2, the top pick).

CONCEPT
-------
Every optimizer is a map on an augmented state z=(theta, m, ...):  z_{t+1}=F(z_t; B_t).
Carry a tangent vector dz alongside a REAL training run and push it through the
LINEARIZED step J_t = dF/dz.  Per-step growth factor a_t = ||dz_{t+1}|| / ||dz_t||.
We renormalize dz to unit norm every step and accumulate log(a_t).  Over the plateau
window we fit
    mu     = mean(log a_t)
    sigma2 = var(log a_t, ddof=1)
and also record mean(a_t), mean(a_t^2).

Three stability theories are three lines in the (sigma^2, mu) plane:
    mu = 0          -> almost-sure / log stability   (top Lyapunov exponent 0)
    mu = -sigma^2/2 -> mean stability                (E[a]=1)
    mu = -sigma^2   -> mean-square stability          (E[a^2]=1)
Whichever line the (optimizer,batch) points fall on is the candidate universal invariant.

Hand-written tangent recursions (NO autodiff-through-the-optimizer); each needs exactly
ONE Hessian-vector product per step (the term H_{B_t} . dtheta):

  SGD:            dtheta_{t+1} = dtheta_t - eta*(H_B dtheta_t)
  SGD-Momentum:   dbuf_{t+1}   = beta*dbuf_t + H_B dtheta_t
                  dtheta_{t+1} = dtheta_t - eta*dbuf_{t+1}
  Adam (frozen-v):dm_{t+1}     = beta1*dm_t + (1-beta1)*H_B dtheta_t
                  dtheta_{t+1} = dtheta_t - eta*(1/(sqrt(vhat)+eps))*(dm_{t+1}/(1-beta1^{t+1}))

Usage:
    python experiments/tangent_propagation.py            # runs the full 9-cell grid
    python experiments/tangent_propagation.py SGD_b8 ... # run only named cells
"""
import os, sys, time, json, math

import numpy as np
import torch as T

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')

# torchvision checksum monkeypatch (dataset was placed manually, no checksum).
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.measure import create_hessian_vector_product, flatt

T.set_num_threads(4)

DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = 2048          # matches FINAL_GRID.json dataset block
MODEL = 'mlp_s'
DATASET_SEED = 888
INIT_SEED = 8888
INIT_SCALE = 0.2
STEPS_MULT = 1.3         # run ~1.3x calib_steps
OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'tangent_prop')
os.makedirs(OUT_DIR, exist_ok=True)

_DATA_CACHE = {}


def get_data():
    if 'xy' not in _DATA_CACHE:
        data = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], DATASET_SEED, loss_type='mse')
        X_train, Y_train, _, _ = data
        _DATA_CACHE['xy'] = (X_train, Y_train)
    return _DATA_CACHE['xy']


# ----------------------------------------------------------------------------- #
#   The 9-cell grid, straight from results/calib2/FINAL_GRID.json:base_grid     #
#   (Muon deferred).  lr / params USED AS-IS, never retuned.                     #
# ----------------------------------------------------------------------------- #
GRID = {
    'SGD_b8':      dict(optimizer='SGD',          params={},                       batch=8,    lr=0.01,  calib_steps=2500),
    'SGD_b128':    dict(optimizer='SGD',          params={},                       batch=128,  lr=0.02,  calib_steps=1800),
    'SGD_b2048':   dict(optimizer='SGD',          params={},                       batch=2048, lr=0.02,  calib_steps=1200),
    'SGDM09_b8':   dict(optimizer='SGD-Momentum', params={'beta': 0.9},            batch=8,    lr=0.002, calib_steps=3000),
    'SGDM09_b128': dict(optimizer='SGD-Momentum', params={'beta': 0.9},            batch=128,  lr=0.003, calib_steps=2200),
    # NOTE: lr=0.006 NOT 0.0015 -- 0.0015 is a false-plateau trap (see FINAL_GRID note).
    'SGDM09_b2048':dict(optimizer='SGD-Momentum', params={'beta': 0.9},            batch=2048, lr=0.006, calib_steps=3500),
    'Adam_b8':     dict(optimizer='Adam',         params={'beta1': 0.9,'beta2':0.99}, batch=8,    lr=0.001, calib_steps=3000),
    'Adam_b128':   dict(optimizer='Adam',         params={'beta1': 0.9,'beta2':0.99}, batch=128,  lr=0.001, calib_steps=2200),
    'Adam_b2048':  dict(optimizer='Adam',         params={'beta1': 0.9,'beta2':0.99}, batch=2048, lr=0.001, calib_steps=1500),
}


def build_net():
    presets = get_model_presets()
    ds_presets = get_dataset_presets()
    mparams = dict(presets[MODEL]['params'])
    mparams['input_dim'] = ds_presets['cifar10']['input_dim']
    mparams['output_dim'] = ds_presets['cifar10']['output_dim']
    net = prepare_net(model_type=presets[MODEL]['type'], params=mparams)
    initialize_net(net, scale=INIT_SCALE, seed=INIT_SEED)
    return net


def run_cell(cell_name, spec, tangent_seed=1234, verbose=True, n_pi=3):
    """Tangent-propagation growth factors, restricted to the realized unstable mode.

    A pure full-parameter tangent is DEGENERATE here: the ~10^6-dim flat null-space
    (H.flat approx 0 -> Jacobian multiplier exactly +1) is co-marginal with the
    unstable direction at the edge, and any leakage into it is frozen (invariant under
    J), so the renormalized tangent collapses onto flat space within ~100 steps giving
    mu approx 0, sigma^2 approx 0 for every cell (verified empirically).

    So we run the SAME hand-written tangent recursion but PROJECTED onto the instantaneous
    top Hessian eigenvector u_t of H_{B_t} (re-found each step by warm-started power
    iteration). This is exactly the scalar/companion-matrix EoS reduction:
        h_t = u_t^T H_{B_t} u_t   (directional batch curvature = top batch-sharpness eigval)
      SGD (state x):            a_t = |1 - eta*h_t|
      SGD-Mom (state x,b):      b' = beta*b + h_t*x ;  x' = x - eta*b'
      Adam frozen-v (x,b):      b' = beta1*b + (1-beta1)*h_t*x
                                x' = x - eta * p_u * b'/(1-beta1^{t+1})
                                with p_u = u_t^T diag(1/(sqrt(vhat)+eps)) u_t
    Each step costs n_pi+1 HVPs (all warm-started). a_t = ||state'||/||state|| with renorm.
    """
    optimizer_name = spec['optimizer']
    optimizer_params = spec['params']
    batch_size = spec['batch']
    lr = spec['lr']
    calib_steps = spec['calib_steps']
    total_steps = int(round(STEPS_MULT * calib_steps))
    accum_from = calib_steps        # only accumulate stats once past plateau

    X_train, Y_train = get_data()
    N = len(X_train)
    net = build_net()
    loss_fn = SquaredLoss()
    opt = create_optimizer(optimizer_name, net, lr, optimizer_params)
    params = [p for p in net.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    is_momentum = (optimizer_name == 'SGD-Momentum')
    is_adam = (optimizer_name == 'Adam')
    beta = optimizer_params.get('beta', 0.9) if is_momentum else None
    beta1 = optimizer_params.get('beta1', 0.9) if is_adam else None
    reduced_2d = is_momentum or is_adam

    g = T.Generator().manual_seed(tangent_seed)
    u = T.randn(P, generator=g); u /= u.norm()          # top-eigvec estimate (warm-started)
    # reduced tangent state on the unstable mode
    if reduced_2d:
        s2 = T.randn(2, generator=g); s2 /= s2.norm()   # (x, b)
    else:
        s2 = None

    log_a = []          # log growth factors (plateau window only)
    a_list = []
    h_list = []         # directional batch curvature (top batch-sharpness eigenvalue)
    loss_trace = []
    t0 = time.time()
    diverged = False

    for step in range(total_steps):
        idx = T.randperm(N)[:batch_size]
        Xb, Yb = X_train[idx], Y_train[idx]
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        lv = loss.item()
        if not np.isfinite(lv) or abs(lv) > 1e6:
            diverged = True
            break

        # gradient WITH graph for the HVP(s)
        grads = T.autograd.grad(loss, params, create_graph=True)
        g_flat = flatt(grads)
        hvp = create_hessian_vector_product(loss, net, params=params, grads=grads, flat_grads=g_flat)
        try:
            # warm-started power iteration -> top eigvec u_t of H_{B_t}
            iters = 20 if step == 0 else n_pi
            Hu = None
            for _ in range(iters):
                Hu = hvp(u, retain_graph_override=True).detach()
                nrm = Hu.norm()
                if nrm < 1e-20:
                    break
                u = Hu / nrm
            Hu = hvp(u, retain_graph_override=False).detach()   # last use -> free graph
        finally:
            hvp.free_memory()
        h = float(T.dot(u, Hu).item())      # Rayleigh quotient = directional curvature

        # preconditioner projected onto u (Adam only)
        if is_adam:
            dinv_sqrt = opt.get_preconditioner_inv_sqrt()   # D^{-1/2}, flat (uses live v)
            if dinv_sqrt is None:
                p_u = 1.0
            else:
                pinv = dinv_sqrt * dinv_sqrt                 # 1/(sqrt(vhat)+eps)
                p_u = float((pinv * u * u).sum().item())

        # advance the REAL iterate
        for p, gp in zip(params, grads):
            p.grad = gp.detach()
        opt.step()

        # --- hand-written tangent recursion on the unstable mode ---
        if is_momentum:
            x, b = float(s2[0]), float(s2[1])
            b_new = beta * b + h * x
            x_new = x - lr * b_new
            s2_new = T.tensor([x_new, b_new])
            a = float(s2_new.norm().item())
        elif is_adam:
            st = opt.inner.state.get(params[0])
            t_count = st['step'].item() if (st and 'step' in st) else (step + 1)
            bc1 = 1.0 - beta1 ** t_count
            x, b = float(s2[0]), float(s2[1])
            b_new = beta1 * b + (1.0 - beta1) * h * x
            x_new = x - lr * p_u * (b_new / bc1)
            s2_new = T.tensor([x_new, b_new])
            a = float(s2_new.norm().item())
        else:  # SGD
            a = abs(1.0 - lr * h)

        if not np.isfinite(a) or a <= 0:
            diverged = True
            break
        if reduced_2d:
            s2 = s2_new / a

        if step >= accum_from:
            log_a.append(math.log(a))
            a_list.append(a)
            h_list.append(h)

        if verbose and (step % max(1, total_steps // 20) == 0):
            mu_now = float(np.mean(log_a)) if log_a else float('nan')
            print(f"    [{cell_name}] step {step:5d}/{total_steps} "
                  f"loss={lv:.4e} h={h:8.2f} a={a:.5f} mu_sofar={mu_now:+.4e} "
                  f"n_accum={len(log_a)} ({time.time()-t0:5.1f}s)", flush=True)
        loss_trace.append(lv)

    dt = time.time() - t0
    log_a = np.asarray(log_a, dtype=np.float64)
    a_arr = np.asarray(a_list, dtype=np.float64)
    h_arr = np.asarray(h_list, dtype=np.float64)

    if len(log_a) >= 3 and not diverged:
        mu = float(np.mean(log_a))
        sigma2 = float(np.var(log_a, ddof=1))
        mean_a = float(np.mean(a_arr))
        mean_a2 = float(np.mean(a_arr ** 2))
    else:
        mu = sigma2 = mean_a = mean_a2 = float('nan')

    result = dict(
        cell=cell_name, optimizer=optimizer_name, params=optimizer_params,
        batch=batch_size, lr=lr, calib_steps=calib_steps, total_steps=total_steps,
        accum_from=accum_from, n_params=P, n_accum=int(len(log_a)),
        diverged=bool(diverged), wall=dt,
        mu=mu, sigma2=sigma2, mean_a=mean_a, mean_a2=mean_a2,
        exp_mu=float(math.exp(mu)) if np.isfinite(mu) else float('nan'),
        exp_mu_plus_half_s2=float(math.exp(mu + 0.5 * sigma2)) if np.isfinite(mu) else float('nan'),
        # directional batch curvature (top batch-sharpness eigenvalue) diagnostics:
        h_mean=float(np.mean(h_arr)) if len(h_arr) else float('nan'),
        h_std=float(np.std(h_arr, ddof=1)) if len(h_arr) > 1 else float('nan'),
        edge_2_over_lr=2.0 / lr,
        eta_h_mean=float(lr * np.mean(h_arr)) if len(h_arr) else float('nan'),
        final_loss=float(loss_trace[-1]) if loss_trace else None,
    )
    # dump per-cell log (raw log_a for reproducibility / re-fitting)
    with open(os.path.join(OUT_DIR, f'{cell_name}.json'), 'w') as f:
        json.dump({**result, 'log_a': log_a.tolist()}, f, indent=1)
    return result


def make_plot(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'(matplotlib unavailable, skipping PNG: {e})')
        return

    opt_colors = {'SGD': '#4C78A8', 'SGD-Momentum': '#F58518', 'Adam': '#54A24B'}
    batch_size_marker = {8: 60, 128: 130, 2048: 260}

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    s2max = max([r['sigma2'] for r in results if np.isfinite(r['sigma2'])] + [1e-6])
    xs = np.linspace(0, s2max * 1.05, 100)
    ax.plot(xs, 0 * xs, '--', color='#555', lw=1.3, label=r'$\mu=0$  (a.s./log)')
    ax.plot(xs, -0.5 * xs, '-.', color='#c00', lw=1.3, label=r'$\mu=-\sigma^2/2$  (mean)')
    ax.plot(xs, -1.0 * xs, ':', color='#00a', lw=1.6, label=r'$\mu=-\sigma^2$  (mean-square)')

    seen = set()
    for r in results:
        if not np.isfinite(r['sigma2']) or not np.isfinite(r['mu']):
            continue
        c = opt_colors.get(r['optimizer'], '#888')
        m = batch_size_marker.get(r['batch'], 100)
        lbl = r['optimizer'] if r['optimizer'] not in seen else None
        seen.add(r['optimizer'])
        ax.scatter(r['sigma2'], r['mu'], s=m, color=c, edgecolor='k', linewidth=0.6,
                   zorder=5, label=lbl)
        ax.annotate(f"b{r['batch']}", (r['sigma2'], r['mu']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

    ax.axhline(0, color='#ccc', lw=0.6, zorder=0)
    ax.axvline(0, color='#ccc', lw=0.6, zorder=0)
    ax.set_xlabel(r'$\sigma^2=\mathrm{Var}[\log a_t]$')
    ax.set_ylabel(r'$\mu=\mathbb{E}[\log a_t]$')
    ax.set_title('Tangent-propagation growth factors: $(\\sigma^2,\\mu)$ plane\n'
                 'marker size = batch (8/128/2048); color = optimizer')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    png = os.path.join(OUT_DIR, 'sigma2_mu_plane.png')
    fig.savefig(png, dpi=130)
    print(f'  wrote {png}')


def which_line(results):
    """For each cell, report residual distance to each theory line and pick the closest
    aggregated over the small-batch (fanned-out) cells."""
    lines = {
        'mu=0 (a.s.)':        lambda s2: 0.0,
        'mu=-s2/2 (mean)':    lambda s2: -0.5 * s2,
        'mu=-s2 (mean-sq)':   lambda s2: -1.0 * s2,
    }
    # weight by sigma2 (only cells with meaningful spread discriminate the lines)
    tot = {k: 0.0 for k in lines}
    wtot = 0.0
    for r in results:
        if not (np.isfinite(r['sigma2']) and np.isfinite(r['mu'])):
            continue
        w = r['sigma2']
        wtot += w
        for k, f in lines.items():
            tot[k] += w * (r['mu'] - f(r['sigma2'])) ** 2
    rmse = {k: (math.sqrt(v / wtot) if wtot > 0 else float('nan')) for k, v in tot.items()}
    return rmse


def main():
    wanted = sys.argv[1:] if len(sys.argv) > 1 else list(GRID.keys())
    results = []
    grand0 = time.time()
    for cell_name in wanted:
        if cell_name not in GRID:
            print(f'!! unknown cell {cell_name}, skipping')
            continue
        spec = GRID[cell_name]
        print(f'\n=== CELL {cell_name}  ({spec["optimizer"]} b{spec["batch"]} '
              f'lr={spec["lr"]} steps~{int(round(STEPS_MULT*spec["calib_steps"]))}) ===', flush=True)
        r = run_cell(cell_name, spec)
        results.append(r)
        print(f'  -> mu={r["mu"]:+.5f}  sigma2={r["sigma2"]:.5f}  '
              f'mean_a={r["mean_a"]:.5f}  mean_a2={r["mean_a2"]:.5f}  '
              f'exp_mu={r["exp_mu"]:.5f}  n={r["n_accum"]}  '
              f'{"DIVERGED " if r["diverged"] else ""}({r["wall"]:.1f}s)', flush=True)

    # summary table
    hdr = f"{'cell':13s} {'opt':13s} {'batch':>5s} {'mu':>10s} {'sigma2':>10s} " \
          f"{'mean_a':>9s} {'mean_a2':>9s} {'exp_mu':>9s} {'n':>5s}"
    lines_out = [hdr, '-' * len(hdr)]
    for r in results:
        lines_out.append(
            f"{r['cell']:13s} {r['optimizer']:13s} {r['batch']:5d} "
            f"{r['mu']:+10.5f} {r['sigma2']:10.5f} {r['mean_a']:9.5f} "
            f"{r['mean_a2']:9.5f} {r['exp_mu']:9.5f} {r['n_accum']:5d}")
    rmse = which_line(results)
    lines_out.append('')
    lines_out.append('sigma2-weighted RMSE of points to each theory line (smaller = better fit):')
    for k, v in sorted(rmse.items(), key=lambda kv: kv[1]):
        lines_out.append(f'    {k:22s} RMSE={v:.5f}')
    best = min(rmse, key=rmse.get)
    lines_out.append(f'  => points best follow: {best}')
    table = '\n'.join(lines_out)
    print('\n' + table, flush=True)

    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(dict(results=results, line_rmse=rmse, best_line=best,
                       total_wall=time.time() - grand0), f, indent=1)
    make_plot(results)
    print(f'\nTotal wall {time.time()-grand0:.1f}s. Outputs in {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
