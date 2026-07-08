"""
Experiment 1 (definitive): long-training GBS stationarity + regime characterization.

The earlier grid trained small-batch cells only ~3k steps; the reconciliation run
showed SGD_b8's GBS rises 1.25 -> ~1.85 by 12k steps. So absolute small-batch
deficits were under-trained. This reruns to near-stationarity (~30k steps for b8)
and answers, RIGOROUSLY:

  (Q1) Does per-batch GBS stabilize, and AT WHAT VALUE? -> stationarity test:
       over the back window, report GBS mean +/- SEM and the linear slope per 1k
       steps; flag "stabilized" only if the slope is small vs the mean.
  (Q2) Dichotomy: does the momentum/Adam/Muon deficit CLOSE toward ~2 like SGD
       (=> under-training artifact) or PLATEAU below (=> real regime dichotomy)?
  (Q3) The loss-stationarity baseline GBS_full = E[s^T H_full s]/E[-g_full^T s]
       (full-batch H,g with the actual stochastic step). At ANY stationary state
       (loss not trending) this MUST be ~2 for every optimizer -- it's just loss
       stationarity. The gap (GBS_full - GBS_batch) isolates the cross-batch
       interference / noise term that per-batch GBS carries and full-batch doesn't.
  (Q4) Regime signature: kappa_t = eta*lambda(H_B) distribution (marginal SGD
       sits ~2 symmetric; metastable momentum sits below 2, right-skewed with rare
       spikes) and the catapult rate (per-step loss spikes).

lean setup (mlp_s, num_data=2048) for speed so we can afford 30k steps + rich probes.
"""
import os, sys, json, time
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.measure import compute_eigenvalues, EigenvectorCache, create_hessian_vector_product, flatt

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = int(os.environ.get('EOSS_NUM_DATA', '2048'))
MODEL = os.environ.get('EOSS_MODEL', 'mlp_s')
OUT_DIR = os.path.join(_REPO, 'results', 'long_train_grid')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


def get_data():
    if 'xy' not in _DATA:
        X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], 888, loss_type='mse')
        _DATA['xy'] = (X, Y)
    return _DATA['xy']


def build():
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets[MODEL]['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets[MODEL]['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    return net, SquaredLoss()


def probe(net, X, Y, loss_fn, opt, lr, batch, n_probe=24, full_cap=2048):
    """Per checkpoint: GBS_batch, GBS_full (loss-stationarity baseline), kappa, and
    the per-probe GBS_batch / lambda arrays for distribution stats."""
    params = [p for p in net.parameters() if p.requires_grad]
    cache = EigenvectorCache(1)
    N = len(X)
    # full-batch g_full and H_full operator (one graph, applied to each s_B)
    Xf, Yf = (X, Y) if N <= full_cap else (X[:full_cap], Y[:full_cap])
    pf = net(Xf).squeeze(-1); lf = loss_fn(pf, Yf)
    gfull = T.autograd.grad(lf, params, create_graph=True)
    gfull_flat = flatt(gfull); gfull_d = gfull_flat.detach()
    hvp_full = create_hessian_vector_product(lf, net, params=params, grads=gfull, flat_grads=gfull_flat)

    Gbs, Lam, Bfull_num, Afull_den = [], [], [], []
    try:
        for _ in range(n_probe):
            idx = T.randperm(N)[:batch]
            Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            grads = T.autograd.grad(lo, params, create_graph=True)
            g = flatt(grads); gd = g.detach()
            s = opt.compute_step_direction(g, params).detach()
            hvp_b = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
            try:
                Hs = hvp_b(s, retain_graph_override=False)
                gbs = (T.dot(s, Hs) / (-T.dot(gd, s) + 1e-30)).item()
            finally:
                hvp_b.free_memory()
            # GBS_full pieces: s^T H_full s and -g_full^T s (accumulate for E[num]/E[den])
            Hfs = hvp_full(s, retain_graph_override=True)
            Bfull_num.append(T.dot(s, Hfs).item())
            Afull_den.append((-T.dot(gfull_d, s)).item())
            # lambda(H_B)
            pr2 = net(Xb).squeeze(-1); lo2 = loss_fn(pr2, Yb)
            try:
                lam = compute_eigenvalues(lo2, net, k=1, max_iterations=40, reltol=0.02,
                                          eigenvector_cache=cache, return_eigenvectors=False,
                                          use_power_iteration=False).item()
            except Exception:
                lam = float('nan')
            if np.isfinite(gbs):
                Gbs.append(gbs)
            if np.isfinite(lam) and lam > 0:
                Lam.append(lam)
    finally:
        hvp_full.free_memory()

    Gbs, Lam = np.array(Gbs), np.array(Lam)
    Bfull_num, Afull_den = np.array(Bfull_num), np.array(Afull_den)
    gbs_full = float(np.mean(Bfull_num) / (np.mean(Afull_den) + 1e-30)) if len(Bfull_num) else float('nan')
    return dict(
        gbs=float(Gbs.mean()), gbs_med=float(np.median(Gbs)), gbs_std=float(Gbs.std()),
        gbs_full=gbs_full, lam=float(Lam.mean()) if len(Lam) else float('nan'),
        kappa=float(lr * Lam.mean()) if len(Lam) else float('nan'),
        kappa_med=float(lr * np.median(Lam)) if len(Lam) else float('nan'),
        kappa_q90=float(lr * np.quantile(Lam, 0.9)) if len(Lam) else float('nan'),
        frac_kappa_gt2=float(np.mean(lr * Lam > 2.0)) if len(Lam) else float('nan'),
        n=len(Gbs),
    )


def stationarity(traj, key='gbs', back_frac=0.4):
    """Rigorous plateau test on the back `back_frac` of checkpoints."""
    if len(traj) < 8:
        return dict(stabilized=False, reason='too_short')
    n = len(traj)
    w = traj[int(n * (1 - back_frac)):]
    steps = np.array([t['step'] for t in w], dtype=float)
    vals = np.array([t[key] for t in w], dtype=float)
    m = np.isfinite(vals)
    steps, vals = steps[m], vals[m]
    if len(vals) < 5:
        return dict(stabilized=False, reason='few_valid')
    mean = float(vals.mean()); sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    slope, intc = np.polyfit(steps, vals, 1)
    drift_over_window = float(slope * (steps[-1] - steps[0]))   # total change across back window
    # stabilized if the drift across the whole back window is small vs the level
    stabilized = abs(drift_over_window) < 0.15 * abs(mean) and sem < 0.12 * abs(mean)
    return dict(stabilized=bool(stabilized), plateau_mean=mean, plateau_sem=sem,
                slope_per_step=float(slope), drift_over_window=drift_over_window,
                n_window=len(vals))


def detect_catapults(losses, K=3.0, win=50, refractory=20):
    """Count loss spikes > K * trailing median, with a refractory period."""
    losses = np.asarray(losses)
    onsets = []
    last = -refractory
    for t in range(win, len(losses)):
        base = np.median(losses[t - win:t])
        if losses[t] > K * base and (t - last) > refractory and np.isfinite(base) and base > 0:
            onsets.append(t); last = t
    return onsets


def run_cell(tag, optn, params, batch, lr, steps, measure_every):
    X, Y = get_data()
    net, loss_fn = build()
    opt = create_optimizer(optn, net, lr, params)
    traj, losses, diverged = [], [], False
    t0 = time.time()
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item(); losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            diverged = True; print(f"  [{tag}] DIVERGED @ {step}", flush=True); break
        opt.zero_grad(); lo.backward(); opt.step()
        if step % measure_every == 0 and step > 0:
            m = probe(net, X, Y, loss_fn, opt, lr, batch)
            m['step'] = step; m['loss'] = lv; traj.append(m)
            print(f"  [{tag}] {step:6d} loss={lv:.4f} GBS={m['gbs']:.3f} GBSfull={m['gbs_full']:.3f} "
                  f"kappa_med={m['kappa_med']:.2f} f(k>2)={m['frac_kappa_gt2']:.2f}", flush=True)
    st = stationarity(traj, 'gbs')
    st_full = stationarity(traj, 'gbs_full')
    # catapult rate over the plateau (back 40% of steps)
    plateau_start = int(len(losses) * 0.6)
    cats = detect_catapults(losses[plateau_start:])
    cat_rate = len(cats) / max(1, len(losses) - plateau_start)
    tailk = [t for t in traj[int(len(traj)*0.6):]] if traj else []
    kappa_med = float(np.median([t['kappa_med'] for t in tailk])) if tailk else float('nan')
    frac_k = float(np.mean([t['frac_kappa_gt2'] for t in tailk])) if tailk else float('nan')
    print(f"  [{tag}] done {time.time()-t0:.0f}s  STABILIZED={st.get('stabilized')} "
          f"GBS={st.get('plateau_mean',float('nan')):.3f}+/-{st.get('plateau_sem',float('nan')):.3f} "
          f"(drift={st.get('drift_over_window',float('nan')):.3f})  "
          f"GBSfull={st_full.get('plateau_mean',float('nan')):.3f}  "
          f"kappa_med={kappa_med:.2f} frac(k>2)={frac_k:.2f} cat_rate={cat_rate:.4f} div={diverged}", flush=True)
    return dict(tag=tag, optimizer=optn, params=params, batch=batch, lr=lr, steps=steps,
                diverged=diverged, stationarity=st, stationarity_full=st_full,
                plateau_kappa_med=kappa_med, plateau_frac_kappa_gt2=frac_k,
                catapult_rate=cat_rate, n_catapults=len(cats), traj=traj)


def plot_cell(r):
    traj = r['traj']
    if not traj:
        return
    steps = [t['step'] for t in traj]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [t['gbs'] for t in traj], '-o', ms=3, label='GBS (batch)')
    ax[0].plot(steps, [t['gbs_full'] for t in traj], '-s', ms=3, label='GBS_full (loss-stat.)')
    ax[0].axhline(2.0, color='k', ls='--', lw=0.8)
    st = r['stationarity']
    if st.get('stabilized'):
        ax[0].axhspan(st['plateau_mean']-st['plateau_sem'], st['plateau_mean']+st['plateau_sem'],
                      color='green', alpha=0.15)
    ax[0].set_title(f"{r['tag']}: GBS={st.get('plateau_mean',float('nan')):.2f} stab={st.get('stabilized')}")
    ax[0].set_xlabel('step'); ax[0].legend(fontsize=8)
    ax[1].plot(steps, [t['kappa_med'] for t in traj], '-o', ms=3, label='median kappa')
    ax[1].plot(steps, [t['frac_kappa_gt2'] for t in traj], '-^', ms=3, label='frac(kappa>2)')
    ax[1].axhline(2.0, color='k', ls='--', lw=0.8); ax[1].axhline(0.5, color='gray', ls=':', lw=0.8)
    ax[1].set_title(f"regime: kappa_med={r['plateau_kappa_med']:.2f} cat_rate={r['catapult_rate']:.4f}")
    ax[1].set_xlabel('step'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{r['tag']}.png"), dpi=90); plt.close(fig)


def main():
    cells = [
        ("SGD_b8",    "SGD",          {},                            8,    0.01,  30000, 1000),
        ("SGDM09_b8", "SGD-Momentum", {"beta": 0.9},                 8,    0.002, 30000, 1000),
        ("Adam_b8",   "Adam",         {"beta1": 0.9, "beta2": 0.99}, 8,    0.001, 30000, 1000),
        ("Muon_b8",   "Muon",         {"momentum": 0.9},             8,    0.003, 30000, 1000),
        ("SGD_b2048", "SGD",          {},                            2048, 0.02,  6000,  300),
        ("SGDM09_b2048","SGD-Momentum",{"beta": 0.9},                2048, 0.006, 6000,  300),
    ]
    results = []
    for tag, optn, params, batch, lr, steps, me in cells:
        print(f"\n=== {tag}: {optn} b={batch} lr={lr} steps={steps} ===", flush=True)
        r = run_cell(tag, optn, params, batch, lr, steps, me)
        results.append(r)
        plot_cell(r)
        with open(os.path.join(OUT_DIR, 'long_train_grid.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n===== VERDICT =====")
    print(f"{'cell':14s} {'stab':>5s} {'GBS_batch':>18s} {'GBS_full':>9s} {'kappa_med':>9s} {'f(k>2)':>7s} {'cat_rate':>9s} {'div':>4s}")
    for r in results:
        st = r['stationarity']; stf = r['stationarity_full']
        gbs = f"{st.get('plateau_mean',float('nan')):.3f}+/-{st.get('plateau_sem',float('nan')):.3f}"
        print(f"{r['tag']:14s} {str(st.get('stabilized')):>5s} {gbs:>18s} "
              f"{stf.get('plateau_mean',float('nan')):9.3f} {r['plateau_kappa_med']:9.2f} "
              f"{r['plateau_frac_kappa_gt2']:7.2f} {r['catapult_rate']:9.4f} {str(r['diverged']):>4s}")
    print("\nReading: GBS_batch->2 for all => under-training was culprit (GBS universal).")
    print("         GBS_batch plateaus <2 for momentum while GBS_full~2 => dichotomy; deficit=cross-batch interference.")
    print("         kappa_med~2 & frac(k>2)~0.5 => marginal (SGD); kappa_med<2 & rare spikes & low cat_rate => metastable.")


if __name__ == '__main__':
    main()
