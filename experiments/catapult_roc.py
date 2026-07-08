"""
Catapult ROC (suggested.txt ranked experiment #7): the clean, non-circular
CERTIFICATE test.

Question
--------
"log-stability" (E[log a_t]=0) DESCRIBES the bounded EoS trajectory but is partly
tautological (any bounded path has realized log-growth ~0). We want a CERTIFICATE:
a quantity whose threshold-crossing PREDICTS instability. Catapults -- transient
batch-loss spikes at the edge -- are within-run ground-truth instability events.

We run cells that oscillate hard but do NOT fully diverge, log per-step candidate
quantities, detect catapult onsets, and score each candidate by how well its
threshold-crossing predicts an IMMINENT catapult (ROC/AUC + event-level
precision/recall at its natural threshold).

Candidate quantities (all from one HVP + warm-started power iteration per step,
exactly like experiments/tangent_propagation.py):
  lambda_B = h_t = u_t^T H_{B_t} u_t   (per-batch directional curvature = batch
                                        sharpness along the top eigvec)
             natural crossing:  h_t > 2/eta
  GBS_t    = s_B^T H_B s_B / (-g_B^T s_B)  (s_B = optimizer.compute_step_direction)
             natural crossing:  GBS_t > 2
  a_t      = |1 - eta*h_t|            (SGD)   -- growth factor of the scalar mode
             rho_t = spectral radius of companion [[1+beta-eta*h_t, -beta],[1,0]]
                                       (momentum)
             natural crossing:  a_t / rho_t > 1   (equivalently log > 0)
  EMA variants of lambda_B and GBS (single-step crossings are noisy).

Usage:
    python experiments/catapult_roc.py              # scan + score default cells
    python experiments/catapult_roc.py SGD_b8_lr0.017 ...   # named cells only
"""
import os, sys, time, json, math

import numpy as np
import torch as T

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')

# torchvision checksum monkeypatch (dataset placed manually, no checksum).
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.measure import create_hessian_vector_product, flatt

T.set_num_threads(4)

DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = 2048
MODEL = 'mlp_s'
DATASET_SEED = 888
INIT_SEED = 8888
INIT_SCALE = 0.2
OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'catapult_roc')
os.makedirs(OUT_DIR, exist_ok=True)

_DATA_CACHE = {}


def get_data():
    if 'xy' not in _DATA_CACHE:
        data = prepare_dataset('cifar10', DATASET_FOLDER, NUM_DATA, [], DATASET_SEED, loss_type='mse')
        X_train, Y_train, _, _ = data
        _DATA_CACHE['xy'] = (X_train, Y_train)
    return _DATA_CACHE['xy']


def build_net():
    presets = get_model_presets()
    ds_presets = get_dataset_presets()
    mparams = dict(presets[MODEL]['params'])
    mparams['input_dim'] = ds_presets['cifar10']['input_dim']
    mparams['output_dim'] = ds_presets['cifar10']['output_dim']
    net = prepare_net(model_type=presets[MODEL]['type'], params=mparams)
    initialize_net(net, scale=INIT_SCALE, seed=INIT_SEED)
    return net


# ----------------------------------------------------------------------------- #
#   Candidate cells: oscillate-hard-but-bounded, pushed just past the calibrated #
#   stable lr so that catapults are frequent (see suggested.txt exp #7 hints).   #
# ----------------------------------------------------------------------------- #
# Each cell is one (optimizer, batch, lr) regime, run over a LIST of batch seeds and
# POOLED, because the near-edge state is metastable: it produces several transient
# catapults (loss spikes that recover) then eventually a terminal runaway. We truncate
# each run at the first non-recovering onset (the runaway) and pool the recovered
# catapults across seeds to get enough events (>=15) for a stable ROC.
_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
GRID = {
    # SGD b8: calibrated stable lr 0.01-0.015; 0.017-0.018 = catapult-rich-then-diverge.
    # lr0.017/K3 pools ~17 recovered catapults over 8 seeds; lr0.018/K3 ~8.
    'SGD_b8_lr0.017':  dict(optimizer='SGD', params={}, batch=8, lr=0.017, steps=5000, seeds=_SEEDS),
    'SGD_b8_lr0.018':  dict(optimizer='SGD', params={}, batch=8, lr=0.018, steps=5000, seeds=_SEEDS),
    # SGD-Momentum b8: calibrated stable lr 0.002; lr0.0023/K3 pools ~23 catapults.
    'SGDM09_b8_lr0.0023': dict(optimizer='SGD-Momentum', params={'beta': 0.9}, batch=8,
                               lr=0.0023, steps=5000, seeds=_SEEDS),
}


def companion_rho(h, eta, beta):
    """Spectral radius of the heavy-ball companion matrix [[1+beta-eta*h, -beta],[1,0]]."""
    tr = 1.0 + beta - eta * h
    det = beta
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        sq = math.sqrt(disc)
        return max(abs((tr + sq) / 2.0), abs((tr - sq) / 2.0))
    return math.sqrt(det)  # complex-conjugate pair, modulus = sqrt(det)


def run_cell(cell_name, spec, tangent_seed=1234, n_pi=3, log_every=1, verbose=True):
    optimizer_name = spec['optimizer']
    optimizer_params = spec['params']
    batch_size = spec['batch']
    lr = spec['lr']
    steps = spec['steps']
    batch_seed = spec.get('batch_seed', 0)   # seed the minibatch order for reproducibility

    X_train, Y_train = get_data()
    N = len(X_train)
    net = build_net()
    loss_fn = SquaredLoss()
    opt = create_optimizer(optimizer_name, net, lr, optimizer_params)
    params = [p for p in net.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    is_momentum = (optimizer_name == 'SGD-Momentum')
    beta = optimizer_params.get('beta', 0.9) if is_momentum else None

    bgen = T.Generator().manual_seed(batch_seed)
    g = T.Generator().manual_seed(tangent_seed)
    u = T.randn(P, generator=g); u /= u.norm()

    rec = {k: [] for k in ('step', 'loss', 'lambda_B', 'GBS', 'growth')}
    t0 = time.time()
    diverged = False

    for step in range(steps):
        idx = T.randperm(N, generator=bgen)[:batch_size]
        Xb, Yb = X_train[idx], Y_train[idx]
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        lv = loss.item()
        if not np.isfinite(lv) or abs(lv) > 1e6:
            diverged = True
            break

        do_log = (step % log_every == 0)
        if do_log:
            grads = T.autograd.grad(loss, params, create_graph=True)
            g_flat = flatt(grads)
            hvp = create_hessian_vector_product(loss, net, params=params,
                                                grads=grads, flat_grads=g_flat)
            try:
                # warm-started power iteration -> top eigvec u_t of H_{B_t}
                iters = 25 if step == 0 else n_pi
                for _ in range(iters):
                    Hu = hvp(u, retain_graph_override=True).detach()
                    nrm = Hu.norm()
                    if nrm < 1e-20:
                        break
                    u = Hu / nrm
                Hu = hvp(u, retain_graph_override=True).detach()
                h = float(T.dot(u, Hu).item())          # lambda_B = directional curvature
                # GBS_t = s^T H_B s / (-g^T s)
                s = opt.compute_step_direction(g_flat, params)
                Hs = hvp(s, retain_graph_override=False).detach()
                A = float(T.dot(g_flat.detach(), s).item())   # g^T s  (<0)
                B = float(T.dot(s, Hs).item())                # s^T H s
                gbs = (B / (-A)) if abs(A) > 1e-15 else float('nan')
            finally:
                hvp.free_memory()

            if is_momentum:
                growth = companion_rho(h, lr, beta)
            else:
                growth = abs(1.0 - lr * h)

            for p_, gp in zip(params, grads):
                p_.grad = gp.detach()
            opt.step()

            rec['step'].append(step)
            rec['loss'].append(lv)
            rec['lambda_B'].append(h)
            rec['GBS'].append(gbs)
            rec['growth'].append(growth)
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and (step % max(1, steps // 15) == 0):
            print(f"    [{cell_name}] step {step:5d}/{steps} loss={lv:.4e} "
                  f"({time.time()-t0:5.1f}s)", flush=True)

    dt = time.time() - t0
    out = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    out['step'] = out['step'].astype(np.int64)
    meta = dict(cell=cell_name, optimizer=optimizer_name, params=optimizer_params,
                batch=batch_size, lr=lr, steps=steps, n_params=P,
                is_momentum=is_momentum, beta=beta if is_momentum else None,
                diverged=bool(diverged), wall=dt,
                edge_2_over_lr=2.0 / lr, n_logged=len(out['step']),
                final_loss=float(out['loss'][-1]) if len(out['loss']) else None,
                loss_max=float(np.max(out['loss'])) if len(out['loss']) else None,
                loss_median=float(np.median(out['loss'])) if len(out['loss']) else None)
    np.savez(os.path.join(OUT_DIR, f'{cell_name}.npz'), **out,
             meta=json.dumps(meta))
    return out, meta


# --------------------------------------------------------------------------- #
#                        CATAPULT ONSET DETECTION                              #
# --------------------------------------------------------------------------- #
def detect_catapults(loss, K=3.0, win=40, refractory=15, recov=1.6, R=80):
    """Onset = step where loss exceeds K * trailing-median(win); refractory suppresses
    double-counting. Returns list of onset indices (recovered spikes only when
    require_recovery in the caller). Plain version returns all onsets."""
    n = len(loss)
    onsets = []
    last = -10 ** 9
    for t in range(win, n):
        base = np.median(loss[t - win:t])
        if base <= 0 or not np.isfinite(base):
            continue
        if loss[t] > K * base and (t - last) > refractory:
            onsets.append(t)
            last = t
    return onsets


def detect_catapults_recovered(loss, K=3.0, win=40, refractory=15, recov=1.6, R=120):
    """Split onsets into (recovered, terminal_cut).

    A recovered catapult = onset where loss falls back below recov*trailing_median
    within R steps. The FIRST onset that does NOT recover marks the terminal runaway;
    we cut the run there (return truncation index) and keep only recovered catapults
    strictly before it. This turns each catapult-rich-then-diverge run into a clean
    bounded run with a set of recovered catapult events."""
    onsets = detect_catapults(loss, K=K, win=win, refractory=refractory)
    n = len(loss)
    recovered = []
    cut = n
    for oc in onsets:
        base = np.median(loss[max(0, oc - win):oc])
        rec = any(loss[j] < recov * base for j in range(oc + 1, min(n, oc + R)))
        if rec:
            recovered.append(oc)
        else:
            cut = oc  # terminal runaway begins here; stop
            break
    recovered = [o for o in recovered if o < cut]
    return recovered, cut


# --------------------------------------------------------------------------- #
#                    ROC / AUC  (per-step imminent-catapult labels)            #
# --------------------------------------------------------------------------- #
def rank_auc(scores, labels):
    """Mann-Whitney rank AUC = P(score_pos > score_neg). Handles ties (0.5)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1  # average rank (1-based)
        i = j
    sum_ranks_pos = ranks[labels].sum()
    n_pos = len(pos); n_neg = len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve(scores, labels, n_thresh=200):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    if labels.sum() == 0 or (~labels).sum() == 0:
        return np.array([0, 1]), np.array([0, 1])
    lo, hi = np.min(scores), np.max(scores)
    ths = np.linspace(hi, lo, n_thresh)
    P = labels.sum(); Nn = (~labels).sum()
    tprs, fprs = [], []
    for th in ths:
        pred = scores >= th
        tp = np.sum(pred & labels)
        fp = np.sum(pred & ~labels)
        tprs.append(tp / P)
        fprs.append(fp / Nn)
    return np.asarray(fprs), np.asarray(tprs)


def build_labels(n, onsets, W):
    """y_t = 1 if a catapult onset falls in (t, t+W]. Steps at/after an onset within
    a short refractory are dropped from the sample (we score PREDICTION, not detection
    of the ongoing spike)."""
    y = np.zeros(n, dtype=bool)
    for oc in onsets:
        y[max(0, oc - W):oc] = True   # window before onset is "positive/imminent"
    mask = np.ones(n, dtype=bool)
    for oc in onsets:
        mask[oc:min(n, oc + W)] = False  # exclude the spike itself + recovery
    return y, mask


def event_precision_recall(scores, theta, onsets, n, W):
    """Event-level: up-crossings of theta vs catapult onsets.
    recall  = frac of catapults with an up-crossing in [t_c-W, t_c-1]
    precision = frac of up-crossings followed by an onset within W steps."""
    scores = np.asarray(scores, dtype=np.float64)
    above = scores > theta
    upcross = [t for t in range(1, n)
               if above[t] and not above[t - 1] and np.isfinite(scores[t])]
    onset_set = np.asarray(onsets, dtype=np.int64)
    # recall
    hit = 0
    for oc in onsets:
        window_cross = any((oc - W) <= tc < oc for tc in upcross)
        if window_cross:
            hit += 1
    recall = hit / len(onsets) if onsets else float('nan')
    # precision
    good = 0
    for tc in upcross:
        if np.any((onset_set > tc) & (onset_set <= tc + W)):
            good += 1
    precision = good / len(upcross) if upcross else float('nan')
    return dict(precision=float(precision), recall=float(recall),
                n_upcross=len(upcross), n_onsets=len(onsets), theta=float(theta),
                n_hit=hit, n_good=good)


SERIES_KEYS = ['lambda_B', 'GBS', 'growth', 'lambda_B_ema', 'GBS_ema']


def ema(x, alpha=0.2):
    y = np.empty_like(x, dtype=np.float64)
    acc = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            v = acc
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def nat_theta_for(key, meta):
    return {'lambda_B': meta['edge_2_over_lr'], 'GBS': 2.0, 'growth': 1.0,
            'lambda_B_ema': meta['edge_2_over_lr'], 'GBS_ema': 2.0}[key]


def prepare_run(out, meta, W=5, K=3.0, win=40, refractory=15, recov=1.6, R=120):
    """One run -> truncated bounded arrays + recovered catapults + per-step labels."""
    loss = out['loss']
    onsets, cut = detect_catapults_recovered(loss, K=K, win=win, refractory=refractory,
                                             recov=recov, R=R)
    n = cut
    series = {
        'lambda_B': out['lambda_B'][:n],
        'GBS': out['GBS'][:n],
        'growth': out['growth'][:n],
        'lambda_B_ema': ema(out['lambda_B'])[:n],
        'GBS_ema': ema(out['GBS'])[:n],
    }
    y, mask = build_labels(n, onsets, W)
    return dict(series=series, y=y, mask=mask, onsets=onsets, cut=cut, n=n,
                loss=loss[:n])


def pool_and_score(runs, meta, W=5):
    """Pool per-step (score, label, mask) across runs -> one AUC per candidate; sum
    event-level hit/upcross counts across runs -> pooled precision/recall."""
    results = {}
    total_onsets = sum(len(r['onsets']) for r in runs)
    for key in SERIES_KEYS:
        pooled_s, pooled_y = [], []
        n_hit = n_up = n_good = n_ons = 0
        for r in runs:
            s = np.asarray(r['series'][key], dtype=np.float64)
            m = r['mask'] & np.isfinite(s)
            pooled_s.append(s[m]); pooled_y.append(r['y'][m])
            pr = event_precision_recall(s, nat_theta_for(key, meta), r['onsets'], r['n'], W)
            n_hit += pr['n_hit']; n_up += pr['n_upcross']; n_good += pr['n_good']
            n_ons += pr['n_onsets']
        ss = np.concatenate(pooled_s) if pooled_s else np.array([])
        yy = np.concatenate(pooled_y) if pooled_y else np.array([], dtype=bool)
        auc = rank_auc(ss, yy) if len(ss) > 5 and yy.sum() > 0 and (~yy).sum() > 0 else float('nan')
        recall = n_hit / n_ons if n_ons else float('nan')
        precision = n_good / n_up if n_up else float('nan')
        results[key] = dict(auc=auc, precision=precision, recall=recall,
                            n_upcross=n_up, n_onsets=n_ons, theta=nat_theta_for(key, meta),
                            n_pos=int(yy.sum()), n_neg=int((~yy).sum()))
    return dict(n_onsets=total_onsets, series_scores=results, W=W)


# --------------------------------------------------------------------------- #
#                                PLOTTING                                       #
# --------------------------------------------------------------------------- #
def make_trace_plot(cell_name, run, meta):
    """Trace of the representative (most-catapults) truncated run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'(matplotlib unavailable: {e})'); return
    n = run['n']; step = np.arange(n)
    loss = run['loss']; onsets = run['onsets']
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax = axes[0]
    ax.semilogy(step, np.clip(loss, 1e-6, None), lw=0.7, color='#333')
    for oc in onsets:
        ax.axvline(oc, color='#e41a1c', alpha=0.4, lw=1.0)
    ax.set_ylabel('batch loss')
    ax.set_title(f"{cell_name}  ({meta['optimizer']} b{meta['batch']} lr={meta['lr']})  "
                 f"repr run: {len(onsets)} recovered catapults  |  2/eta={meta['edge_2_over_lr']:.1f}")
    panels = [('lambda_B', r'$\lambda_B=h_t$', '#377eb8', meta['edge_2_over_lr'], r'$2/\eta$'),
              ('GBS', 'GBS$_t$', '#4daf4a', 2.0, '2'),
              ('growth', (r'$\rho_t$' if meta['is_momentum'] else r'$a_t$'), '#984ea3', 1.0, '1')]
    for ax, (key, lbl, col, thr, thrlbl) in zip(axes[1:], panels):
        s = run['series'][key]
        ax.plot(step, s, lw=0.7, color=col, label=lbl)
        ax.axhline(thr, color=col, ls='--', lw=1.0, label=thrlbl)
        for oc in onsets:
            ax.axvline(oc, color='#e41a1c', alpha=0.3, lw=0.8)
        if key == 'GBS' and np.isfinite(s).sum() > 2:
            ax.set_ylim(np.nanpercentile(s, 1) - 0.5, np.nanpercentile(s, 99) + 0.5)
        ax.set_ylabel(lbl); ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('step')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{cell_name}_trace.png'), dpi=110); plt.close(fig)


def make_roc_plot(cell_name, runs, meta, scored):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'(matplotlib unavailable: {e})'); return
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {'lambda_B': '#377eb8', 'GBS': '#4daf4a', 'growth': '#984ea3'}
    for key in ('lambda_B', 'GBS', 'growth'):
        pooled_s, pooled_y = [], []
        for r in runs:
            s = np.asarray(r['series'][key], dtype=np.float64)
            m = r['mask'] & np.isfinite(s)
            pooled_s.append(s[m]); pooled_y.append(r['y'][m])
        ss = np.concatenate(pooled_s); yy = np.concatenate(pooled_y)
        fpr, tpr = roc_curve(ss, yy)
        auc = scored['series_scores'][key]['auc']
        ax.plot(fpr, tpr, color=colors[key], lw=1.8, label=f'{key}  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], ls=':', color='#999', lw=1.0)
    ax.set_xlabel('FPR (spurious crossings)'); ax.set_ylabel('TPR (imminent-catapult steps caught)')
    ax.set_title(f"{cell_name}: pooled catapult-prediction ROC\n"
                 f"{scored['n_onsets']} catapults, W={scored['W']}")
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{cell_name}_roc.png'), dpi=120); plt.close(fig)


def main():
    wanted = sys.argv[1:] if len(sys.argv) > 1 else list(GRID.keys())
    W = int(os.environ.get('EOSS_W', '5'))
    grand0 = time.time()
    all_summ = {}
    for cell_name in wanted:
        if cell_name not in GRID:
            print(f'!! unknown cell {cell_name}, skipping'); continue
        spec = GRID[cell_name]
        seeds = spec.get('seeds', [0])
        print(f"\n=== CELL {cell_name}  ({spec['optimizer']} b{spec['batch']} "
              f"lr={spec['lr']} steps={spec['steps']} seeds={seeds}) ===", flush=True)
        runs = []
        meta0 = None
        for sd in seeds:
            s2 = dict(spec); s2['batch_seed'] = sd
            out, meta = run_cell(f'{cell_name}_s{sd}', s2, verbose=False)
            meta0 = meta
            run = prepare_run(out, meta, W=W)
            run['seed'] = sd; run['diverged'] = meta['diverged']
            runs.append(run)
            print(f"    seed {sd}: logged={meta['n_logged']} cut={run['cut']} "
                  f"recovered_catapults={len(run['onsets'])} diverged={meta['diverged']} "
                  f"({meta['wall']:.1f}s)", flush=True)
        scored = pool_and_score(runs, meta0, W=W)
        repr_run = max(runs, key=lambda r: len(r['onsets']))
        make_trace_plot(cell_name, repr_run, meta0)
        make_roc_plot(cell_name, runs, meta0, scored)
        ss = scored['series_scores']
        print(f"  POOLED: catapults={scored['n_onsets']} "
              f"(pos-steps={ss['lambda_B']['n_pos']}, neg-steps={ss['lambda_B']['n_neg']})", flush=True)
        for key in SERIES_KEYS:
            r = ss[key]
            print(f"    {key:14s} AUC={r['auc']:.3f}  "
                  f"prec={r['precision']:.3f} rec={r['recall']:.3f} "
                  f"(nup={r['n_upcross']}, theta={r['theta']:.3g})", flush=True)
        all_summ[cell_name] = dict(
            meta=dict(optimizer=meta0['optimizer'], batch=meta0['batch'], lr=meta0['lr'],
                      params=meta0['params'], edge_2_over_lr=meta0['edge_2_over_lr']),
            n_onsets=scored['n_onsets'], W=W,
            per_seed=[dict(seed=r['seed'], cut=r['cut'], n=r['n'],
                           n_catapults=len(r['onsets']), diverged=r['diverged']) for r in runs],
            series_scores=ss)

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_summ, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    write_table(all_summ)
    print(f"\nTotal wall {time.time()-grand0:.1f}s. Outputs in {OUT_DIR}", flush=True)


def write_table(all_summ):
    lines = []
    hdr = (f"{'cell':22s} {'opt':13s} {'b':>4s} {'lr':>7s} {'#cat':>5s} "
           f"{'lamB_AUC':>9s} {'GBS_AUC':>9s} {'grow_AUC':>9s} "
           f"{'lamB_rec':>9s} {'GBS_rec':>8s} {'grow_rec':>9s} "
           f"{'lamB_prc':>9s} {'GBS_prc':>8s} {'grow_prc':>9s}")
    lines.append(hdr); lines.append('-' * len(hdr))
    for cell, d in all_summ.items():
        m = d['meta']; s = d['series_scores']
        def g(k, f):
            v = s[k][f]; return v if v == v else float('nan')
        lines.append(
            f"{cell:22s} {m['optimizer']:13s} {m['batch']:4d} {m['lr']:7.4f} "
            f"{d['n_onsets']:5d} "
            f"{g('lambda_B','auc'):9.3f} {g('GBS','auc'):9.3f} {g('growth','auc'):9.3f} "
            f"{g('lambda_B','recall'):9.3f} {g('GBS','recall'):8.3f} {g('growth','recall'):9.3f} "
            f"{g('lambda_B','precision'):9.3f} {g('GBS','precision'):8.3f} {g('growth','precision'):9.3f}")
    table = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, 'roc_table.txt'), 'w') as f:
        f.write(table + '\n')
    print('\n' + table, flush=True)


if __name__ == '__main__':
    main()
