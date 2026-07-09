"""
Experiment: AR-pole marginality test (path-statistics version of the stochastic edge).

Companion to frozen_cocycle.py. Same invariant, second independent route: instead of
autodiffing the closed-loop Jacobian, we SYSTEM-IDENTIFY the top loop pole from the
recorded iterate path, treating the optimizer as an unknown LTI filter that is NEVER
differentiated.

Near a plateau, the closed loop from gradient (batch) noise to the unstable coordinate
    x_t = u^T (theta_t - EMA(theta_t))     (u = top Hessian eigenvector, fixed)
is a linear time-invariant filter driven by persistently-exciting batch noise. So its top
pole is recoverable from x_t alone by fitting an AR(k) model over stationary
(inter-catapult) windows and reading off the top companion-matrix eigenvalue.

Claim under test:
    top AR-pole modulus rho ~ 1 universally         (dream metric, pure path statistics)
vs the dichotomy alternative:
    rho ~ 1, phase ~ pi (period-2 sign-flip) for marginal SGD, but rho well inside the
    unit circle (rho ~ beta-ish, damped) for genuinely sub-edge metastable momentum,
    touching 1 only at catapults.

This is NON-tautological: a bounded stationary noise-driven AR process can have rho < 1
(damped). rho = 1 is a real, falsifiable marginality statement, not forced by boundedness.

CAVEAT (closed-loop identification bias): in feedback the regressor (past x) is correlated
with the driving noise, biasing naive AR. First pass reports the naive AR pole (still
decisive for the rho~1 vs rho~beta discrimination). IV mitigation is left as a note.

lean setup (mlp_s, num_data=2048), CPU. lr/steps from results/calib2/FINAL_GRID.json.
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
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import experiments.long_train_grid as L
from utils.optimizer import create_optimizer
from utils.measure import compute_eigenvalues, param_vector

OUT_DIR = os.path.join(_REPO, 'results', 'ar_pole')
os.makedirs(OUT_DIR, exist_ok=True)

EMA_HALFLIFE = 100      # steps; removes slow manifold drift, keeps fast oscillation
CAT_K = 3.0             # catapult: batch-loss spike > K * trailing median
KMIN, KMAX = 2, 8       # AR orders to fit


# ------------------------------ AR machinery ------------------------------

def ema_detrend(x, halflife):
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    ema = np.empty_like(x)
    m = x[0]
    for t in range(len(x)):
        m = alpha * x[t] + (1 - alpha) * m
        ema[t] = m
    return x - ema, ema


def fit_ar(x, k):
    """Least-squares AR(k): x_t = sum_i a_i x_{t-i} + e_t. Returns (coeffs, poles).
    Poles = eigenvalues of the companion matrix = roots of z^k - a1 z^{k-1} - ... - ak."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < k + 20:
        return None, None
    Y = x[k:N]
    cols = [x[k - 1 - i: N - 1 - i] for i in range(k)]   # lag i+1
    Xd = np.column_stack(cols)
    a, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    C = np.zeros((k, k))
    C[0, :] = a
    if k > 1:
        C[1:, :-1] = np.eye(k - 1)
    poles = np.linalg.eigvals(C)
    return a, poles


def top_pole(poles):
    """Top pole = max modulus. Returns (rho, phase_radians)."""
    idx = int(np.argmax(np.abs(poles)))
    p = poles[idx]
    return float(np.abs(p)), float(np.angle(p))


def clean_windows(losses, n, min_len=300):
    """Inter-catapult windows over the logged segment. Returns list of (start, end)
    index pairs (into the logged arrays), each a stationary run between catapults."""
    onsets = L.detect_catapults(np.asarray(losses), K=CAT_K, win=50, refractory=20)
    # segment boundaries: [0, onset0+skip, onset1+skip, ..., n]
    skip = 30   # skip the nonlinear transient right after a catapult
    bounds = [0]
    for o in onsets:
        bounds.append(min(o + skip, n))
    bounds.append(n)
    wins = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s >= min_len:
            wins.append((s, e))
    return wins, onsets


def analyze_series(x_detr, losses):
    """Fit AR(k) k=KMIN..KMAX on the longest clean inter-catapult window; also aggregate
    top-pole across all clean windows. Returns a dict of pole estimates."""
    n = len(x_detr)
    wins, onsets = clean_windows(losses, n)
    if not wins:
        wins = [(0, n)]   # fall back to whole series
    # longest window is the primary fit target
    wins_sorted = sorted(wins, key=lambda w: w[1] - w[0], reverse=True)
    s, e = wins_sorted[0]
    xw = x_detr[s:e]
    xw = xw - xw.mean()

    per_k = {}
    for k in range(KMIN, KMAX + 1):
        a, poles = fit_ar(xw, k)
        if poles is None:
            per_k[k] = None
            continue
        rho, phase = top_pole(poles)
        per_k[k] = dict(rho=rho, phase=phase,
                        poles=[[float(p.real), float(p.imag)] for p in poles])
    # rho stability across k (use k>=3)
    rhos = [per_k[k]['rho'] for k in range(3, KMAX + 1) if per_k.get(k)]
    phases = [per_k[k]['phase'] for k in range(3, KMAX + 1) if per_k.get(k)]
    rho_med = float(np.median(rhos)) if rhos else float('nan')
    rho_spread = float(np.std(rhos)) if rhos else float('nan')
    phase_med = float(np.median(phases)) if phases else float('nan')

    # aggregate top-pole (rho at a representative order, k=4) across ALL clean windows
    agg = []
    for (ws, we) in wins_sorted:
        xx = x_detr[ws:we]; xx = xx - xx.mean()
        _, poles = fit_ar(xx, 4)
        if poles is not None:
            agg.append(top_pole(poles))
    agg_rho = float(np.mean([r for r, _ in agg])) if agg else float('nan')

    reliable = bool(rhos) and rho_spread < 0.08
    return dict(primary_window=[int(s), int(e)], window_len=int(e - s),
                n_windows=len(wins), n_catapults=len(onsets),
                per_k={str(k): per_k[k] for k in per_k},
                rho_med_k3plus=rho_med, rho_spread_k3plus=rho_spread,
                phase_med_k3plus=phase_med, agg_rho_k4=agg_rho,
                reliable=reliable, onsets=[int(o) for o in onsets])


# ------------------------------ run a cell ------------------------------

def run_cell(tag, optn, params, batch, lr, train_steps, log_steps):
    X, Y = L.get_data()
    net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    t0 = time.time()

    # 1) train to plateau
    for step in range(train_steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item()
        if not np.isfinite(lv) or lv > 1e6:
            print(f"  [{tag}] DIVERGED during train @ {step}", flush=True)
            return dict(tag=tag, diverged=True, step=step)
        opt.zero_grad(); lo.backward(); opt.step()

    # 2) fix u = top eigenvector of the full-data Hessian at the plateau
    Xf, Yf = X, Y
    pf = net(Xf).squeeze(-1); lf = loss_fn(pf, Yf)
    lam, u = compute_eigenvalues(lf, net, k=1, max_iterations=100, reltol=1e-2,
                                 return_eigenvectors=True, use_power_iteration=False)
    u = u.detach().reshape(-1)
    u = u / (u.norm() + 1e-30)
    lam = float(lam)

    # 3) log x_t = u^T theta_t for log_steps steps of REAL training (cheap dot product)
    xs, losses = [], []
    for step in range(log_steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item(); losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            print(f"  [{tag}] DIVERGED during log @ {step}", flush=True); break
        opt.zero_grad(); lo.backward(); opt.step()
        theta = param_vector(net)                       # detached clone
        xs.append(float(T.dot(u, theta)))

    xs = np.asarray(xs); losses = np.asarray(losses)
    # 4) detrend
    x_detr, ema = ema_detrend(xs, EMA_HALFLIFE)
    # 5) analyze
    an = analyze_series(x_detr, losses)

    np.savez(os.path.join(OUT_DIR, f"{tag}.npz"),
             x_raw=xs, x_detr=x_detr, ema=ema, losses=losses,
             lam=lam, lr=lr, eta_lam=lr * lam)

    dt = time.time() - t0
    print(f"  [{tag}] lam={lam:.2f} eta*lam={lr*lam:.3f}  "
          f"rho(k3+)={an['rho_med_k3plus']:.3f}+/-{an['rho_spread_k3plus']:.3f}  "
          f"phase/pi={an['phase_med_k3plus']/np.pi:+.2f}  "
          f"cats={an['n_catapults']} win={an['window_len']}  "
          f"reliable={an['reliable']}  ({dt:.0f}s)", flush=True)

    return dict(tag=tag, optimizer=optn, params=params, batch=batch, lr=lr,
                train_steps=train_steps, log_steps=log_steps,
                lam=lam, eta_lam=lr * lam, diverged=False, **an)


# ------------------------------ plotting ------------------------------

def plot_all(results):
    rs = [r for r in results if not r.get('diverged')]
    ncell = len(rs)
    fig, axes = plt.subplots(ncell, 2, figsize=(12, 2.6 * ncell))
    if ncell == 1:
        axes = axes.reshape(1, 2)
    for i, r in enumerate(rs):
        tag = r['tag']
        d = np.load(os.path.join(OUT_DIR, f"{tag}.npz"))
        x_detr = d['x_detr']; losses = d['losses']
        s, e = r['primary_window']
        # left: detrended x_t trace, primary window highlighted, catapults marked
        ax = axes[i, 0]
        ax.plot(x_detr, lw=0.5, color='0.4')
        ax.plot(np.arange(s, e), x_detr[s:e], lw=0.6, color='C0')
        for o in r['onsets']:
            ax.axvline(o, color='r', ls=':', lw=0.6)
        ax.set_title(f"{tag}: x_t=u.theta detrended  (eta*lam={r['eta_lam']:.2f})", fontsize=9)
        ax.set_xlabel('log step', fontsize=8)
        # right: AR poles (k=4) in the complex plane with unit circle
        ax = axes[i, 1]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), 'k-', lw=0.8)
        pk = r['per_k'].get('4') or next((v for v in r['per_k'].values() if v), None)
        if pk:
            P = np.array(pk['poles'])
            ax.scatter(P[:, 0], P[:, 1], c='C3', s=30, zorder=3)
            ax.scatter([pk['poles'][int(np.argmax([np.hypot(*p) for p in pk['poles']]))][0]],
                       [pk['poles'][int(np.argmax([np.hypot(*p) for p in pk['poles']]))][1]],
                       edgecolors='C0', facecolors='none', s=120, zorder=4)
        ax.axhline(0, color='0.7', lw=0.5); ax.axvline(0, color='0.7', lw=0.5)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal')
        ax.set_title(f"AR(4) poles  rho={r['rho_med_k3plus']:.3f} "
                     f"phase/pi={r['phase_med_k3plus']/np.pi:+.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ar_pole_marginality.png'), dpi=90)
    plt.close(fig)


# ------------------------------ main ------------------------------

def main():
    # (tag, optimizer, params, batch, lr, train_steps, log_steps)
    # b8 cells need ~20k steps to reach stationarity (Exp1); b2048 anchors ~6k.
    cells = [
        ("SGD_b8",       "SGD",          {},                            8,    0.01,  20000, 6000),
        ("SGDM09_b8",    "SGD-Momentum", {"beta": 0.9},                 8,    0.002, 20000, 6000),
        ("Adam_b8",      "Adam",         {"beta1": 0.9, "beta2": 0.99}, 8,    0.001, 20000, 6000),
        ("Muon_b8",      "Muon",         {"momentum": 0.9},             8,    0.003, 20000, 6000),
        ("SGD_b2048",    "SGD",          {},                            2048, 0.02,  6000,  4000),
        ("SGDM09_b2048", "SGD-Momentum", {"beta": 0.9},                 2048, 0.006, 6000,  4000),
    ]
    results = []
    for tag, optn, params, batch, lr, ts, ls in cells:
        print(f"\n=== {tag}: {optn} b={batch} lr={lr} train={ts} log={ls} ===", flush=True)
        r = run_cell(tag, optn, params, batch, lr, ts, ls)
        results.append(r)
        with open(os.path.join(OUT_DIR, 'ar_pole_marginality.json'), 'w') as f:
            json.dump(results, f, indent=2)
    plot_all(results)

    print("\n===== VERDICT =====")
    hdr = f"{'cell':14s} {'eta*lam':>8s} {'rho(k3+)':>16s} {'phase/pi':>9s} {'agg_rho':>8s} {'cats':>5s} {'reliable':>8s}"
    print(hdr)
    for r in results:
        if r.get('diverged'):
            print(f"{r['tag']:14s}  diverged"); continue
        rho = f"{r['rho_med_k3plus']:.3f}+/-{r['rho_spread_k3plus']:.3f}"
        print(f"{r['tag']:14s} {r['eta_lam']:8.2f} {rho:>16s} "
              f"{r['phase_med_k3plus']/np.pi:+9.2f} {r['agg_rho_k4']:8.3f} "
              f"{r['n_catapults']:5d} {str(r['reliable']):>8s}")
    print("\n rho~1 for ALL => dream metric: marginal oscillation, pure path statistics.")
    print(" rho~1 phase~pi for SGD vs rho<1 damped for small-batch momentum => dichotomy,")
    print(" confirmed independently of the autodiff cocycle.")


if __name__ == '__main__':
    main()
