"""
Kesten / moment-Lyapunov tail test -- the quantitative metastable-theory law, done right.

Along the unstable coordinate momentum is a random linear recursion x_{t+1}=a_t x_t + b_t; the
stationary tail is P(|x|>u) ~ u^{-alpha}. Because the a_t are TEMPORALLY CORRELATED through the
buffer (the R story; iid was arbiter v2's failure), the correct condition is the moment
Lyapunov Lambda(alpha)=0, Lambda(alpha)=lim (1/T) log E[||prod J||^alpha], estimated on BLOCK
products of length m >> correlation time 1/(1-beta): E[G_block^alpha]=1.

Four refinements (per reviewer):
 1. beta=0 anchor: Kesten is the LINEAR theory; at marginality the cubic self-stabilization
    TRUNCATES the tail. So beta=0 => heaviest tail up to a nonlinear cutoff -- score the
    quantitative Lambda(alpha)=0 vs Hill match on the DAMPED cells (beta>=0.6) only; at beta=0
    the qualitative "heaviest+truncated" is the prediction (a fitted alpha *matching* there
    would be suspicious, not confirmatory).
 2. block-moment (above), not naive per-step E[a^alpha].
 3. tail hygiene: Hill over multiple tail fractions (stability), on a ROTATION-ROBUST
    coordinate (top-K subspace amplitude, not fixed-u); + input-noise tail control (is the
    OUTPUT tail heavier than the input h_t tail? -- Kesten's signature is output>input).
 4. alpha=2 is exactly the mean-square boundary (E[a^2]=1 <=> Lambda(2)=0). Report where
    alpha crosses 2 vs the R~1 crossover (beta~0.3) -- ties the moment hierarchy to tail physics.
"""
import os, sys, json
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

import experiments.perturb_relax as PR
from utils.measure import compute_eigenvalues, EigenvectorCache, param_vector

T.set_num_threads(4)
OUT_DIR = os.path.join(_REPO, 'results', 'kesten_test')
os.makedirs(OUT_DIR, exist_ok=True)


def collect(state, beta, lr, batch, steps=3000, K=5, ema_hl=100):
    """Along the plateau trajectory: per-step companion growth a_t (driven by realized
    h_t=lambda(H_B)); the h_t series (input tail); and top-K subspace excursion amplitude
    (rotation-robust output coordinate)."""
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']
    # top-K eigvecs at window start for the rotation-robust coordinate
    Xs, Ys = X[:2048], Y[:2048]; pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, V = compute_eigenvalues(lo, net, k=K, max_iterations=80, reltol=0.01,
                               eigenvector_cache=EigenvectorCache(K), return_eigenvectors=True,
                               use_power_iteration=False)
    V = V.detach()                                  # [n, K]
    cache = EigenvectorCache(1)
    v = np.array([1.0, 0.0]); a_list, h_list, amp_list = [], [], []
    p_ema = (V.t() @ param_vector(net)).numpy(); alpha_ema = 1 - 0.5 ** (1.0 / ema_hl)
    g = T.Generator().manual_seed(31)
    for t in range(steps):
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        h = float(compute_eigenvalues(lo, net, k=1, max_iterations=40, reltol=0.01,
                                      eigenvector_cache=cache, return_eigenvectors=False,
                                      use_power_iteration=False))
        h_list.append(h)
        M = np.array([[1 + beta - lr * h, -beta], [1.0, 0.0]])
        v2 = M @ v; nrm = np.linalg.norm(v2); a_list.append(nrm / (np.linalg.norm(v) + 1e-30)); v = v2 / (nrm + 1e-30)
        p = (V.t() @ param_vector(net)).numpy(); p_ema = (1 - alpha_ema) * p_ema + alpha_ema * p
        amp_list.append(float(np.linalg.norm(p - p_ema)))
        opt.zero_grad(); lo.backward(); opt.step()
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            break
    return np.array(a_list), np.array(h_list), np.array(amp_list)


def block_moment_alpha(a, beta):
    """Solve E[G_block^alpha]=1 on block products (block length m >> 1/(1-beta))."""
    a = a[np.isfinite(a) & (a > 0)]
    m = int(min(120, max(20, round(4.0 / (1.0 - beta + 1e-9)))))
    nb = len(a) // m
    if nb < 8:
        return float('nan'), m, 0
    G = np.array([np.prod(a[j * m:(j + 1) * m]) for j in range(nb)])
    G = G[np.isfinite(G) & (G > 0)]
    if len(G) < 8 or np.all(G <= 1):
        return float('inf'), m, len(G)
    def mom(al):
        return np.mean(G ** al)
    lo, hi = 1e-3, 0.05
    for _ in range(80):
        if mom(hi) >= 1.0 or hi > 300:
            break
        hi *= 1.5
    if mom(hi) < 1.0:
        return float('inf'), m, len(G)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if mom(mid) < 1.0 else (lo, mid)
    return float(0.5 * (lo + hi)), m, len(G)


def hill(x, fracs=(0.05, 0.10, 0.20)):
    ax = np.sort(np.abs(x[np.isfinite(x)]))[::-1]
    out = {}
    for f in fracs:
        k = max(10, int(len(ax) * f))
        if len(ax) < k + 5 or ax[k] <= 0:
            out[f] = float('nan'); continue
        out[f] = float(1.0 / (np.mean(np.log(ax[:k] / ax[k])) + 1e-30))
    return out


def main():
    cells = [
        (0.0,  0.006, 12000),
        (0.3,  0.005, 12000),
        (0.6,  0.004, 12000),
        (0.9,  0.002, 14000),
        (0.99, 0.0001, 16000),
    ]
    batch = 8
    results = []
    for beta, lr, steps in cells:
        print(f"\n=== beta={beta} lr={lr} steps={steps} ===", flush=True)
        st = PR.train_plateau("SGD-Momentum", {"beta": beta}, batch, lr, steps)
        if st is None:
            print(f"  beta={beta}: diverged"); results.append(dict(beta=beta, diverged=True)); continue
        a, h, amp = collect(st, beta, lr, batch)
        alpha, m, nb = block_moment_alpha(a, beta)
        hill_out = hill(amp)          # output tail (top-K excursion amplitude)
        hill_in = hill(h - np.median(h))   # input tail (curvature fluctuation)
        Elog = float(np.mean(np.log(a[a > 0]))); Ea2 = float(np.mean(a ** 2)); pa1 = float(np.mean(a > 1))
        rec = dict(beta=beta, lr=lr, diverged=False, alpha_block=alpha, block_m=m, n_blocks=nb,
                   E_log_a=Elog, E_a2=Ea2, P_a_gt1=pa1,
                   hill_out=hill_out, hill_in=hill_in)
        results.append(rec)
        print(f"  E[log a]={Elog:+.4f} E[a^2]={Ea2:.3f} P(a>1)={pa1:.3f}  block m={m} nb={nb}  "
              f"alpha_block={alpha:.2f}  hill_out(10%)={hill_out.get(0.1,float('nan')):.2f} "
              f"hill_in(10%)={hill_in.get(0.1,float('nan')):.2f}", flush=True)
        with open(os.path.join(OUT_DIR, 'kesten_test.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n===== KESTEN / MOMENT-LYAPUNOV VERDICT =====")
    print(f"{'beta':>5s} {'E[log a]':>9s} {'alpha_block':>11s} {'hill_out(5/10/20%)':>22s} {'hill_in(10%)':>12s}")
    for r in results:
        if r.get('diverged'): print(f"{r['beta']:5.2f}  diverged"); continue
        ho = r['hill_out']; hstr = '/'.join(f"{ho.get(f,float('nan')):.1f}" for f in (0.05,0.10,0.20))
        print(f"{r['beta']:5.2f} {r['E_log_a']:+9.4f} {r['alpha_block']:11.2f} {hstr:>22s} "
              f"{r['hill_in'].get(0.1,float('nan')):12.2f}")
    print("\n Score match (alpha_block ~ hill_out) on DAMPED cells beta>=0.6 only (beta=0 is")
    print(" truncated by nonlinearity, expect heaviest+cutoff). alpha=2 <=> mean-square boundary;")
    print(" note where alpha crosses 2 vs R~1 (beta~0.3). Kesten signature: hill_out < hill_in")
    print(" (output tail HEAVIER than input curvature tail).")


if __name__ == '__main__':
    main()
