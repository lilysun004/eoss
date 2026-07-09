"""
Metastable tail test with the CORRELATION-vs-TRUNCATION discriminator.

The light realized excursion tail admits two explanations and the earlier smoke didn't
separate them:
  (1) NONLINEAR TRUNCATION: linear theory predicts a heavy tail (alpha<2 from E[a^2]>1),
      the cubic self-stabilization caps excursions -> light tail. Needs a nonlinear model.
  (2) TEMPORAL CORRELATION: the a_t are correlated through the buffer/curvature-feedback;
      negatively-correlated growth factors make the true (correlated) tail LIGHT even though
      the iid/per-step moment E[a^2]>1. Pure LINEAR correlated random-matrix theory explains
      it -- a much cleaner claim, and it's the R-filter story again (the buffer averages ~10
      heavy-tailed input draws, and averaging lightens tails).

DISCRIMINATOR (shuffle test, no new training, reuses the collected a_t): simulate the LINEAR
recursion x_{t+1}=a_t x_t + noise three ways and Hill-fit the tail exponent:
  - alpha_iid   : a_t sampled IID from the empirical marginal (pure Kesten, no correlation, no
                  nonlinearity).
  - alpha_ord   : a_t in their REAL temporal order (linear + correlation, NO nonlinearity).
  - alpha_real  : the actual realized excursion amplitude (everything, incl. nonlinearity).
Readout:
  alpha_iid << alpha_ord ~ alpha_real  => CORRELATION lightens the tail and linear-correlated
      dynamics MATCH reality -> nonlinearity NOT needed; linear random-matrix theory survives.
  alpha_iid ~ alpha_ord (both heavy) << alpha_real => both linear sims heavier than reality ->
      NONLINEAR TRUNCATION is doing the work.
  (disagree / ambiguous => report both-open.)

Also: Hill-based ordering across beta (NOT kurtosis -- kurtosis at beta=0.99 is inflated by
slow drift; the surviving ordering claim rests on Hill). alpha=2 = mean-square boundary; note
where alpha crosses 2 vs R~1. Raw a_t/h_t/amp series saved to npz for the truncation-scale
follow-up (excursion cutoff vs Damian x* ~ sqrt(2 alpha_sharp/beta_sharp)).
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


def collect(state, beta, lr, batch, steps=4000, K=5, ema_hl=100):
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']
    Xs, Ys = X[:2048], Y[:2048]; pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, V = compute_eigenvalues(lo, net, k=K, max_iterations=80, reltol=0.01,
                               eigenvector_cache=EigenvectorCache(K), return_eigenvectors=True,
                               use_power_iteration=False)
    V = V.detach()
    cache = EigenvectorCache(1)
    v = np.array([1.0, 0.0]); a_list, h_list, amp_list = [], [], []
    p_ema = (V.t() @ param_vector(net)).numpy(); ae = 1 - 0.5 ** (1.0 / ema_hl)
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
        p = (V.t() @ param_vector(net)).numpy(); p_ema = (1 - ae) * p_ema + ae * p
        amp_list.append(float(np.linalg.norm(p - p_ema)))
        opt.zero_grad(); lo.backward(); opt.step()
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            break
    return np.array(a_list), np.array(h_list), np.array(amp_list)


def hill(x, fracs=(0.05, 0.10, 0.20)):
    ax = np.sort(np.abs(x[np.isfinite(x)]))[::-1]
    out = {}
    for f in fracs:
        k = max(20, int(len(ax) * f))
        if len(ax) < k + 5 or ax[k] <= 0:
            out[f] = float('nan'); continue
        out[f] = float(1.0 / (np.mean(np.log(ax[:k] / ax[k])) + 1e-30))
    return out


def sim_recursion(a_provider, n, reps=6):
    """x_{t+1}=a_t x_t + xi_t; pool |x| across reps (each with fresh noise & fresh a draw)."""
    pooled = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        a = a_provider(n, rng); x = 0.0
        xs = np.empty(n)
        for t in range(n):
            x = a[t] * x + rng.standard_normal()
            xs[t] = x
        pooled.append(np.abs(xs[n // 5:]))   # drop burn-in
    return np.concatenate(pooled)


def discriminate(a):
    a = a[np.isfinite(a) & (a > 0)]
    n = 15000
    iid = sim_recursion(lambda nn, rng: rng.choice(a, nn), n)
    ordered = sim_recursion(lambda nn, rng: np.resize(a, nn), n)   # real temporal order (tiled)
    return dict(alpha_iid=hill(iid), alpha_ordered=hill(ordered))


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
        np.savez(os.path.join(OUT_DIR, f'series_beta{beta}.npz'), a=a, h=h, amp=amp)
        disc = discriminate(a)
        hill_real = hill(amp); hill_in = hill(h - np.median(h))
        Elog = float(np.mean(np.log(a[a > 0]))); Ea2 = float(np.mean(a ** 2)); pa1 = float(np.mean(a > 1))
        rec = dict(beta=beta, lr=lr, diverged=False, E_log_a=Elog, E_a2=Ea2, P_a_gt1=pa1,
                   alpha_iid=disc['alpha_iid'], alpha_ordered=disc['alpha_ordered'],
                   alpha_real=hill_real, alpha_in=hill_in)
        results.append(rec)
        f = 0.10
        print(f"  E[log a]={Elog:+.4f} E[a^2]={Ea2:.3f}  Hill@10%: iid={disc['alpha_iid'][f]:.2f} "
              f"ordered={disc['alpha_ordered'][f]:.2f} real={hill_real[f]:.2f} input={hill_in[f]:.2f}", flush=True)
        with open(os.path.join(OUT_DIR, 'kesten_test.json'), 'w') as jf:
            json.dump(results, jf, indent=2)

    print("\n===== DISCRIMINATOR VERDICT (Hill @10%) =====")
    print(f"{'beta':>5s} {'E[log a]':>9s} {'E[a^2]':>7s} {'a_iid':>6s} {'a_ord':>6s} {'a_real':>6s} {'a_input':>7s}")
    for r in results:
        if r.get('diverged'): print(f"{r['beta']:5.2f}  diverged"); continue
        f = 0.10
        print(f"{r['beta']:5.2f} {r['E_log_a']:+9.4f} {r['E_a2']:7.3f} {r['alpha_iid'][f]:6.2f} "
              f"{r['alpha_ordered'][f]:6.2f} {r['alpha_real'][f]:6.2f} {r['alpha_in'][f]:7.2f}")
    print("\n a_iid << a_ord ~ a_real  => CORRELATION lightens tail, linear theory survives.")
    print(" a_iid ~ a_ord (heavy) << a_real => NONLINEAR TRUNCATION needed.")
    print(" Ordering claim on Hill a_real across beta (not kurtosis). alpha=2 = mean-square boundary.")


if __name__ == '__main__':
    main()
