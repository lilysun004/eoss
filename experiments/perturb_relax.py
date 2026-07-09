"""
Perturb-and-relax: the CAUSAL, yardstick-free test of marginal vs metastable.

The R table currently decides "edge reached" via GBS~2 and eta*lam/edge. But eta*lam/edge
uses the DETERMINISTIC edge 2(1+beta)/eta -- the formula this whole investigation
discredited, and whose validity degrades with beta (the decoherence we're studying). That's
circular. This test replaces it with a direct dynamical measurement, no beta, no edge formula:

At the plateau, kick the iterate by amplitude a along the unstable direction and run the REAL
optimizer forward with a fixed (seeded) batch sequence; a reference copy runs the same batches
un-kicked. Track the induced deviation delta_t = theta_kicked(t) - theta_ref(t):
  - MARGINAL (held at boundary by feedback): delta persists (relax rate gamma ~ 0); larger
    kicks trigger self-stabilization (sharpness dips then recovers -- thermostat visible).
  - METASTABLE (parked in a basin by damping): delta DECAYS at a measurable rate gamma < 0
    (the AR damping made causal), until a large-enough kick ESCAPES (delta blows up = catapult).
This measures restoring force (gamma) and distance-to-boundary (escape amplitude) directly.

Run along the beta-sweep at fixed b8: beta=0 (expect marginal, gamma~0), beta=0.9 (expect
damped + finite escape threshold), and beta=0.3 (near the R~1 crossover -- the response
character should visibly change here). If gamma flips sign / escape-threshold appears at the
same R where GBS declines, the regime transition is confirmed by an independent causal
instrument -- and the R claim no longer rests on the deterministic edge formula.

Kick direction u: top eigvec of the (subset) Hessian at theta* AND, as a parameter-free
alternative, the top principal component of recent realized steps (step-PCA). We report both.
lean mlp_s/2048.
"""
import os, sys, json, copy
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

import experiments.long_train_grid as L
from utils.optimizer import create_optimizer
from utils.measure import (compute_eigenvalues, EigenvectorCache, param_vector, flatt,
                           create_hessian_vector_product)
from utils.curvature_segment import set_params_inplace

T.set_num_threads(4)
OUT_DIR = os.path.join(_REPO, 'results', 'perturb_relax')
os.makedirs(OUT_DIR, exist_ok=True)


def set_optimizer_state(opt, state_snapshot):
    opt.inner.load_state_dict(copy.deepcopy(state_snapshot))


def train_plateau(optn, params, batch, lr, steps, step_window=200):
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer(optn, net, lr, params)
    recent_steps = []
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return None
        params_l = [p for p in net.parameters() if p.requires_grad]
        opt.zero_grad(); lo.backward()
        if step >= steps - step_window:
            g = flatt([p.grad for p in params_l]).detach()
            recent_steps.append(opt.compute_step_direction(g, params_l).detach().clone())
        opt.step()
    theta_star = param_vector(net).detach().clone()
    net_sd = copy.deepcopy(net.state_dict())
    opt_sd = copy.deepcopy(opt.inner.state_dict())
    # subset-H top eigvec at theta*
    Xs, Ys = X[:2048], Y[:2048]
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, u_hess = compute_eigenvalues(lo, net, k=1, max_iterations=80, reltol=0.005,
                                    eigenvector_cache=EigenvectorCache(1),
                                    return_eigenvectors=True, use_power_iteration=False)
    u_hess = u_hess.detach(); u_hess = u_hess / u_hess.norm()
    # step-PCA direction (parameter-free)
    S = T.stack(recent_steps, 0)           # [W, n]
    S = S - S.mean(0, keepdim=True)
    # top right singular vector of S
    try:
        _, _, Vh = T.linalg.svd(S, full_matrices=False)
        u_step = Vh[0]; u_step = u_step / u_step.norm()
    except Exception:
        u_step = u_hess.clone()
    return dict(net=net, opt=opt, loss_fn=loss_fn, XY=(X, Y), theta_star=theta_star,
                net_sd=net_sd, opt_sd=opt_sd, u_hess=u_hess, u_step=u_step, batch=batch)


def relax(state, u, amp, K=150, seed=0):
    """Restore plateau, kick by amp*u, run K steps with a fixed batch seed; return the
    u-projected deviation series x_t = u^T (theta_t - theta*) and loss series."""
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']; theta_star = state['theta_star']; batch = state['batch']
    net.load_state_dict(copy.deepcopy(state['net_sd']))
    set_optimizer_state(opt, state['opt_sd'])
    with T.no_grad():
        set_params_inplace(net, theta_star + amp * u)
    g = T.Generator().manual_seed(seed)
    xs, losses = [], []
    for t in range(K):
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        with T.no_grad():
            dev = param_vector(net) - theta_star
            xs.append(float(T.dot(dev, u)))
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item(); losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            xs += [float('nan')] * (K - len(xs)); losses += [float('nan')] * (K - len(losses)); break
        opt.zero_grad(); lo.backward(); opt.step()
    return np.array(xs), np.array(losses)


def analyze(state, u, tag_dir):
    # natural amplitude scale = std of the un-kicked u-projection
    x0, _ = relax(state, u, 0.0, K=150, seed=1)
    scale = float(np.nanstd(x0)) + 1e-12
    amps = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0]) * scale
    out = []
    for a in amps:
        # induced deviation vs the a=0 reference under the SAME batch seed
        xa, la = relax(state, u, a, K=150, seed=7)
        x0b, _ = relax(state, u, 0.0, K=150, seed=7)
        dx = xa - x0b
        # early-window relaxation rate gamma from log|dx| (steps 2..40), if it's decaying/persisting
        w = dx[2:40]; w = w[np.isfinite(w)]
        if len(w) > 5 and np.all(np.abs(w) > 0):
            gamma = float(np.polyfit(np.arange(len(w)), np.log(np.abs(w) + 1e-30), 1)[0])
        else:
            gamma = float('nan')
        escaped = bool(np.any(~np.isfinite(la)) or np.nanmax(np.abs(dx)) > 20 * (abs(a) + scale))
        out.append(dict(amp=float(a), amp_over_scale=float(a / scale), gamma=gamma,
                        max_absdx=float(np.nanmax(np.abs(dx))), escaped=escaped))
    return dict(scale=scale, amps=out)


def main():
    cells = [
        ("beta0.0", "SGD-Momentum", {"beta": 0.0}, 8, 0.006, 12000),
        ("beta0.3", "SGD-Momentum", {"beta": 0.3}, 8, 0.005, 12000),
        ("beta0.9", "SGD-Momentum", {"beta": 0.9}, 8, 0.002, 14000),
    ]
    results = []
    for tag, optn, params, batch, lr, steps in cells:
        print(f"\n=== {tag}: {optn} b8 lr={lr} steps={steps} ===", flush=True)
        st = train_plateau(optn, params, batch, lr, steps)
        if st is None:
            print(f"  {tag}: diverged in training"); results.append(dict(tag=tag, diverged=True)); continue
        for uname, u in [("u_hess", st['u_hess']), ("u_step", st['u_step'])]:
            res = analyze(st, u, tag)
            # linear-regime gamma = gamma at the smallest nonzero amp
            lin = next((a for a in res['amps'] if a['amp_over_scale'] > 0.5), res['amps'][-1])
            esc = next((a['amp_over_scale'] for a in res['amps'] if a['escaped']), None)
            rec = dict(tag=tag, beta=params['beta'], kick_dir=uname, scale=res['scale'],
                       gamma_linear=lin['gamma'], escape_amp_over_scale=esc, amps=res['amps'])
            results.append(rec)
            print(f"  [{uname}] gamma_relax(linear)={lin['gamma']:+.4f}/step  "
                  f"escape_threshold={esc if esc else '>16'}x natural  scale={res['scale']:.3g}", flush=True)
            with open(os.path.join(OUT_DIR, 'perturb_relax.json'), 'w') as f:
                json.dump(results, f, indent=2)

    print("\n===== CAUSAL VERDICT (no edge formula, no beta) =====")
    print(f"{'beta':>5s} {'dir':>7s} {'gamma_relax':>11s} {'escape(x nat)':>13s}")
    for r in results:
        if r.get('diverged'): print(f"{r['tag']}: diverged"); continue
        esc = r['escape_amp_over_scale']
        print(f"{r['beta']:5.2f} {r['kick_dir']:>7s} {r['gamma_linear']:+11.4f} {str(esc) if esc else '>16':>13s}")
    print("\n gamma~0 & no finite escape => MARGINAL (persists, feedback-held).")
    print(" gamma<0 & finite escape threshold => METASTABLE (damped basin + catapult escape).")
    print(" If the response character flips near the R~1 crossover (beta~0.3), the regime")
    print(" transition is confirmed causally, independent of GBS and the edge formula.")


if __name__ == '__main__':
    main()
