"""
THE ADAM ADJUDICATOR (branch lily) — resolve the marginal-vs-metastable contradiction.

The contradiction:
  * Control parameter R = (buffer memory 1/(1-beta)) / (u_B rotation timescale tau_rot).
    Small-batch SGD-Momentum is METASTABLE (sub-edge) because u_B rotates fast (tau_rot~1)
    while its buffer memory is 10 -> R~9 -> buffer can't track the rotating unstable dir
    (cos(buffer,u_B)=0.066) -> never couples to its edge.
  * Adam b8 ALSO has a beta1=0.9 momentum buffer, so the SAME R-mechanism predicts Adam
    metastable. YET a fragile single-number result said lean Adam b8 sits at its
    PRECONDITIONED edge (kappa_precond = eta*lambda(P^-1/2 H_B P^-1/2) ~ 3.4-4.3 ~ 2(1+beta1)).
  * These reconcile ONLY IF Adam's PRECONDITIONED geometry rotates slower than its raw one,
    so the buffer CAN track the preconditioned unstable direction.

DESIGN PRINCIPLE: no metric may take an optimizer HYPERPARAMETER as an input. P = diag(sqrt(vhat)+eps)
is read from Adam's STATE (an observable). The realized step is a path observable. beta1 may be
used only to *cross-check* against 2(1+beta1); it must never appear in a computed metric. Each
metric is annotated "beta1 used: yes/no".

Subcommands:
  python experiments/adam_adjudicator.py adam    # Task A + Task B(i) + Task B(iii)  [DECISIVE]
  python experiments/adam_adjudicator.py arpca   # Task B(ii): step-PCA AR-pole (Adam/SGDM/SGD)
  python experiments/adam_adjudicator.py trend   # Task C: flat-vs-climbing kappa, catapult rate
  python experiments/adam_adjudicator.py all
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

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.measure import (
    compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
    _run_lobpcg_with_operator, flatt, param_vector,
)
import experiments.ar_pole_marginality as AR

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
NUM_DATA = int(os.environ.get('EOSS_NUM_DATA', '2048'))
MODEL = os.environ.get('EOSS_MODEL', 'mlp_s')
OUT_DIR = os.path.join(_REPO, 'results', 'adam_adjudicator')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


# ------------------------------------------------------------------ scaffold
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


def cosabs(a, b):
    d = (a * b).sum()
    return float(abs(d) / (a.norm() * b.norm() + 1e-30))


def buffer_flat(opt, params):
    """Adam m-buffer (exp_avg) or SGD-Momentum momentum_buffer, flat; cosine is
    bias-correction/scale invariant so exp_avg (unnormalized) is fine."""
    pieces = []
    for p in params:
        st = opt.inner.state.get(p)
        if st and 'exp_avg' in st:                 # Adam
            pieces.append(st['exp_avg'].flatten())
        elif st and 'momentum_buffer' in st and st['momentum_buffer'] is not None:
            pieces.append(st['momentum_buffer'].flatten())
        else:
            pieces.append(T.zeros(p.numel()))
    return T.cat(pieces).detach()


def sqrt_vhat_flat(opt):
    """Bias-corrected sqrt(v_hat) flat vector from Adam state (matches get_preconditioner_inv_sqrt)."""
    beta2 = opt.beta2
    pieces = []
    for p in opt.inner.param_groups[0]['params']:
        state = opt.inner.state.get(p)
        v = state['exp_avg_sq']
        t = state['step'].item() if T.is_tensor(state['step']) else state['step']
        v_hat = v / (1 - beta2 ** t)
        pieces.append(v_hat.sqrt().view(-1))
    return T.cat(pieces).detach()


def raw_top_eig(loss, net, cache):
    lam, u = compute_eigenvalues(loss, net, k=1, max_iterations=60, reltol=5e-3,
                                 eigenvector_cache=cache, return_eigenvectors=True,
                                 use_power_iteration=False)
    return float(lam), u.detach().reshape(-1)


def precond_top_eig(loss, net, d, cache):
    """Top eigen-pair of P^-1/2 H_B P^-1/2 with operator v -> d * H_B (d*v), d = D^-1/2."""
    hvp = create_hessian_vector_product(loss, net, retain_graph=True)

    def op(v):
        if v.ndim == 1:
            return d * hvp(d * v).detach()
        out = T.empty_like(v)
        for j in range(v.shape[1]):
            out[:, j] = d * hvp(d * v[:, j]).detach()
        return out
    try:
        eigvals, eigvecs = _run_lobpcg_with_operator(
            op, net, k=1, max_iterations=80, reltol=1e-2, init_vectors=None,
            eigenvector_cache=cache, return_eigenvectors=True)
    finally:
        hvp.free_memory()
    idx = int(T.argmax(eigvals))
    return float(eigvals[idx]), eigvecs[:, idx].detach().reshape(-1)


# ============================================================ TASK A + B(i) + B(iii)
def task_adam(lr=0.001, batch=8, beta1=0.9, beta2=0.99,
              train_steps=20000, window=400):
    """Train lean Adam b8 to plateau; over a ~window-step measurement window (continuing
    REAL training) with FROZEN P, measure the preconditioned rotation/tracking/GBS."""
    X, Y = get_data(); net, loss_fn = build()
    opt = create_optimizer('Adam', net, lr, {'beta1': beta1, 'beta2': beta2})
    params = [p for p in net.parameters() if p.requires_grad]
    t0 = time.time()
    for step in range(1, train_steps + 1):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        opt.zero_grad(); lo.backward(); opt.step()
    print(f"[adam] trained {train_steps} in {time.time()-t0:.0f}s loss={lo.item():.4f}", flush=True)

    # ---- freeze P (robust eps handling, from precond_edge_robust) ----
    sqv = sqrt_vhat_flat(opt)
    med = float(sqv.median())
    eps_rel = 0.1 * med                       # robust relative floor: dead coords can't blow up
    D_frozen = sqv + eps_rel
    d_frozen = (1.0 / D_frozen.sqrt()).detach()
    frac_floor = float((sqv < 1e-6).float().mean())
    print(f"[adam] sqrt(vhat) med={med:.3e} eps_rel={eps_rel:.3e} frac(<1e-6)={frac_floor:.3f}", flush=True)

    # ---- Task B(iii): kappa_precond eps stability (snapshot batches) ----
    eps_variants = [('adam_1e-8', 1e-8), ('rel_0.01', 0.01*med), ('rel_0.1', 0.1*med),
                    ('rel_1.0', 1.0*med), ('rel_10', 10.0*med)]
    kappa_eps = {name: [] for name, _ in eps_variants}
    for _ in range(8):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        for name, eps in eps_variants:
            d = (1.0 / (sqv + eps).sqrt()).detach()
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            try:
                lam_p, _ = precond_top_eig(lo, net, d, EigenvectorCache(1))
                kappa_eps[name].append(lr * lam_p)
            except Exception:
                pass
    kappa_eps_summ = {name: (float(np.mean(v)), float(np.std(v))) for name, v in kappa_eps.items() if v}
    print("[adam] Task B(iii) kappa_precond across eps:", flush=True)
    for name, (m, s) in kappa_eps_summ.items():
        print(f"   {name:12s} {m:7.3f} +/- {s:.3f}", flush=True)

    # ---- Task A + B(i): measurement window, FROZEN P ----
    cache_raw = EigenvectorCache(1)
    cache_pre = EigenvectorCache(1)
    u_raw_prev, u_pre_prev = None, None
    rot_raw, rot_pre = [], []
    cos_bufW_uP, cos_bufR_uR = [], []     # whitened buffer vs uP ; raw buffer vs uR
    cos_stepW_uP, cos_stepR_uR = [], []
    lam_raw_l, lam_pre_l = [], []
    gbs_raw_tot, gbs_raw_top, gbs_pre_top = [], [], []
    losses = []
    for t in range(window):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item(); losses.append(lv)
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads); gd = g.detach()
        s = opt.compute_step_direction(g, params).detach()     # realized Adam step (path observable)
        m = buffer_flat(opt, params)                           # Adam m-buffer (state observable)

        # raw top eigvec u_R of H_B (reuse the graph via grads)
        hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
        try:
            # raw GBS (total) and raw eig
            Hs = hvp(s, retain_graph_override=True).detach()
            A = float(-T.dot(gd, s)); B = float(T.dot(s, Hs))
            if abs(A) > 1e-18:
                gbs_raw_tot.append(B / A)
        finally:
            pass
        # raw top eig (own solve, warm-started)
        try:
            lam_r, u_R = raw_top_eig(lo, net, cache_raw)
            lam_raw_l.append(lam_r)
        except Exception:
            hvp.free_memory(); continue
        # raw top-mode GBS: s_top=(s.u)u
        cRs = float(T.dot(s, u_R)); cRg = float(T.dot(gd, u_R))
        if abs(cRs) > 1e-18 and abs(cRg) > 1e-18:
            gbs_raw_top.append(cRs * lam_r / (-cRg))
        hvp.free_memory()

        # preconditioned top eig ũ of P^-1/2 H_B P^-1/2 (frozen P), warm-started
        pr2 = net(Xb).squeeze(-1); lo2 = loss_fn(pr2, Yb)
        try:
            lam_p, u_P = precond_top_eig(lo2, net, d_frozen, cache_pre)
            lam_pre_l.append(lam_p)
        except Exception:
            continue
        # rotations
        if u_raw_prev is not None:
            rot_raw.append(cosabs(u_R, u_raw_prev))
        if u_pre_prev is not None:
            rot_pre.append(cosabs(u_P, u_pre_prev))
        u_raw_prev = u_R.clone(); u_pre_prev = u_P.clone()
        # whitened buffer/step:  m_tilde = D^-1/2 m = d*m ;  s_tilde = D^1/2 s = s/d
        m_tilde = d_frozen * m
        s_tilde = s / d_frozen
        g_tilde = d_frozen * gd
        cos_bufW_uP.append(cosabs(m_tilde, u_P))
        cos_bufR_uR.append(cosabs(m, u_R))
        cos_stepW_uP.append(cosabs(s_tilde, u_P))
        cos_stepR_uR.append(cosabs(s, u_R))
        # Task B(i): preconditioned top-mode GBS = (s~.u~) lam~ / (-(g~.u~))
        c = float(T.dot(s_tilde, u_P)); cg = float(T.dot(g_tilde, u_P))
        if abs(c) > 1e-18 and abs(cg) > 1e-18:
            gbs_pre_top.append(c * lam_p / (-cg))

        # advance one REAL step so geometry/buffer keep evolving
        opt.zero_grad(); lo3 = loss_fn(net(Xb).squeeze(-1), Yb); lo3.backward(); opt.step()

    def summ(a):
        a = np.array([x for x in a if np.isfinite(x)])
        return dict(mean=float(a.mean()) if len(a) else float('nan'),
                    med=float(np.median(a)) if len(a) else float('nan'),
                    std=float(a.std()) if len(a) else float('nan'), n=int(len(a)))

    mean_rot_raw = float(np.mean(rot_raw)); mean_rot_pre = float(np.mean(rot_pre))
    tau_rot_raw = 1.0 / max(1e-6, 1.0 - mean_rot_raw)
    tau_rot_pre = 1.0 / max(1e-6, 1.0 - mean_rot_pre)
    buf_mem = 1.0 / (1.0 - beta1)
    R_raw = buf_mem / tau_rot_raw
    R_precond = buf_mem / tau_rot_pre

    out = dict(
        cell='Adam_b8', lr=lr, batch=batch, beta1=beta1, beta2=beta2,
        train_steps=train_steps, window=window, eps_rel=eps_rel, med_sqrt_vhat=med,
        frac_floor=frac_floor,
        # --- Task A ---
        mean_cos_uB_raw=mean_rot_raw, mean_cos_uB_precond=mean_rot_pre,
        tau_rot_raw=tau_rot_raw, tau_rot_precond=tau_rot_pre,
        buf_mem=buf_mem, R_raw=R_raw, R_precond=R_precond,
        cos_buffer_uP_whitened=summ(cos_bufW_uP), cos_buffer_uR_raw=summ(cos_bufR_uR),
        cos_step_uP_whitened=summ(cos_stepW_uP), cos_step_uR_raw=summ(cos_stepR_uR),
        lam_raw=summ(lam_raw_l), lam_precond=summ(lam_pre_l),
        kappa_raw=float(lr * np.mean(lam_raw_l)), kappa_precond=float(lr * np.mean(lam_pre_l)),
        # --- Task B(i) ---
        gbs_raw_total=summ(gbs_raw_tot), gbs_raw_topmode=summ(gbs_raw_top),
        gbs_precond_topmode=summ(gbs_pre_top),
        edge_2=2.0, edge_2_1plusbeta1=2 * (1 + beta1),
        # --- Task B(iii) ---
        kappa_precond_eps_sweep={k: dict(mean=v[0], std=v[1]) for k, v in kappa_eps_summ.items()},
        beta1_annotations=dict(
            tau_rot_precond='no', cos_buffer_uP='no', gbs_precond_topmode='no (enters only via realized step, an observable)',
            kappa_precond='no', R_precond='yes (buffer-memory numerator 1/(1-beta1), same as raw R by definition)'),
    )
    with open(os.path.join(OUT_DIR, 'task_adam.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print("\n===== TASK A (adjudicator) =====", flush=True)
    print(f"  tau_rot RAW      = {tau_rot_raw:.2f}   (mean|cos(uR(t),uR(t-1))|={mean_rot_raw:.3f})", flush=True)
    print(f"  tau_rot PRECOND  = {tau_rot_pre:.2f}   (mean|cos(uP(t),uP(t-1))|={mean_rot_pre:.3f})", flush=True)
    print(f"  buf_mem 1/(1-b1) = {buf_mem:.1f}   R_raw={R_raw:.2f}   R_precond={R_precond:.2f}", flush=True)
    print(f"  cos(buffer,uP) WHITENED = {out['cos_buffer_uP_whitened']['mean']:.3f}   "
          f"cos(buffer,uR) RAW = {out['cos_buffer_uR_raw']['mean']:.3f}", flush=True)
    print(f"  cos(step,uP) WHITENED  = {out['cos_step_uP_whitened']['mean']:.3f}   "
          f"cos(step,uR) RAW  = {out['cos_step_uR_raw']['mean']:.3f}", flush=True)
    print("\n===== TASK B(i) preconditioned top-mode GBS (beta1 used: no) =====", flush=True)
    print(f"  raw TOTAL GBS       = {out['gbs_raw_total']['mean']:.3f}  (expect ~0.46)", flush=True)
    print(f"  raw TOP-MODE GBS    = {out['gbs_raw_topmode']['mean']:.3f}  (expect <2 per lean subspace exp)", flush=True)
    print(f"  PRECOND TOP-MODE GBS= {out['gbs_precond_topmode']['mean']:.3f}  (marginal prediction ~2)", flush=True)
    print("\n===== TASK B(iii) kappa_precond (~3.4-4.3?) =====", flush=True)
    print(f"  window kappa_precond = {out['kappa_precond']:.3f}   kappa_raw = {out['kappa_raw']:.3f}", flush=True)
    return out


# ============================================================ TASK B(ii): step-PCA AR-pole
def _sketch_indices(D, m, seed=12345):
    g = T.Generator().manual_seed(seed)
    return T.randperm(D, generator=g)[:m]


def task_arpca_cell(tag, optn, params, batch, lr, train_steps, log_steps,
                    m_sketch=16384):
    """Step-PCA AR-pole: x_t = p . theta_t with p = top PC of REALIZED steps (steps live in
    the optimizer's own geometry -> P applied for us, no P/beta needed). Uses the identity
    x_t (up to scale/const) = cumsum(v), v = top eigvec of the step Gram matrix. Steps are
    coordinate-sketched (random subset) to keep the Gram matrix cheap/memory-safe.
    beta1 used: no (steps are realized path observables)."""
    X, Y = get_data(); net, loss_fn = build()
    opt = create_optimizer(optn, net, lr, params)
    pl = [p for p in net.parameters() if p.requires_grad]
    D = sum(p.numel() for p in pl)
    idx_s = _sketch_indices(D, min(m_sketch, D))
    t0 = time.time()
    for step in range(train_steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item()
        if not np.isfinite(lv) or lv > 1e6:
            return dict(tag=tag, diverged=True)
        opt.zero_grad(); lo.backward(); opt.step()

    # log window: record realized step (theta_{t+1}-theta_t) sketch + batch loss
    Z = np.empty((log_steps, len(idx_s)), dtype=np.float32)
    losses = np.empty(log_steps, dtype=np.float64)
    theta_prev = param_vector(net)
    for t in range(log_steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        losses[t] = lo.item()
        opt.zero_grad(); lo.backward(); opt.step()
        theta = param_vector(net)
        s = (theta - theta_prev)
        Z[t] = s[idx_s].numpy()
        theta_prev = theta
    # Gram matrix of sketched steps, top eigvec -> x_t = cumsum(v)
    G = Z @ Z.T                                   # (T,T)
    w, V = np.linalg.eigh(G)
    v = V[:, -1]                                  # top eigvec
    x = np.cumsum(v).astype(float)
    x_detr, ema = AR.ema_detrend(x, AR.EMA_HALFLIFE)
    an = AR.analyze_series(x_detr, losses)
    np.savez(os.path.join(OUT_DIR, f'arpca_{tag}.npz'), x=x, x_detr=x_detr, losses=losses,
             top_gram_eig=float(w[-1]))
    dt = time.time() - t0
    print(f"  [arpca {tag}] rho(k3+)={an['rho_med_k3plus']:.3f}+/-{an['rho_spread_k3plus']:.3f} "
          f"phase/pi={an['phase_med_k3plus']/np.pi:+.2f} cats={an['n_catapults']} "
          f"win={an['window_len']} reliable={an['reliable']} ({dt:.0f}s)", flush=True)
    return dict(tag=tag, optimizer=optn, params=params, batch=batch, lr=lr,
                train_steps=train_steps, log_steps=log_steps, m_sketch=int(len(idx_s)),
                diverged=False, beta1_used='no', **an)


def task_arpca():
    cells = [
        ("Adam_b8",  "Adam",         {"beta1": 0.9, "beta2": 0.99}, 8, 0.001, 20000, 4000),
        ("SGDM_b8",  "SGD-Momentum", {"beta": 0.9},                 8, 0.002, 20000, 4000),
        ("SGD_b8",   "SGD",          {},                            8, 0.01,  20000, 4000),
    ]
    results = []
    for tag, optn, params, batch, lr, ts, ls in cells:
        print(f"\n=== arpca {tag} ===", flush=True)
        r = task_arpca_cell(tag, optn, params, batch, lr, ts, ls)
        results.append(r)
        with open(os.path.join(OUT_DIR, 'task_arpca.json'), 'w') as f:
            json.dump(results, f, indent=2)
    print("\n===== TASK B(ii) step-PCA AR-pole (beta1 used: no) =====", flush=True)
    print(f"{'cell':10s} {'rho(k3+)':>16s} {'phase/pi':>9s} {'cats':>5s} {'reliable':>8s}", flush=True)
    for r in results:
        if r.get('diverged'):
            print(f"{r['tag']:10s} diverged"); continue
        print(f"{r['tag']:10s} {r['rho_med_k3plus']:.3f}+/-{r['rho_spread_k3plus']:.3f}   "
              f"{r['phase_med_k3plus']/np.pi:+9.2f} {r['n_catapults']:5d} {str(r['reliable']):>8s}", flush=True)
    return results


# ============================================================ TASK C: flat vs climbing kappa
def _kappa_raw_at(net, X, Y, loss_fn, lr, batch, n=10):
    cache = EigenvectorCache(1); vals = []
    for _ in range(n):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        try:
            lam, _ = raw_top_eig(lo, net, cache); vals.append(lr * lam)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float('nan')


def _kappa_precond_at(net, X, Y, loss_fn, opt, lr, batch, n=10):
    sqv = sqrt_vhat_flat(opt); med = float(sqv.median())
    d = (1.0 / (sqv + 0.1 * med).sqrt()).detach()
    cache = EigenvectorCache(1); vals = []
    for _ in range(n):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        try:
            lam, _ = precond_top_eig(lo, net, d, cache); vals.append(lr * lam)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float('nan')


def task_trend_cell(tag, optn, params, batch, lr, steps=30000, ckpts=(5000, 10000, 15000, 20000, 25000, 30000)):
    X, Y = get_data(); net, loss_fn = build()
    opt = create_optimizer(optn, net, lr, params)
    pl = [p for p in net.parameters() if p.requires_grad]
    is_adam = (optn == 'Adam')
    ckpts = sorted(ckpts); ci = 0
    losses = []; traj = []
    for step in range(1, steps + 1):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb); lv = lo.item(); losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            print(f"  [trend {tag}] diverged @ {step}"); break
        opt.zero_grad(); lo.backward(); opt.step()
        if ci < len(ckpts) and step == ckpts[ci]:
            k = (_kappa_precond_at(net, X, Y, loss_fn, opt, lr, batch) if is_adam
                 else _kappa_raw_at(net, X, Y, loss_fn, lr, batch))
            traj.append(dict(step=step, kappa=k, loss=lv))
            print(f"  [trend {tag}] {step:6d} kappa={k:.3f} loss={lv:.4f}", flush=True)
            ci += 1
    # slope over last 3 checkpoints
    tk = traj[-3:] if len(traj) >= 3 else traj
    st = np.array([t['step'] for t in tk], float); kv = np.array([t['kappa'] for t in tk], float)
    slope = float(np.polyfit(st, kv, 1)[0]) if len(tk) >= 2 else float('nan')
    drift = float(slope * (st[-1] - st[0])) if len(tk) >= 2 else float('nan')
    plateau = losses[int(len(losses) * 0.6):]
    cats = AR.L.detect_catapults(np.array(plateau)) if hasattr(AR, 'L') else []
    from experiments.long_train_grid import detect_catapults
    cats = detect_catapults(np.array(plateau))
    cat_rate = len(cats) / max(1, len(plateau))
    trend = 'flat' if abs(drift) < 0.15 * abs(kv[-1]) else ('climbing' if drift > 0 else 'declining')
    print(f"  [trend {tag}] kappa_type={'precond' if is_adam else 'raw'} last={kv[-1]:.3f} "
          f"slope/step={slope:.2e} drift(last3)={drift:.3f} => {trend}  cat_rate={cat_rate:.4f}", flush=True)
    return dict(tag=tag, optimizer=optn, kappa_type=('precond' if is_adam else 'raw'),
                traj=traj, slope_per_step=slope, drift_last3=drift, trend=trend,
                catapult_rate=cat_rate, n_catapults=len(cats))


def task_trend():
    cells = [
        ("SGDM_b8", "SGD-Momentum", {"beta": 0.9},                 8, 0.002),
        ("Adam_b8", "Adam",         {"beta1": 0.9, "beta2": 0.99}, 8, 0.001),
    ]
    results = []
    for tag, optn, params, batch, lr in cells:
        print(f"\n=== trend {tag} ===", flush=True)
        r = task_trend_cell(tag, optn, params, batch, lr)
        results.append(r)
        with open(os.path.join(OUT_DIR, 'task_trend.json'), 'w') as f:
            json.dump(results, f, indent=2)
    print("\n===== TASK C flat-vs-climbing kappa =====", flush=True)
    for r in results:
        print(f"  {r['tag']:10s} kappa({r['kappa_type']}) trend={r['trend']} "
              f"drift(last3)={r['drift_last3']:.3f} cat_rate={r['catapult_rate']:.4f}", flush=True)
    return results


# ============================================================ main
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('adam', 'all'):
        task_adam()
    if which in ('arpca', 'all'):
        task_arpca()
    if which in ('trend', 'all'):
        task_trend()


if __name__ == '__main__':
    main()
