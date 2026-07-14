"""
Comprehensive marginal-vs-metastable sweep -- ONE grid point (B, beta, lr) worker.

Maps the SGD-Momentum regime (mlp_s/2048/MSE) across batch B x momentum beta x learning
rate, with THREE causal, yardstick-free instruments (NO deterministic-edge formula anywhere):

  metrics trajectory : every ~checkpoint log step,loss,GBS,lambda(H_B),kappa=eta*lambda, and
                       R = (buffer memory 1/(1-beta)) / (u_B rotation timescale tau_rot),
                       with cos(buffer,u_B)/cos(grad,u_B). R<1 marginal, R>>1 metastable.
  TEST 1 perturb-relax: kick along u, measure relaxation rate gamma (proj / full-||delta|| /
                        precond metrics), 3 small linear-regime scales x 5 seeds -> mean+/-std
                        with per-fit CI; plus a scale sweep to locate the ESCAPE threshold.
                        gamma~0 & no escape => MARGINAL; gamma<0 & finite escape => METASTABLE.
  TEST 2 sharpening-suppression: drop lr/10 for a stretch, 5 seeds; net sharpness rise
                        (lr-drop minus same-length baseline). Marginal resumes climbing;
                        metastable stays flat. Loss logged to flag the interpolation confound.
  TEST 3 catapult stats: rotation-robust unstable coordinate x_t = ||V^T theta_t|| (top-K
                        subspace amplitude, NOT fixed-u) + detrended x-EMA over a long plateau;
                        FULL arrays saved; kurtosis + p99/p50 + Hill tail index.

Reuses train/relax/gamma-fit from perturb_relax, R machinery from mechanism_buffer_rotation,
lam_subset from causal_regime_tests, Hill from kesten_test, data/build from long_train_grid.

Usage:
  python -m experiments.comprehensive_sweep --B 8 --beta 0.9 --lr 0.002 --lr_index 2 --tag b8_beta0.9_lr2
Outputs -> results/comprehensive_sweep/<tag>/{metrics_traj.npz, perturb.json, perturb_series.npz,
           sharpening.json, catapult.npz, catapult_stats.json, meta.json}
"""
import os, sys, json, copy, time, argparse
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
import experiments.mechanism_buffer_rotation as M
import experiments.perturb_relax as PR
import experiments.causal_regime_tests as C
import experiments.kesten_test as KE
from utils.optimizer import create_optimizer
from utils.measure import (compute_eigenvalues, EigenvectorCache, create_hessian_vector_product,
                           flatt, param_vector)
from utils.curvature_segment import set_params_inplace

T.set_num_threads(int(os.environ.get('EOSS_THREADS', '4')))
OUT_ROOT = os.path.join(_REPO, 'results', 'comprehensive_sweep')
os.makedirs(OUT_ROOT, exist_ok=True)

OPTN = 'SGD-Momentum'


# ----------------------------------------------------------------------------- metrics probes
def light_probe(net, loss_fn, X, Y, opt, lr, batch, n=8):
    """Cheap per-batch GBS + lambda(H_B) (no full-batch baseline)."""
    params = [p for p in net.parameters() if p.requires_grad]
    cache = EigenvectorCache(1)
    Gbs, Lam, Bs = [], [], []
    for _ in range(n):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads); gd = g.detach(); s = opt.compute_step_direction(g, params).detach()
        hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
        try:
            Hs = hvp(s, retain_graph_override=True)
            Hg = hvp(gd, retain_graph_override=False)
            A = -T.dot(gd, s).item(); B = T.dot(s, Hs).item()
            gg = T.dot(gd, gd).item(); gHg = T.dot(gd, Hg).item()
        finally:
            hvp.free_memory()
        if abs(A) > 1e-15 and np.isfinite(B / A):
            Gbs.append(B / A)
        if gg > 1e-24 and np.isfinite(gHg / gg):
            Bs.append(gHg / gg)          # batch sharpness g^T H_B g / ||g||^2
        pr2 = net(Xb).squeeze(-1); lo2 = loss_fn(pr2, Yb)
        try:
            lam = compute_eigenvalues(lo2, net, k=1, max_iterations=40, reltol=0.02,
                                      eigenvector_cache=cache, return_eigenvectors=False,
                                      use_power_iteration=False).item()
        except Exception:
            lam = float('nan')
        if np.isfinite(lam) and lam > 0:
            Lam.append(lam)
    return (float(np.mean(Gbs)) if Gbs else float('nan'),
            float(np.mean(Lam)) if Lam else float('nan'),
            float(np.mean(Bs)) if Bs else float('nan'))


def R_window(net, loss_fn, X, Y, opt, batch, beta, n=20):
    """Advance n real steps computing u_B(t) each step -> tau_rot, R, alignments.
    These steps count as training (the trajectory keeps evolving)."""
    params = [p for p in net.parameters() if p.requires_grad]
    cache = EigenvectorCache(1)
    is_mom_active = beta > 0
    u_prev = None
    rot, cos_bu, cos_gu, cos_su, cos_bg, cos_sg = [], [], [], [], [], []
    gn, bn, sn, lams = [], [], [], []
    for t in range(n):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        if not np.isfinite(lo.item()) or lo.item() > 1e6:
            return None
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads); gd = g.detach()
        m = M.buffer_flat(opt, params) if is_mom_active else gd
        s = opt.compute_step_direction(g, params).detach()
        lam, u = M.batch_top_eigvec(net, loss_fn, Xb, Yb, cache)
        lams.append(lam)
        if u_prev is not None:
            rot.append(M.cosabs(u, u_prev))       # |cos(u_t, u_{t+1})| -- rotation overlap
        u_prev = u.clone()
        cos_bu.append(M.cosabs(m, u)); cos_gu.append(M.cosabs(gd, u)); cos_su.append(M.cosabs(s, u))
        cos_bg.append(M.cosabs(m, gd)); cos_sg.append(M.cosabs(s, gd))
        gn.append(float(gd.norm())); bn.append(float(m.norm())); sn.append(float(s.norm()))
        opt.zero_grad(); lo2 = loss_fn(net(Xb).squeeze(-1), Yb); lo2.backward(); opt.step()
    mean_rot = float(np.mean(rot)) if rot else float('nan')
    tau = 1.0 / max(1e-6, 1.0 - mean_rot)
    buf_mem = 1.0 / (1.0 - beta) if is_mom_active else 1.0
    def _m(a): return float(np.mean(a)) if a else float('nan')
    return dict(tau_rot=tau, R=buf_mem / tau, buf_mem=buf_mem, mean_cos_uB=mean_rot,
                cos_buf_uB=_m(cos_bu), cos_grad_uB=_m(cos_gu), cos_step_uB=_m(cos_su),
                cos_buf_grad=_m(cos_bg), cos_step_grad=_m(cos_sg),
                grad_norm=_m(gn), buf_norm=_m(bn), step_norm=_m(sn), lam_batch=_m(lams))


# ----------------------------------------------------------------------------- training + snapshot
def train_and_log(beta, batch, lr, steps_max, measure_every, steps_min_frac=0.6,
                  step_window=200, rwin=20):
    """Train to plateau with a stationarity gate; log a rich metrics trajectory. Returns the
    perturb_relax-style state dict (net/opt/theta_star/u_hess/u_step/...) + trajectory + status."""
    X, Y = L.get_data(); net, loss_fn = L.build()
    params_dict = {'beta': beta}
    opt = create_optimizer(OPTN, net, lr, params_dict)
    params_l = [p for p in net.parameters() if p.requires_grad]
    traj = []
    recent_steps = []
    step = 0
    steps_min = int(steps_max * steps_min_frac)
    diverged = False
    last_loss = float('nan')
    while step < steps_max:
        block = max(1, measure_every - rwin)
        for _ in range(block):
            idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            last_loss = lo.item()
            if not np.isfinite(last_loss) or last_loss > 1e6:
                diverged = True; break
            opt.zero_grad(); lo.backward()
            if step >= steps_max - step_window:
                g = flatt([p.grad for p in params_l]).detach()
                recent_steps.append(opt.compute_step_direction(g, params_l).detach().clone())
            opt.step(); step += 1
        if diverged:
            break
        # R window (advances rwin real steps) + light GBS/lambda probe
        rw = R_window(net, loss_fn, X, Y, opt, batch, beta, n=rwin)
        step += rwin
        if rw is None:
            diverged = True; break
        gbs, lam, bs = light_probe(net, loss_fn, X, Y, opt, lr, batch)
        kappa = lr * lam if np.isfinite(lam) else float('nan')
        rec = dict(step=step, loss=last_loss, gbs=gbs, lam=lam, kappa=kappa,
                   batch_sharpness=bs, eta_bs=lr * bs if np.isfinite(bs) else float('nan'),
                   alpha_g=(gbs / kappa if (np.isfinite(kappa) and abs(kappa) > 1e-9) else float('nan')),
                   R=rw['R'], tau_rot=rw['tau_rot'], buf_mem=rw['buf_mem'], mean_cos_uB=rw['mean_cos_uB'],
                   cos_buf_uB=rw['cos_buf_uB'], cos_grad_uB=rw['cos_grad_uB'], cos_step_uB=rw['cos_step_uB'],
                   cos_buf_grad=rw['cos_buf_grad'], cos_step_grad=rw['cos_step_grad'],
                   grad_norm=rw['grad_norm'], buf_norm=rw['buf_norm'], step_norm=rw['step_norm'],
                   lam_batch=rw['lam_batch'])
        traj.append(rec)
        print(f"    step={step:6d} loss={last_loss:.4g} GBS={gbs:.3f} lam={lam:.2f} "
              f"kappa={rec['kappa']:.2f} R={rw['R']:.2f} cosBu={rw['cos_buf_uB']:.3f}", flush=True)
        # stationarity early-stop on GBS
        if step >= steps_min:
            st = L.stationarity(traj, 'gbs', back_frac=0.5)
            if st.get('stabilized'):
                break
    st = L.stationarity(traj, 'gbs', back_frac=0.5) if len(traj) >= 5 else dict(stabilized=False, reason='short')
    if diverged:
        return dict(diverged=True, traj=traj, stationarity=st, steps=step)
    # -------- snapshot for the tests (matches perturb_relax.train_plateau state schema)
    theta_star = param_vector(net).detach().clone()
    net_sd = copy.deepcopy(net.state_dict())
    opt_sd = copy.deepcopy(opt.inner.state_dict())
    Xs, Ys = X[:2048], Y[:2048]
    pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, u_hess = compute_eigenvalues(lo, net, k=1, max_iterations=80, reltol=0.005,
                                    eigenvector_cache=EigenvectorCache(1),
                                    return_eigenvectors=True, use_power_iteration=False)
    u_hess = u_hess.detach(); u_hess = u_hess / u_hess.norm()
    if recent_steps:
        S = T.stack(recent_steps, 0); S = S - S.mean(0, keepdim=True)
        try:
            _, _, Vh = T.linalg.svd(S, full_matrices=False)
            u_step = Vh[0]; u_step = u_step / u_step.norm()
        except Exception:
            u_step = u_hess.clone()
    else:
        u_step = u_hess.clone()
    state = dict(net=net, opt=opt, loss_fn=loss_fn, XY=(X, Y), theta_star=theta_star,
                 net_sd=net_sd, opt_sd=opt_sd, u_hess=u_hess, u_step=u_step, batch=batch,
                 optn=OPTN, lr=lr, params=params_dict)
    return dict(diverged=False, traj=traj, stationarity=st, steps=step, state=state)


# ----------------------------------------------------------------------------- TEST 1 perturb-relax
def _fit_gamma(sig, lo=2, hi=40):
    """slope of log|sig| over an early linear window + stderr of the slope (fit CI)."""
    w = np.asarray(sig)[lo:hi]
    aw = np.abs(w[np.isfinite(w)])
    m = aw > 0
    x = np.arange(len(aw))[m]; y = np.log(aw[m])
    if len(y) < 6:
        return float('nan'), float('nan')
    Amat = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
    slope = float(coef[0])
    resid = y - Amat @ coef
    dof = max(1, len(y) - 2)
    s2 = float(resid @ resid) / dof
    sxx = float(((x - x.mean()) ** 2).sum()) + 1e-30
    se = float(np.sqrt(s2 / sxx))
    return slope, se


def relax_lockstep(state, u, amp, K, seed):
    """Run a kicked copy and an un-kicked reference in lockstep on identical batches.
    Returns u-projected deviation, full ||delta||, and kicked loss series."""
    netK, optK, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']; theta_star = state['theta_star']; batch = state['batch']
    netK.load_state_dict(copy.deepcopy(state['net_sd'])); PR.set_optimizer_state(optK, state['opt_sd'])
    with T.no_grad():
        set_params_inplace(netK, theta_star + amp * u)
    netR, _ = L.build(); netR.load_state_dict(copy.deepcopy(state['net_sd']))
    optR = create_optimizer(state['optn'], netR, state['lr'], state['params'])
    PR.set_optimizer_state(optR, state['opt_sd'])
    g = T.Generator().manual_seed(seed)
    xproj, dnorm, losses = [], [], []
    for t in range(K):
        with T.no_grad():
            dv = param_vector(netK) - param_vector(netR)
            xproj.append(float(T.dot(dv, u))); dnorm.append(float(dv.norm()))
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        bad = False
        for net, opt in ((netK, optK), (netR, optR)):
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            if net is netK:
                losses.append(lo.item())
            if not np.isfinite(lo.item()) or lo.item() > 1e6:
                bad = True
            opt.zero_grad(); lo.backward(); opt.step()
        if bad:
            pad = K - len(xproj)
            xproj += [float('nan')] * pad; dnorm += [float('nan')] * pad; losses += [float('nan')] * (K - len(losses))
            break
    return np.array(xproj), np.array(dnorm), np.array(losses)


def test_perturb(state, uname, u, K=150):
    # natural amplitude scale = std of free (un-kicked) u-projection
    x0, _ = PR.relax(state, u, 0.0, K=K, seed=1)
    scale = float(np.nanstd(x0)) + 1e-12
    series = {}
    # (a) 3 small linear-regime scales x 5 seeds -> gamma proj/full/precond mean+/-std + CI
    scale_mults = [0.3, 1.0, 3.0]
    per_scale = []
    for sm in scale_mults:
        amp = sm * scale
        gp, gf, ci_p, ci_f = [], [], [], []
        for seed in range(5):
            xproj, dnorm, _ = relax_lockstep(state, u, amp, K, seed=100 + seed)
            gp_s, se_p = _fit_gamma(xproj)
            gf_s, se_f = _fit_gamma(dnorm)
            gp.append(gp_s); gf.append(gf_s); ci_p.append(se_p); ci_f.append(se_f)
            series[f"{uname}_sm{sm}_seed{seed}_xproj"] = xproj
            series[f"{uname}_sm{sm}_seed{seed}_dnorm"] = dnorm
        gp, gf = np.array(gp), np.array(gf)
        rec = dict(scale_mult=sm, amp=float(amp),
                   gamma_proj_mean=float(np.nanmean(gp)), gamma_proj_std=float(np.nanstd(gp)),
                   gamma_proj_ci_mean=float(np.nanmean(ci_p)),
                   gamma_full_mean=float(np.nanmean(gf)), gamma_full_std=float(np.nanstd(gf)),
                   gamma_full_ci_mean=float(np.nanmean(ci_f)),
                   # pure momentum: preconditioner = Euclidean identity => precond == full metric
                   gamma_precond_mean=float(np.nanmean(gf)), gamma_precond_std=float(np.nanstd(gf)),
                   n_seeds=int(np.sum(np.isfinite(gp))))
        per_scale.append(rec)
    # (b) single-seed scale sweep to locate the ESCAPE threshold
    esc_mults = [0.3, 1.0, 3.0, 6.0, 10.0, 20.0, 40.0]
    esc = []
    threshold = None
    for sm in esc_mults:
        amp = sm * scale
        xa, la = PR.relax(state, u, amp, K=K, seed=7)
        x0b, _ = PR.relax(state, u, 0.0, K=K, seed=7)
        dx = xa - x0b
        gsm, _ = _fit_gamma(dx)
        escaped = bool(np.any(~np.isfinite(la)) or np.nanmax(np.abs(dx)) > 20 * (amp + scale))
        esc.append(dict(scale_mult=sm, gamma=gsm, max_absdx=float(np.nanmax(np.abs(dx))), escaped=escaped))
        if escaped and threshold is None:
            threshold = sm
    return dict(kick_dir=uname, natural_scale=scale, per_scale=per_scale,
                escape_sweep=esc, escape_threshold_over_natural=threshold), series


# ----------------------------------------------------------------------------- TEST 2 sharpening
def test_sharpening(state, lr, batch, stretch=1000, every=50, seeds=5):
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']

    def run_at(mult, seed):
        net.load_state_dict(copy.deepcopy(state['net_sd']))
        PR.set_optimizer_state(opt, state['opt_sd'])
        opt.inner.param_groups[0]['lr'] = lr * mult
        cache = EigenvectorCache(1); lams, losses = [], []
        g = T.Generator().manual_seed(seed)
        for t in range(stretch):
            if t % every == 0:
                lams.append(C.lam_subset(net, loss_fn, X, Y, cache))
            idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
            pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
            lv = lo.item(); losses.append(lv)
            if not np.isfinite(lv) or lv > 1e6:
                break
            opt.zero_grad(); lo.backward(); opt.step()
        opt.inner.param_groups[0]['lr'] = lr
        return np.array(lams), np.array(losses)

    def rel_rise(a):
        return float((a[-1] - a[0]) / (a[0] + 1e-12)) if len(a) > 1 else float('nan')

    drop_rises, base_rises, net_rises = [], [], []
    drop_min_loss, base_min_loss = [], []
    ex_drop, ex_base = None, None
    for s in range(seeds):
        drop_l, drop_loss = run_at(0.1, seed=11 + s)
        base_l, base_loss = run_at(1.0, seed=11 + s)
        dr, br = rel_rise(drop_l), rel_rise(base_l)
        drop_rises.append(dr); base_rises.append(br); net_rises.append(dr - br)
        drop_min_loss.append(float(np.nanmin(drop_loss)) if len(drop_loss) else float('nan'))
        base_min_loss.append(float(np.nanmin(base_loss)) if len(base_loss) else float('nan'))
        if s == 0:
            ex_drop, ex_base = drop_l.tolist(), base_l.tolist()
    net_rises = np.array(net_rises)
    interpolated = bool(np.nanmin(drop_min_loss + base_min_loss) < 1e-3)
    return dict(lr_drop_rel_rise_mean=float(np.nanmean(drop_rises)),
                lr_drop_rel_rise_std=float(np.nanstd(drop_rises)),
                baseline_rel_rise_mean=float(np.nanmean(base_rises)),
                baseline_rel_rise_std=float(np.nanstd(base_rises)),
                net_rise_mean=float(np.nanmean(net_rises)), net_rise_std=float(np.nanstd(net_rises)),
                min_loss_drop=float(np.nanmin(drop_min_loss)), min_loss_base=float(np.nanmin(base_min_loss)),
                interpolated=interpolated, n_seeds=seeds,
                example_drop_lambda=ex_drop, example_base_lambda=ex_base)


# ----------------------------------------------------------------------------- TEST 3 catapult
def test_catapult(state, batch, window=4000, ema_hl=100, K=5):
    """Rotation-robust unstable coordinate: x_t = ||V^T theta_t|| in the top-K subspace,
    detrended by an EMA. Save full arrays; compute kurtosis, p99/p50, Hill index."""
    from scipy import stats as _stats
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']
    net.load_state_dict(copy.deepcopy(state['net_sd']))
    PR.set_optimizer_state(opt, state['opt_sd'])
    Xs, Ys = X[:2048], Y[:2048]; pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    _, V = compute_eigenvalues(lo, net, k=K, max_iterations=80, reltol=0.01,
                               eigenvector_cache=EigenvectorCache(K), return_eigenvectors=True,
                               use_power_iteration=False)
    V = V.detach()
    alpha = 1 - 0.5 ** (1.0 / ema_hl)
    p = (V.t() @ param_vector(net)).numpy()
    p_ema = p.copy()
    x_raw, x_det, losses = [], [], []
    g = T.Generator().manual_seed(23)
    for t in range(window):
        p = (V.t() @ param_vector(net)).numpy()
        p_ema = (1 - alpha) * p_ema + alpha * p
        x_raw.append(float(np.linalg.norm(p)))
        x_det.append(float(np.linalg.norm(p - p_ema)))
        idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item(); losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            break
        opt.zero_grad(); lo.backward(); opt.step()
    x_raw = np.array(x_raw); x_det = np.array(x_det); losses = np.array(losses)
    ax = np.abs(x_det[np.isfinite(x_det)])
    stats = dict(n=int(len(ax)))
    if len(ax) >= 50:
        med = float(np.median(ax)) + 1e-30
        stats.update(kurtosis=float(_stats.kurtosis(x_det[np.isfinite(x_det)])),
                     p99_over_p50=float(np.quantile(ax, 0.99) / med),
                     cat_rate=float(np.sum(ax > 5 * med) / len(ax)),
                     hill_index=KE.hill(x_det))
    else:
        stats.update(kurtosis=float('nan'), p99_over_p50=float('nan'), cat_rate=float('nan'),
                     hill_index={})
    return x_raw, x_det, losses, stats


# ----------------------------------------------------------------------------- driver entry
def run_cell(B, beta, lr, lr_index, tag):
    out = os.path.join(OUT_ROOT, tag)
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, 'meta.json')
    if os.path.exists(meta_path):
        print(f"[{tag}] already done, skipping", flush=True)
        return
    t0 = time.time()
    # training-length schedule by batch (small batch needs more steps)
    sched = {8: (18000, 800), 32: (14000, 700), 128: (10000, 600), 512: (8000, 500), 2048: (6000, 400)}
    steps_max, me = sched.get(B, (12000, 600))
    if os.environ.get('EOSS_SMOKE'):   # fast end-to-end validation
        steps_max, me = 1600, 400
    print(f"\n===== CELL {tag}: B={B} beta={beta} lr={lr} (idx {lr_index}) steps_max={steps_max} =====", flush=True)

    tr = train_and_log(beta, B, lr, steps_max, me)
    meta = dict(tag=tag, B=B, beta=beta, lr=lr, lr_index=lr_index, optimizer=OPTN,
                steps_max=steps_max, measure_every=me, steps_trained=tr['steps'],
                diverged=tr['diverged'], stationarity=tr['stationarity'])
    # save trajectory
    traj = tr['traj']
    if traj:
        np.savez(os.path.join(out, 'metrics_traj.npz'),
                 **{k: np.array([t.get(k, np.nan) for t in traj]) for k in
                    ('step', 'loss', 'gbs', 'lam', 'kappa', 'batch_sharpness', 'eta_bs', 'alpha_g',
                     'R', 'tau_rot', 'buf_mem', 'mean_cos_uB', 'cos_buf_uB', 'cos_grad_uB', 'cos_step_uB',
                     'cos_buf_grad', 'cos_step_grad', 'grad_norm', 'buf_norm', 'step_norm', 'lam_batch')})
        tail = traj[max(0, len(traj) // 2):]
        meta['plateau_gbs'] = float(np.nanmean([t['gbs'] for t in tail]))
        meta['plateau_kappa'] = float(np.nanmean([t['kappa'] for t in tail]))
        meta['plateau_R'] = float(np.nanmean([t['R'] for t in tail]))
        meta['plateau_cos_buf_uB'] = float(np.nanmean([t['cos_buf_uB'] for t in tail]))
    if tr['diverged']:
        meta['status'] = 'diverged'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[{tag}] DIVERGED during training at step {tr['steps']}", flush=True)
        return
    state = tr['state']

    errors = {}
    # ---- TEST 1
    try:
        perturb = {}
        all_series = {}
        for uname, u in (('u_hess', state['u_hess']), ('u_step', state['u_step'])):
            res, series = test_perturb(state, uname, u)
            perturb[uname] = res
            all_series.update(series)
        with open(os.path.join(out, 'perturb.json'), 'w') as f:
            json.dump(perturb, f, indent=2)
        np.savez(os.path.join(out, 'perturb_series.npz'),
                 **{k: v for k, v in all_series.items()})
        lin = perturb['u_hess']['per_scale'][0]  # smallest (0.3x) scale = linear regime
        meta['perturb_gamma_proj_mean'] = lin['gamma_proj_mean']
        meta['perturb_gamma_proj_std'] = lin['gamma_proj_std']
        meta['perturb_gamma_full_mean'] = lin['gamma_full_mean']
        meta['perturb_escape_over_natural'] = perturb['u_hess']['escape_threshold_over_natural']
        print(f"[{tag}] TEST1 gamma_proj(0.3x)={lin['gamma_proj_mean']:+.4f}+/-{lin['gamma_proj_std']:.4f} "
              f"escape={perturb['u_hess']['escape_threshold_over_natural']}", flush=True)
    except Exception as e:
        errors['perturb'] = repr(e); print(f"[{tag}] TEST1 error: {e}", flush=True)

    # ---- TEST 2
    try:
        sharp = test_sharpening(state, lr, B)
        with open(os.path.join(out, 'sharpening.json'), 'w') as f:
            json.dump(sharp, f, indent=2)
        meta['sharpen_net_rise_mean'] = sharp['net_rise_mean']
        meta['sharpen_net_rise_std'] = sharp['net_rise_std']
        meta['sharpen_interpolated'] = sharp['interpolated']
        print(f"[{tag}] TEST2 net_rise={sharp['net_rise_mean']:+.3f}+/-{sharp['net_rise_std']:.3f} "
              f"interp={sharp['interpolated']}", flush=True)
    except Exception as e:
        errors['sharpening'] = repr(e); print(f"[{tag}] TEST2 error: {e}", flush=True)

    # ---- TEST 3
    try:
        x_raw, x_det, closs, cstats = test_catapult(state, B)
        np.savez(os.path.join(out, 'catapult.npz'), x_raw=x_raw, x_detrended=x_det, loss=closs)
        with open(os.path.join(out, 'catapult_stats.json'), 'w') as f:
            json.dump(cstats, f, indent=2)
        meta['catapult_kurtosis'] = cstats.get('kurtosis')
        meta['catapult_p99_p50'] = cstats.get('p99_over_p50')
        meta['catapult_hill'] = cstats.get('hill_index', {}).get(0.1) if isinstance(cstats.get('hill_index'), dict) else None
        print(f"[{tag}] TEST3 kurt={cstats.get('kurtosis'):.2f} p99/p50={cstats.get('p99_over_p50'):.2f} "
              f"hill@10%={meta['catapult_hill']}", flush=True)
    except Exception as e:
        errors['catapult'] = repr(e); print(f"[{tag}] TEST3 error: {e}", flush=True)

    meta['errors'] = errors
    meta['status'] = 'done' if not errors else 'partial'
    meta['elapsed_s'] = round(time.time() - t0, 1)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{tag}] {meta['status'].upper()} in {meta['elapsed_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, required=True)
    ap.add_argument('--beta', type=float, required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--lr_index', type=int, required=True)
    ap.add_argument('--tag', type=str, required=True)
    a = ap.parse_args()
    run_cell(a.B, a.beta, a.lr, a.lr_index, a.tag)


if __name__ == '__main__':
    main()
