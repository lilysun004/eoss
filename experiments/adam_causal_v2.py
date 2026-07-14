"""Adam perturb-and-relax, geometry-aware. For each cell kick along u_hess, u_step, AND (Adam)
the PRECONDITIONED top eigvec u_precond (top eigvec of P^{-1/2} H P^{-1/2}, P from Adam state).
Track the induced deviation delta_t = theta_kick - theta_ref and fit the decay rate in THREE
measures: |u.delta| (Euclidean projection), ||delta|| (full Euclidean norm), and ||P^{1/2}
delta|| (preconditioned norm, Adam only). Lyapunov rate is metric-independent asymptotically,
so gamma's SIGN should agree across measures; reporting all is the robustness check the
preconditioner geometry demands. gamma~0 => marginal (persists); gamma<0 => metastable (decays)."""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
import numpy as np, torch as T
import experiments.perturb_relax as PR
from utils.measure import compute_eigenvalues, EigenvectorCache, create_hessian_vector_product, flatt, param_vector
from utils.measure import _run_lobpcg_with_operator
from utils.curvature_segment import set_params_inplace
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'adam_causal')
os.makedirs(OUT, exist_ok=True)

def precond_top_eigvec(net, loss_fn, X, Y, dinv, cap=2048):
    Xs, Ys = X[:cap], Y[:cap]; pr = net(Xs).squeeze(-1); lo = loss_fn(pr, Ys)
    hvp = create_hessian_vector_product(lo, net, retain_graph=True); d = dinv
    def op(v):
        if v.ndim == 1: return d * hvp(d * v).detach()
        out = T.empty_like(v)
        for j in range(v.shape[1]): out[:, j] = d * hvp(d * v[:, j]).detach()
        return out
    try:
        _, vecs = _run_lobpcg_with_operator(op, net, 1, 60, 0.01, None, EigenvectorCache(1), True)
    finally: hvp.free_memory()
    u = vecs[:, 0].detach(); return u / u.norm()

def relax_geom(state, u, amp, dinv, K=150, seed=7):
    net, opt, loss_fn = state['net'], state['opt'], state['loss_fn']
    X, Y = state['XY']; theta0 = state['theta_star']; batch = state['batch']
    def run(kick):
        net.load_state_dict(copy.deepcopy(state['net_sd'])); PR.set_optimizer_state(opt, state['opt_sd'])
        with T.no_grad(): set_params_inplace(net, theta0 + kick)
        g = T.Generator().manual_seed(seed); thetas = []
        for t in range(K):
            thetas.append(param_vector(net).clone())
            idx = T.randperm(len(X), generator=g)[:batch]; Xb, Yb = X[idx], Y[idx]
            lo = loss_fn(net(Xb).squeeze(-1), Yb)
            if not np.isfinite(lo.item()) or lo.item() > 1e6: break
            opt.zero_grad(); lo.backward(); opt.step()
        return thetas
    tk = run(amp * u); tr = run(0.0 * u)
    n = min(len(tk), len(tr)); 
    proj, full, pre = [], [], []
    for t in range(n):
        d = tk[t] - tr[t]
        proj.append(abs(float(T.dot(d, u)))); full.append(float(d.norm()))
        pre.append(float((d / dinv).norm()) if dinv is not None else float('nan'))
    return np.array(proj), np.array(full), np.array(pre)

def gamma_of(series):
    w = series[2:40]; w = w[np.isfinite(w) & (w > 0)]
    if len(w) < 6: return float('nan')
    return float(np.polyfit(np.arange(len(w)), np.log(w), 1)[0])

cells = [
    ("SGD_b8",  "SGD",          {},                          0.01,  16000),
    ("Adam_b8", "Adam",         {"beta1":0.9,"beta2":0.99},  0.001, 20000),
    ("SGDM_b8", "SGD-Momentum", {"beta":0.9},                0.002, 14000),
]
res = []
for tag, optn, params, lr, steps in cells:
    print(f"\n=== {tag} ===", flush=True)
    st = PR.train_plateau(optn, params, 8, lr, steps)
    if st is None: print(f"  {tag}: diverged"); res.append({"tag":tag,"diverged":True}); continue
    dinv = st['opt'].get_preconditioner_inv_sqrt() if optn == "Adam" else None
    X, Y = st['XY']
    dirs = [("u_hess", st['u_hess']), ("u_step", st['u_step'])]
    if optn == "Adam" and dinv is not None:
        dirs.append(("u_precond", precond_top_eigvec(st['net'], st['loss_fn'], X, Y, dinv)))
    # natural kick scale from un-kicked projection std
    p0, _, _ = relax_geom(st, st['u_hess'], 0.0, dinv, seed=1)
    scale = float(np.std(p0)) + 1e-9
    for uname, u in dirs:
        p_ref, f_ref, pr_ref = relax_geom(st, u, 0.0, dinv)
        row = {"tag":tag, "kick_dir":uname}
        for amul in (0.5, 1.0, 2.0):   # small-amplitude LINEAR regime (below escape)
            proj, full, pre = relax_geom(st, u, amul * scale, dinv)
            gp = gamma_of(np.abs(proj - p_ref)); gf = gamma_of(np.abs(full - f_ref)); gpre = gamma_of(np.abs(pre - pr_ref))
            row[f"a{amul}"] = {"proj":gp, "full":gf, "precond":gpre}
            print(f"  [{uname:9s}] amp={amul}x: proj={gp:+.4f} ||d||={gf:+.4f} ||P.5 d||={gpre:+.4f}", flush=True)
        res.append(row)
        json.dump(res, open(os.path.join(OUT, 'adam_causal_v2.json'), 'w'), indent=2)
print("\ndone -- see per-amplitude gamma above; discriminate at smallest amp (linear regime).")
