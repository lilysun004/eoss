"""
Experiment (3): coupled/decoupled step decomposition -- the EXACT Mechanism-A test.

For SGD-Momentum (torch convention, dampening=0):
    buf_t = beta*buf_{t-1} + g_t ;  s = -lr*buf_t
so the step splits exactly into
    s_c = -lr*g_t             (CURRENT batch -- this is literally the SGD step)
    s_d = -lr*beta*buf_{t-1}  (HISTORY -- independent of the current batch B_t)

GBS numerator B = s^T H_B s splits as
    B = B_cc + 2 B_cd + B_dd,   B_cc=s_c^T H_B s_c, B_cd=s_c^T H_B s_d, B_dd=s_d^T H_B s_d
GBS denominator A = -g_B^T s splits exactly (bilinear) as A = A_c + A_d.

Mechanism A predicts: the coupled block ratio GBS_cc = B_cc/(-A_c) should be ~2 for
EVERY batch size (because s_c=-lr*g_t is exactly the SGD step, whose curvature ratio
is lr*batch_sharpness ~ 2 at the edge), while the history/cross blocks -- which pair a
batch-INDEPENDENT direction s_d with the current batch's Hessian -- carry the deficit,
growing with beta and with small batch (more noise decoheres the buffer).

We report, per (batch, beta), averaged over probe batches:
  - GBS_total, GBS_cc (coupled), GBS_dd (history), and the cross contribution,
  - the energy fractions ||s_c||^2 / ||s||^2 etc,
  - cos(s_c, s_d) and cos(g_t, buf).
CPU MLP, uses FINAL_GRID lr's. No config.py, standalone.
"""
import os, sys, time, json
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
from utils.measure import create_hessian_vector_product, flatt

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
OUT_DIR = os.path.join(_REPO, 'results', 'coupled_decoupled')
os.makedirs(OUT_DIR, exist_ok=True)
_DATA = {}


def get_data():
    if 'xy' not in _DATA:
        X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, 2048, [], 888, loss_type='mse')
        _DATA['xy'] = (X, Y)
    return _DATA['xy']


def build():
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets['mlp_s']['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets['mlp_s']['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    return net, SquaredLoss()


def momentum_buffer_flat(opt_inner, params):
    """Read buf_{t-1} (the momentum buffer BEFORE this step) as a flat vector."""
    pieces = []
    for p in params:
        st = opt_inner.state.get(p)
        if st and 'momentum_buffer' in st and st['momentum_buffer'] is not None:
            pieces.append(st['momentum_buffer'].flatten())
        else:
            pieces.append(T.zeros(p.numel()))
    return T.cat(pieces).detach()


def probe(net, X, Y, loss_fn, lr, beta, batch_size, n_probe=16):
    params = [p for p in net.parameters() if p.requires_grad]
    rows = []
    N = len(X)
    for _ in range(n_probe):
        idx = T.randperm(N)[:batch_size]
        Xb, Yb = X[idx], Y[idx]
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        grads = T.autograd.grad(loss, params, create_graph=True)
        g = flatt(grads)
        gd = g.detach()
        buf_prev = momentum_buffer_flat(net.__opt_inner, params)   # buf_{t-1}
        s_c = (-lr * gd)
        s_d = (-lr * beta * buf_prev)
        s = s_c + s_d
        hvp = create_hessian_vector_product(loss, net, params=params, grads=grads, flat_grads=g)
        try:
            Hs_c = hvp(s_c, retain_graph_override=True)
            Hs_d = hvp(s_d, retain_graph_override=False)
        finally:
            hvp.free_memory()
        B_cc = T.dot(s_c, Hs_c).item()
        B_cd = T.dot(s_c, Hs_d).item()
        B_dd = T.dot(s_d, Hs_d).item()
        B_tot = B_cc + 2 * B_cd + B_dd
        A_c = -T.dot(gd, s_c).item()
        A_d = -T.dot(gd, s_d).item()
        A_tot = A_c + A_d
        def r(b, a):
            return b / a if abs(a) > 1e-15 else float('nan')
        rows.append(dict(
            GBS_tot=r(B_tot, A_tot), GBS_cc=r(B_cc, A_c), GBS_dd=r(B_dd, A_d),
            B_cc=B_cc, B_cd=B_cd, B_dd=B_dd, B_tot=B_tot, A_c=A_c, A_d=A_d, A_tot=A_tot,
            sc_frac=(T.dot(s_c, s_c) / (T.dot(s, s) + 1e-30)).item(),
            cos_sc_sd=(T.dot(s_c, s_d) / (s_c.norm() * s_d.norm() + 1e-30)).item(),
            cos_g_buf=(T.dot(gd, buf_prev) / (gd.norm() * buf_prev.norm() + 1e-30)).item(),
        ))
    return rows


def run_cell(beta, batch_size, lr, steps, n_probe=16, tag=None):
    X, Y = get_data()
    net, loss_fn = build()
    opt = create_optimizer('SGD-Momentum', net, lr, {'beta': beta})
    net.__opt_inner = opt.inner
    t0 = time.time()
    for step in range(steps):
        idx = T.randperm(len(X))[:batch_size]
        Xb, Yb = X[idx], Y[idx]
        preds = net(Xb).squeeze(-1)
        loss = loss_fn(preds, Yb)
        lv = loss.item()
        if not np.isfinite(lv) or lv > 1e6:
            return dict(beta=beta, batch=batch_size, lr=lr, diverged=True, tag=tag)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"  [{tag}] step {step}/{steps} loss={lv:.4f}", flush=True)
    rows = probe(net, X, Y, loss_fn, lr, beta, batch_size, n_probe=n_probe)
    def m(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else float('nan')
    # inside-placement block ratios (mean B / mean(-A))
    def inside(bk, ak):
        bs = [r[bk] for r in rows]; as_ = [r[ak] for r in rows]
        return float(np.mean(bs) / -np.mean([-a for a in as_])) if np.mean(as_) != 0 else float('nan')
    agg = dict(
        beta=beta, batch=batch_size, lr=lr, diverged=False, tag=tag, n=len(rows), wall=time.time() - t0,
        GBS_tot=m('GBS_tot'), GBS_cc=m('GBS_cc'), GBS_dd=m('GBS_dd'),
        GBS_cc_inside=(float(np.mean([r['B_cc'] for r in rows]) / np.mean([r['A_c'] for r in rows]))),
        GBS_tot_inside=(float(np.mean([r['B_tot'] for r in rows]) / np.mean([r['A_tot'] for r in rows]))),
        cross_contrib=float(np.mean([2 * r['B_cd'] for r in rows])),
        B_tot_mean=float(np.mean([r['B_tot'] for r in rows])),
        sc_frac=m('sc_frac'), cos_sc_sd=m('cos_sc_sd'), cos_g_buf=m('cos_g_buf'),
    )
    return agg


def main():
    # (tag, beta, batch, lr, steps) -- lr from FINAL_GRID; small batch is where the deficit lives
    cells = [
        ("b8_beta0.9",    0.9,  8,    0.002, 3500),
        ("b128_beta0.9",  0.9,  128,  0.003, 2600),
        ("b2048_beta0.9", 0.9,  2048, 0.006, 4000),
        ("b8_beta0.3",    0.3,  8,    0.005, 3500),
        ("b8_beta0.6",    0.6,  8,    0.004, 3500),
    ]
    results = []
    for tag, beta, batch, lr, steps in cells:
        print(f"\n=== {tag}: beta={beta} b={batch} lr={lr} steps={steps} ===", flush=True)
        agg = run_cell(beta, batch, lr, steps, tag=tag)
        results.append(agg)
        if agg['diverged']:
            print(f"  {tag}: DIVERGED", flush=True); continue
        print(f"  {tag}: GBS_tot={agg['GBS_tot']:.3f}  GBS_cc(coupled)={agg['GBS_cc']:.3f}  "
              f"GBS_dd(history)={agg['GBS_dd']:.3f}  cross={agg['cross_contrib']:.4g}  "
              f"sc_frac={agg['sc_frac']:.3f}  cos(g,buf)={agg['cos_g_buf']:.3f}", flush=True)
    with open(os.path.join(OUT_DIR, 'coupled_decoupled.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n===== SUMMARY =====")
    print(f"{'cell':16s} {'GBS_tot':>8s} {'GBS_cc':>8s} {'GBS_dd':>8s} {'cross':>10s} {'sc_frac':>8s} {'cos(g,buf)':>10s}")
    for a in results:
        if a['diverged']:
            print(f"{a['tag']:16s}  DIVERGED"); continue
        print(f"{a['tag']:16s} {a['GBS_tot']:8.3f} {a['GBS_cc']:8.3f} {a['GBS_dd']:8.3f} "
              f"{a['cross_contrib']:10.4g} {a['sc_frac']:8.3f} {a['cos_g_buf']:10.3f}")
    print(f"\nwrote {OUT_DIR}/coupled_decoupled.json")


if __name__ == '__main__':
    main()
