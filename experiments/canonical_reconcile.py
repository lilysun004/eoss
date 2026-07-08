"""
Validity gate: is SGD b8's GBS<2 a lean-setup artifact (suppressed gradient/
eigenvector alignment) or intrinsic? Track, OVER TRAINING, the two factors of
the SGD decomposition GBS = alpha_g * kappa separately:
    kappa   = eta * E[lambda_max(H_B)]         (batch edge parameter, ~2 at EoS)
    alpha_g = E[ (g^T H_B g / g^T g) / lambda_max(H_B) ]   (gradient spectral alignment)
    cos_gu  = E[ |cos(g, u_B)| ]               (direct alignment of grad with top eigvec)
and GBS itself, for a canonical-scale cell (mlp, num_data=8192) vs the lean cell
(mlp_s, num_data=2048). If alpha_g keeps climbing toward ~1 with more steps/capacity
at canonical scale (GBS->2), the lean deficit was curtailed alignment; if it
plateaus low like lean, the small-batch deficit is intrinsic.

Usage:
    python experiments/canonical_reconcile.py mlp   8192 8 0.02 20000
    python experiments/canonical_reconcile.py mlp_s 2048 8 0.01 12000
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
from utils.measure import compute_eigenvalues, EigenvectorCache, create_hessian_vector_product, flatt

T.set_num_threads(4)
DATASET_FOLDER = os.environ['DATASETS']
OUT_DIR = os.path.join(_REPO, 'results', 'canonical_reconcile')
os.makedirs(OUT_DIR, exist_ok=True)


def build(model, num_data):
    X, Y, _, _ = prepare_dataset('cifar10', DATASET_FOLDER, num_data, [], 888, loss_type='mse')
    presets = get_model_presets(); ds = get_dataset_presets()
    mp = dict(presets[model]['params'])
    mp['input_dim'] = ds['cifar10']['input_dim']; mp['output_dim'] = ds['cifar10']['output_dim']
    net = prepare_net(model_type=presets[model]['type'], params=mp)
    initialize_net(net, scale=0.2, seed=8888)
    return net, SquaredLoss(), X, Y


def probe(net, X, Y, loss_fn, opt, lr, batch, n_probe=16):
    """Per-batch: R_g=g^T H_B g/g^T g, lambda_max(H_B), GBS_t, |cos(g,u_B)|."""
    params = [p for p in net.parameters() if p.requires_grad]
    cache = EigenvectorCache(1)
    Rg, Lam, Gbs, Cos = [], [], [], []
    N = len(X)
    for _ in range(n_probe):
        idx = T.randperm(N)[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        grads = T.autograd.grad(lo, params, create_graph=True)
        g = flatt(grads); gd = g.detach()
        s = opt.compute_step_direction(g, params).detach()
        hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=g)
        try:
            Hg = hvp(gd, retain_graph_override=True)
            Hs = hvp(s, retain_graph_override=False)
            rg = (T.dot(gd, Hg) / (T.dot(gd, gd) + 1e-30)).item()
            gbs = (T.dot(s, Hs) / (-T.dot(gd, s) + 1e-30)).item()
        finally:
            hvp.free_memory()
        # top eigvec of H_B for alignment
        pr2 = net(Xb).squeeze(-1); lo2 = loss_fn(pr2, Yb)
        try:
            lam, u = compute_eigenvalues(lo2, net, k=1, max_iterations=40, reltol=0.02,
                                         eigenvector_cache=cache, return_eigenvectors=True,
                                         use_power_iteration=False)
            lam = lam.item(); u = u.detach()
            cos = abs((T.dot(gd, u) / (gd.norm() + 1e-30)).item())
        except Exception:
            lam, cos = float('nan'), float('nan')
        if np.isfinite(rg) and np.isfinite(lam) and lam > 0:
            Rg.append(rg); Lam.append(lam); Gbs.append(gbs); Cos.append(cos)
    Rg, Lam, Gbs, Cos = map(np.array, (Rg, Lam, Gbs, Cos))
    return dict(
        Rg=float(Rg.mean()), lam=float(Lam.mean()), gbs=float(Gbs.mean()),
        kappa=float(lr * Lam.mean()), alpha_g=float(np.mean(Rg / Lam)),
        cos_gu=float(Cos.mean()), n=len(Rg),
    )


def main():
    model = sys.argv[1]; num_data = int(sys.argv[2]); batch = int(sys.argv[3])
    lr = float(sys.argv[4]); steps = int(sys.argv[5])
    measure_every = max(500, steps // 25)
    tag = f"{model}_nd{num_data}_b{batch}_lr{lr}"
    print(f"=== {tag} steps={steps} measure_every={measure_every} ===", flush=True)
    net, loss_fn, X, Y = build(model, num_data)
    opt = create_optimizer('SGD', net, lr, {})
    traj = []
    t0 = time.time()
    for step in range(steps):
        idx = T.randperm(len(X))[:batch]
        Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item()
        if not np.isfinite(lv) or lv > 1e6:
            print(f"  DIVERGED @ {step}", flush=True); break
        opt.zero_grad(); lo.backward(); opt.step()
        if step % measure_every == 0 and step > 0:
            m = probe(net, X, Y, loss_fn, opt, lr, batch)
            m['step'] = step; m['loss'] = lv
            traj.append(m)
            print(f"  step {step:6d} loss={lv:.4f}  GBS={m['gbs']:.3f}  kappa={m['kappa']:.3f}  "
                  f"alpha_g={m['alpha_g']:.3f}  cos(g,u)={m['cos_gu']:.3f}  lam={m['lam']:.1f}",
                  flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT_DIR, f"{tag}.json"), 'w') as f:
        json.dump(dict(model=model, num_data=num_data, batch=batch, lr=lr, steps=steps, traj=traj), f, indent=2)
    if traj:
        tail = traj[-min(5, len(traj)):]
        print(f"\n  TAIL MEAN ({tag}): GBS={np.mean([t['gbs'] for t in tail]):.3f}  "
              f"kappa={np.mean([t['kappa'] for t in tail]):.3f}  "
              f"alpha_g={np.mean([t['alpha_g'] for t in tail]):.3f}  "
              f"cos(g,u)={np.mean([t['cos_gu'] for t in tail]):.3f}", flush=True)


if __name__ == '__main__':
    main()
