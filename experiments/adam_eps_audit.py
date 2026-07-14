"""Adjudicator eps-robustness audit for the Adam b2048 kappa_raw~34 (preconditioned) number:
replay Adam b2048 s0 to a plateau state, then kappa~ = lr * top-eig(d H d) across eps variants
(adam 1e-8, rel 0.01/0.1/1.0/10 x median) on 6 fresh batches each. The quoted kappa_spec is
less exposed (gain and lambda share the frame) but the raw number must pass this before print."""
import os, sys, json
import numpy as np, torch as T
sys.path.insert(0, ".")
os.environ.setdefault("DATASETS", "/Users/xq/Desktop/moonshot/eoss/datasets")
os.environ.setdefault("EOSS_SKIP_CHECKSUM", "1")
import torchvision.datasets.cifar as _c; _c.check_integrity = lambda *a, **k: True
import experiments.long_train_grid as L
from experiments.adam_adjudicator import sqrt_vhat_flat
from utils.optimizer import create_optimizer
from utils.measure import create_hessian_vector_product, flatt
T.set_num_threads(4)

m = json.load(open("results/kspec/L_adam_b2048_s0/meta.json"))
T.manual_seed(m["seed"]); X, Y = L.get_data(); net, loss_fn = L.build()
opt = create_optimizer("Adam", net, m["lr"], {"beta1": 0.9, "beta2": 0.99})
params = [p for p in net.parameters() if p.requires_grad]
gen = T.Generator().manual_seed(1000 + m["seed"])
for step in range(5000):
    idx = T.randperm(len(X), generator=gen)[:m["batch"]]
    lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
    opt.zero_grad(); lo.backward(); opt.step()
print(f"replayed to 5000, loss={float(lo):.4g}", flush=True)
sqv = sqrt_vhat_flat(opt); med = float(sqv.median())
print(f"sqrt(vhat): median={med:.3e} frac<1e-6={float((sqv<1e-6).float().mean()):.3f} "
      f"frac<1e-8={float((sqv<1e-8).float().mean()):.3f}")
for name, eps in [("adam_1e-8", 1e-8), ("rel_0.01", 0.01*med), ("rel_0.1", 0.1*med),
                  ("rel_1.0", med), ("rel_10", 10*med)]:
    d = (1.0 / (sqv + eps).sqrt()).detach()
    ks = []
    g2 = T.Generator().manual_seed(4321)
    for _ in range(6):
        idx = T.randperm(len(X), generator=g2)[:m["batch"]]
        lo = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        grads = T.autograd.grad(lo, params, create_graph=True)
        gf = flatt(grads)
        hvp = create_hessian_vector_product(lo, net, params=params, grads=grads, flat_grads=gf)
        try:
            u = (d * gf.detach()); u /= u.norm() + 1e-30
            for _ in range(14):
                Hu = d * hvp(d * u, retain_graph_override=True)
                n = float(Hu.norm())
                if n < 1e-20: break
                u = (Hu / n).detach()
            lam = float(u @ (d * hvp(d * u, retain_graph_override=True)))
        finally:
            hvp.free_memory()
        ks.append(m["lr"] * lam)
    print(f"  {name:10s} kappa~ = {np.mean(ks):8.3f} +/- {np.std(ks):.3f}", flush=True)
