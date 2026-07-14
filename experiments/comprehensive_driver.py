"""
Driver for the comprehensive marginal-vs-metastable sweep.

Grid: SGD-Momentum, mlp_s/2048/MSE.
  batch  B    in {8, 32, 128, 512, 2048}         (5)
  beta        in {0, 0.3, 0.6, 0.9, 0.95, 0.99}  (6; beta=0 = plain-SGD marginal anchor)
  lr_index    0..3  (4 lrs per (B,beta), chosen below)
= 120 grid points. Each is a full cell (train + 3 causal tests) run by comprehensive_sweep.py.

LR SELECTION (per (B,beta), cached in lr_plan.json):
  model lr_hi(B,beta) = lr_ref(beta) * (B/8)^0.19, anchored on results/calib2/FINAL_GRID.json
  (SGDM b8/b128/b2048 = 0.002/0.003/0.006; beta-sweep b8 = 0.006/0.005/0.004/0.002/-/0.0001).
  Then a quick empirical pre-check trains ~1500 steps at lr_hi and DROPS lr x0.6 until it no
  longer diverges (loss>1e6/NaN). The 4 lrs = geomspace(lr_top*0.22, lr_top, 4) -- clearly
  sub-edge up to just-below-divergence. Logged per cell.

EXECUTION: one long-running process that maintains a pool of <=N concurrent worker subprocesses
(each torch.set_num_threads(EOSS_THREADS)), skips already-done cells (meta.json present),
updates results/comprehensive_sweep/manifest.json after every event. RESUMABLE: re-run to
continue. Core betas {0.3..0.99} x 5 batches (100 cells) are ordered FIRST; the beta=0 anchor
row (20 cells) last, so one complete core set lands before the anchors.

Usage:
  python -m experiments.comprehensive_driver --plan-only            # build lr_plan.json then exit
  python -m experiments.comprehensive_driver --concurrency 3        # build plan (if needed) + run pool
  python -m experiments.comprehensive_driver --cells b8_beta0.9_lr3 b2048_beta0_lr3   # run listed cells only
"""
import os, sys, json, time, argparse, subprocess
import numpy as np
import torch as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
if os.environ.get('EOSS_SKIP_CHECKSUM'):
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.check_integrity = lambda *a, **k: True

OUT_ROOT = os.path.join(_REPO, 'results', 'comprehensive_sweep')
os.makedirs(OUT_ROOT, exist_ok=True)
PLAN_PATH = os.path.join(OUT_ROOT, 'lr_plan.json')
MANIFEST_PATH = os.path.join(OUT_ROOT, 'manifest.json')

BATCHES = [8, 32, 128, 512, 2048]
# core betas first, beta=0 anchor row last -> one complete core set before anchors
BETAS_CORE = [0.3, 0.6, 0.9, 0.95, 0.99]
BETAS_ANCHOR = [0.0]
LR_REF = {0.0: 0.006, 0.3: 0.005, 0.6: 0.004, 0.9: 0.002, 0.95: 0.0008, 0.99: 0.0001}


def beta_tag(beta):
    return f"{beta:g}"


def cell_tag(B, beta, idx):
    return f"b{B}_beta{beta_tag(beta)}_lr{idx}"


def lr_hi_model(B, beta):
    return LR_REF[beta] * (B / 8.0) ** 0.19


# ------------------------------------------------------------------ lr planning (pre-check)
def quick_diverges(beta, batch, lr, steps=1500):
    import experiments.long_train_grid as L
    from utils.optimizer import create_optimizer
    X, Y = L.get_data(); net, loss_fn = L.build()
    opt = create_optimizer('SGD-Momentum', net, lr, {'beta': beta})
    for s in range(steps):
        idx = T.randperm(len(X))[:batch]; Xb, Yb = X[idx], Y[idx]
        pr = net(Xb).squeeze(-1); lo = loss_fn(pr, Yb)
        lv = lo.item()
        if not np.isfinite(lv) or lv > 1e6:
            return True
        opt.zero_grad(); lo.backward(); opt.step()
    return False


def build_plan(betas):
    T.set_num_threads(int(os.environ.get('EOSS_THREADS', '4')))
    plan = {}
    if os.path.exists(PLAN_PATH):
        with open(PLAN_PATH) as f:
            plan = json.load(f)
    for beta in betas:
        for B in BATCHES:
            key = f"b{B}_beta{beta_tag(beta)}"
            if key in plan:
                continue
            lr_top = lr_hi_model(B, beta)
            tries = []
            for _ in range(6):
                div = quick_diverges(beta, B, lr_top)
                tries.append(dict(lr=lr_top, diverged=div))
                if div:
                    lr_top *= 0.6
                else:
                    break
            lrs = list(np.geomspace(lr_top * 0.22, lr_top, 4))
            plan[key] = dict(B=B, beta=beta, lr_top=float(lr_top), lrs=[float(x) for x in lrs],
                             precheck=tries)
            with open(PLAN_PATH, 'w') as f:
                json.dump(plan, f, indent=2)
            print(f"[plan] {key}: lr_top={lr_top:.5g} lrs={[round(x,6) for x in lrs]} "
                  f"({len(tries)} precheck tries)", flush=True)
    return plan


# ------------------------------------------------------------------ cell list + manifest
def all_cells(betas):
    cells = []
    for beta in betas:            # ordered: caller passes core first
        for B in BATCHES:
            for idx in range(4):
                cells.append((B, beta, idx))
    return cells


def cell_done(tag):
    return os.path.exists(os.path.join(OUT_ROOT, tag, 'meta.json'))


def write_manifest(betas, plan):
    entries = []
    for B, beta, idx in all_cells(betas):
        tag = cell_tag(B, beta, idx)
        key = f"b{B}_beta{beta_tag(beta)}"
        lr = plan.get(key, {}).get('lrs', [None] * 4)[idx] if key in plan else None
        done = cell_done(tag)
        status = None
        if done:
            try:
                with open(os.path.join(OUT_ROOT, tag, 'meta.json')) as f:
                    status = json.load(f).get('status')
            except Exception:
                status = 'unknown'
        entries.append(dict(tag=tag, B=B, beta=beta, lr_index=idx, lr=lr, done=done, status=status))
    manifest = dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                    total=len(entries), done=sum(e['done'] for e in entries), cells=entries)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ------------------------------------------------------------------ concurrent pool
def run_pool(betas, plan, concurrency, only_tags=None):
    cells = all_cells(betas)
    todo = []
    for B, beta, idx in cells:
        tag = cell_tag(B, beta, idx)
        if only_tags and tag not in only_tags:
            continue
        if cell_done(tag):
            continue
        key = f"b{B}_beta{beta_tag(beta)}"
        if key not in plan:
            print(f"[pool] no plan for {key}, skipping {tag}", flush=True); continue
        lr = plan[key]['lrs'][idx]
        todo.append((B, beta, idx, tag, lr))
    print(f"[pool] {len(todo)} cells to run, concurrency={concurrency}", flush=True)

    running = {}  # proc -> (tag, logf)
    py = sys.executable
    env = dict(os.environ)
    env.setdefault('EOSS_THREADS', '4')
    i = 0

    def launch(job):
        B, beta, idx, tag, lr = job
        logpath = os.path.join(OUT_ROOT, tag, 'run.log')
        os.makedirs(os.path.join(OUT_ROOT, tag), exist_ok=True)
        logf = open(logpath, 'w')
        cmd = [py, '-m', 'experiments.comprehensive_sweep', '--B', str(B), '--beta', str(beta),
               '--lr', repr(lr), '--lr_index', str(idx), '--tag', tag]
        p = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[p] = (tag, logf)
        print(f"[pool] launched {tag} (lr={lr:.5g}) pid={p.pid}", flush=True)

    while i < len(todo) or running:
        while i < len(todo) and len(running) < concurrency:
            launch(todo[i]); i += 1
        time.sleep(3)
        for p in list(running):
            if p.poll() is not None:
                tag, logf = running.pop(p); logf.close()
                st = 'ok' if p.returncode == 0 else f'rc={p.returncode}'
                print(f"[pool] finished {tag} ({st})", flush=True)
                write_manifest(betas, plan)
    m = write_manifest(betas, plan)
    print(f"[pool] ALL DONE: {m['done']}/{m['total']} cells", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--include-anchor', action='store_true', default=True,
                    help='include beta=0 anchor row (default on)')
    ap.add_argument('--cells', nargs='*', default=None, help='run only these cell tags')
    a = ap.parse_args()
    betas = BETAS_CORE + BETAS_ANCHOR  # core first, anchor last
    plan = build_plan(betas)
    write_manifest(betas, plan)
    if a.plan_only:
        print("[driver] plan-only done", flush=True); return
    run_pool(betas, plan, a.concurrency, only_tags=set(a.cells) if a.cells else None)


if __name__ == '__main__':
    main()
