"""
Driver: run all config.py cells needed for
  (A) the momentum/Adam beta-sweep, and
  (C) the GBS_t distribution grid (12 base cells),
in ONE sequential pass, with cheap measurement settings (no per-batch power
iteration, no batch_sharpness) so each cell is fast. All lr/params/steps come
from results/calib2/FINAL_GRID.json (already validated -- no retuning).

Each cell -> results/gbs_grid_v3/<auto run folder>/results.txt with a per-step
GBS column. Analysis scripts (gbs_dist_v3.py, beta_sweep_v3.py) read from there.
"""
import os, sys, subprocess, time, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = 'gbs_grid_v3'

# (tag, optimizer, params, batch, lr, calib_steps)
CELLS = [
    # --- base grid (Mechanism C distributions) ---
    ("SGD_b8",       "SGD",          {},                          8,    0.01,  2500),
    ("SGD_b128",     "SGD",          {},                          128,  0.02,  1800),
    ("SGD_b2048",    "SGD",          {},                          2048, 0.02,  1200),
    ("SGDM09_b8",    "SGD-Momentum", {"beta": 0.9},               8,    0.002, 3000),
    ("SGDM09_b128",  "SGD-Momentum", {"beta": 0.9},               128,  0.003, 2200),
    ("SGDM09_b2048", "SGD-Momentum", {"beta": 0.9},               2048, 0.006, 3500),
    ("Adam_b8",      "Adam",         {"beta1": 0.9, "beta2": 0.99},8,   0.001, 3000),
    ("Adam_b128",    "Adam",         {"beta1": 0.9, "beta2": 0.99},128, 0.001, 2200),
    ("Adam_b2048",   "Adam",         {"beta1": 0.9, "beta2": 0.99},2048,0.001, 1500),
    ("Muon_b8",      "Muon",         {"momentum": 0.9},           8,    0.003, 3000),
    ("Muon_b128",    "Muon",         {"momentum": 0.9},           128,  0.003, 5000),
    ("Muon_b2048",   "Muon",         {"momentum": 0.9},           2048, 0.003, 6000),
    # --- extra beta-sweep cells (Mechanism A), SGD-Momentum b8 ---
    ("SGDM_b0_b8",   "SGD-Momentum", {"beta": 0.0},               8,    0.006, 3000),
    ("SGDM_b03_b8",  "SGD-Momentum", {"beta": 0.3},               8,    0.005, 3000),
    ("SGDM_b06_b8",  "SGD-Momentum", {"beta": 0.6},               8,    0.004, 3000),
    ("SGDM_b099_b8", "SGD-Momentum", {"beta": 0.99},              8,    0.0001,6000),
    # --- extra Adam beta1-sweep cells (Mechanism A bonus), b8, beta2=0.99 ---
    ("AdamB1_0_b8",  "Adam",         {"beta1": 0.0,  "beta2": 0.99},8,  0.001, 3000),
    ("AdamB1_05_b8", "Adam",         {"beta1": 0.5,  "beta2": 0.99},8,  0.001, 3000),
    ("AdamB1_099_b8","Adam",         {"beta1": 0.99, "beta2": 0.99},8,  0.0003,5000),
]


def main():
    env = dict(os.environ)
    env.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
    env.setdefault('EOSS_SKIP_CHECKSUM', '1')
    env['RESULTS'] = os.path.join(_REPO, 'results')

    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for tag, opt, params, batch, lr, csteps in CELLS:
        if only and tag not in only:
            continue
        steps = int(round(1.3 * csteps))
        probe_every = 25
        cmd = [
            sys.executable, os.path.join(_REPO, 'config.py'),
            '--model', 'mlp_s',
            '--optimizer_name', opt,
            '--optimizer_params', json.dumps(params).replace('"', "'"),
            '--batch_size', str(batch),
            '--lr', str(lr),
            '--num_data', '2048',
            '--steps', str(steps),
            '--compute_probe_batch_every', str(probe_every),
            '--probe_samples', '12',
            '--compute_quantities_with_uB', 'False',
            '--gpu', 'cpu',
            '--results_subfolder', f'{SUB}/{tag}',
        ]
        t0 = time.time()
        print(f"\n=== {tag}: {opt} b={batch} lr={lr} steps={steps} params={params} ===", flush=True)
        r = subprocess.run(cmd, env=env, cwd=_REPO,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        dt = time.time() - t0
        ok = (r.returncode == 0)
        tail = (r.stderr or '').strip().splitlines()[-1] if r.stderr else ''
        print(f"    {'OK' if ok else 'FAIL rc=%d' % r.returncode} in {dt:.0f}s   {tail[:120]}", flush=True)


if __name__ == '__main__':
    main()
