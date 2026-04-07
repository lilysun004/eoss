"""
Run multiple experiments across GPUs.

Edit the `jobs` list below. Each dict contains only the overrides —
everything else comes from config.py defaults.

Usage: python config_loop.py
"""

from utils.job_runner import run_jobs

# =============================================
# ORIGINAL FULL LOOP (commented out)
# =============================================
BATCH_SIZES = [4096,8192]#[2048,1024,512,256,128,64,32,16,8,4]
#
# # (optimizer_name, [lr1, lr2, lr3], optimizer_params)
OPTIMIZERS = [
    ('Adam',         [1e-3],        {'beta1': 0.9, 'beta2': 0.999}),
    ('Muon',         [0.002],        {'momentum': 0.95}),  # reduced from [0.02, 0.05, 0.1] — old LRs diverged
    ('SGD',          [0.007],        {}),
    ('SGD-Momentum', [0.004],        {'beta': 0.9}),
    ('SGD-Nesterov', [0.004],        {'beta': 0.9}),
]

jobs = []
for batch_size in BATCH_SIZES:
    for opt_name, lrs, opt_params in OPTIMIZERS:
        for lr in lrs:
            job = {'batch_size': batch_size, 'optimizer_name': opt_name, 'lr': lr}
            if opt_params:
                job['optimizer_params'] = opt_params
            jobs.append(job)

# =============================================
# TARGETED RERUN (space-killed jobs)
# Missing: b8 SGD-Nesterov lr=0.008
#          b64 SGD (all), SGD-Momentum (all), SGD-Nesterov (all), Muon lr=0.1
#          b256 all
#          b1024 all (several never started)
# =============================================

# jobs = [
#     # --- b1024 ---
#     {'batch_size': 1024, 'optimizer_name': 'Adam',         'lr': 3e-4,  'optimizer_params': {'beta1': 0.9, 'beta2': 0.999}},
#     {'batch_size': 1024, 'optimizer_name': 'Adam',         'lr': 1e-3,  'optimizer_params': {'beta1': 0.9, 'beta2': 0.999}},
#     {'batch_size': 1024, 'optimizer_name': 'Adam',         'lr': 3e-3,  'optimizer_params': {'beta1': 0.9, 'beta2': 0.999}},
#     {'batch_size': 1024, 'optimizer_name': 'Muon',         'lr': 0.002, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 1024, 'optimizer_name': 'Muon',         'lr': 0.005, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 1024, 'optimizer_name': 'Muon',         'lr': 0.01,  'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 1024, 'optimizer_name': 'SGD',          'lr': 0.007},
#     {'batch_size': 1024, 'optimizer_name': 'SGD',          'lr': 0.015},
#     {'batch_size': 1024, 'optimizer_name': 'SGD',          'lr': 0.04},
#     {'batch_size': 1024, 'optimizer_name': 'SGD-Momentum', 'lr': 0.004, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 1024, 'optimizer_name': 'SGD-Momentum', 'lr': 0.008, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 1024, 'optimizer_name': 'SGD-Momentum', 'lr': 0.02,  'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 1024, 'optimizer_name': 'SGD-Nesterov', 'lr': 0.004, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 1024, 'optimizer_name': 'SGD-Nesterov', 'lr': 0.008, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 1024, 'optimizer_name': 'SGD-Nesterov', 'lr': 0.02,  'optimizer_params': {'beta': 0.9}},

#     # --- b8 (Muon rerun with reduced LRs) ---
#     {'batch_size': 8,    'optimizer_name': 'Muon',         'lr': 0.002, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 8,    'optimizer_name': 'Muon',         'lr': 0.005, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 8,    'optimizer_name': 'Muon',         'lr': 0.01,  'optimizer_params': {'momentum': 0.95}},

#     # --- b64 (Muon rerun with reduced LRs) ---
#     {'batch_size': 64,   'optimizer_name': 'Muon',         'lr': 0.002, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 64,   'optimizer_name': 'Muon',         'lr': 0.005, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 64,   'optimizer_name': 'Muon',         'lr': 0.01,  'optimizer_params': {'momentum': 0.95}},

#     # --- b256 ---
#     {'batch_size': 256,  'optimizer_name': 'Adam',         'lr': 3e-4,  'optimizer_params': {'beta1': 0.9, 'beta2': 0.999}},
#     {'batch_size': 256,  'optimizer_name': 'Adam',         'lr': 1e-3,  'optimizer_params': {'beta1': 0.9, 'beta2': 0.999}},
#     {'batch_size': 256,  'optimizer_name': 'Adam',         'lr': 3e-3,  'optimizer_params': {'beta1': 0.9, 'beta2': 0.999}},
#     {'batch_size': 256,  'optimizer_name': 'Muon',         'lr': 0.002, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 256,  'optimizer_name': 'Muon',         'lr': 0.005, 'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 256,  'optimizer_name': 'Muon',         'lr': 0.01,  'optimizer_params': {'momentum': 0.95}},
#     {'batch_size': 256,  'optimizer_name': 'SGD',          'lr': 0.007},
#     {'batch_size': 256,  'optimizer_name': 'SGD',          'lr': 0.015},
#     {'batch_size': 256,  'optimizer_name': 'SGD',          'lr': 0.04},
#     {'batch_size': 256,  'optimizer_name': 'SGD-Momentum', 'lr': 0.004, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 256,  'optimizer_name': 'SGD-Momentum', 'lr': 0.008, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 256,  'optimizer_name': 'SGD-Momentum', 'lr': 0.02,  'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 256,  'optimizer_name': 'SGD-Nesterov', 'lr': 0.004, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 256,  'optimizer_name': 'SGD-Nesterov', 'lr': 0.008, 'optimizer_params': {'beta': 0.9}},
#     {'batch_size': 256,  'optimizer_name': 'SGD-Nesterov', 'lr': 0.02,  'optimizer_params': {'beta': 0.9}},

# ]

print(f'=============== Running {len(jobs)} jobs ===============')
if __name__ == '__main__':
    run_jobs(jobs, num_gpus=2, max_per_gpu=2)
