"""
Run multiple jacobian experiments across GPUs.

Edit the `jobs` list below. Each dict contains only the overrides —
everything else comes from jacobian/config.py defaults.

Usage: python jacobian/config_loop.py
   or: cd jacobian && python config_loop.py
"""

from utils.job_runner import run_jobs

# =============================================
# DEFINE YOUR JOBS HERE (only overrides)
# =============================================

jobs = []
jobs.append({'optimizer_name': 'SGD',          'lr': 0.005, 'batch_size': 64})
jobs.append({'optimizer_name': 'SGD-Momentum',  'lr': 0.007, 'optimizer_params': {'beta': 0.9}, 'batch_size': 64})

print(f'=============== Running {len(jobs)} jobs ===============')
if __name__ == '__main__':
    run_jobs(jobs, num_gpus=2, max_per_gpu=1)
