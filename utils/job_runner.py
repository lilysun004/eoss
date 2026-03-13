"""
GPU job scheduler: runs jobs across GPUs with configurable concurrency.

Jobs are split evenly across GPUs (round-robin). Each GPU runs up to
`max_per_gpu` jobs concurrently. When one finishes, the next job on
that GPU is immediately launched.
"""

import subprocess
import sys
import threading
import queue


def job_to_args(overrides, gpu):
    """Convert a dict of overrides to CLI args for config.py."""
    args = [sys.executable, 'config.py', '--gpu', str(gpu)]
    for key, val in overrides.items():
        args.extend([f'--{key}', repr(val)])
    return args


def _job_label(overrides):
    return ', '.join(f'{k}={v}' for k, v in overrides.items())


def _slot_worker(gpu, gpu_queue, total_jobs):
    """A single worker slot on a GPU. Grabs jobs from its GPU queue until empty."""
    while True:
        try:
            idx, overrides = gpu_queue.get_nowait()
        except queue.Empty:
            return
        label = _job_label(overrides)
        print(f"[GPU {gpu}] Starting job {idx+1}/{total_jobs}: {label}")
        args = job_to_args(overrides, gpu)
        result = subprocess.run(args)
        if result.returncode != 0:
            print(f"[GPU {gpu}] Job FAILED (exit {result.returncode}): {label}")
        else:
            print(f"[GPU {gpu}] Job done: {label}")
        gpu_queue.task_done()


def run_jobs(jobs, num_gpus=2, max_per_gpu=10):
    """Run a list of jobs across GPUs with up to max_per_gpu concurrent per GPU.

    Jobs are split round-robin across GPUs for even distribution.

    Parameters
    ----------
    jobs : list[dict]
        Each dict contains config overrides (e.g. {'lr': 0.01, 'optimizer_name': 'SGD'}).
    num_gpus : int
        Number of GPUs to use.
    max_per_gpu : int
        Max concurrent jobs per GPU.
    """
    total = len(jobs)

    # Split jobs round-robin into per-GPU queues
    gpu_queues = [queue.Queue() for _ in range(num_gpus)]
    gpu_job_lists = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        gpu = i % num_gpus
        gpu_queues[gpu].put((i, job))
        gpu_job_lists[gpu].append((i, job))

    # Print assignment
    for g in range(num_gpus):
        n = len(gpu_job_lists[g])
        print(f"GPU {g}: {n} jobs")
        for idx, j in gpu_job_lists[g]:
            print(f"  {idx+1}. {_job_label(j)}")

    # Launch worker slots per GPU
    threads = []
    for g in range(num_gpus):
        n_slots = min(max_per_gpu, gpu_queues[g].qsize())
        for _ in range(n_slots):
            t = threading.Thread(target=_slot_worker, args=(g, gpu_queues[g], total))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    print("All jobs finished.")
