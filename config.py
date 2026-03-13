import torch
import torch as T
import torch.nn as nn
import os
import numpy as np
import random
from pathlib import Path
from types import SimpleNamespace

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, prepare_net, initialize_net, get_model_presets
from utils.optimizer import create_optimizer
from utils.storage import initialize_folders
from utils.wandb_utils import (
    find_closest_checkpoint_wandb,
    load_checkpoint_wandb,
    get_checkpoint_dir_for_run,
    generate_run_id,
)
from utils.frequency import frequency_calculator
from training import train


# -------------------------------------
# Environment
# -------------------------------------
if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
if 'RESULTS' not in os.environ:
    raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

DATASET_FOLDER = Path(os.environ.get('DATASETS')).expanduser()
RES_FOLDER = Path(os.environ.get('RESULTS')).expanduser()


# =============================================
# CONFIGURE YOUR EXPERIMENT HERE
# =============================================

# --- Training ---
batch_size         = 64        # mini-batch size
steps              = 100_000     # total training steps (set to None if using epochs)
epochs             = None      # total epochs (set to None if using steps; cannot set both)
lr                 = 0.005      # learning rate
optimizer_name     = 'SGD'     # 'SGD' | 'SGD-Momentum' | 'SGD-Nesterov' | 'Adam'
optimizer_params   = {}
# Optimizer params examples:
#   SGD:          {}
#   SGD-Momentum: {'beta': 0.9}
#   SGD-Nesterov: {'beta': 0.9}
#   Adam:         {'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}  (all optional, these are defaults)
probe_samples      = 128       # number of probe batches for batch_sharpness and probe GBS
stop_loss          = 0.00001      # early-stop when loss drops below this; None = disabled
gpu                = 0         # which GPU to use (0 or 1); None = default; 'cpu' = force CPU
results_subfolder  = '0311_alloptimizers'      # save inside RES_FOLDER/results_subfolder/; None = RES_FOLDER directly

# --- Loss ---
loss_type          = 'mse'     # 'mse' (SquaredLoss) | 'ce' (CrossEntropyLoss)

# --- Dataset ---
dataset            = 'cifar10' # 'cifar10' | 'cifar10_2cls' | 'cifar10_ez' | 'svhn' | 'fmnist' | 'cinic10' | 'imagenet32'
num_data           = 8192      # number of training samples to use

# --- Model ---
model              = 'mlp'     # 'mlp' | 'mlp2' | 'mlp3' | 'mlp_s' | 'mlp_l' | 'mlp_silu' | 'mlp_tanh' | 'linear' | 'cnn' | 'resnet' | 'resnet_bn' | 'wrn' | 'wrn_no_bn'
init_scale         = 0.2       # weight initialization scale
no_init            = False     # True = skip custom weight initialization

# --- Measurements (True/False to enable/disable) ---
compute_full_batch          = True    # compute GBS using actual batch g/s + full-batch Hessian u
compute_full_batch_every    = 256   # also controls lambdamax frequency (must be the same)

lambdamax          = True      # compute lambda_max (top Hessian eigenvalue on full batch)
lambdamax_every    = compute_full_batch_every  # keep in sync with compute_full_batch_every

batch_sharpness    = True      # E[gHg/g^2] averaged over probe_samples mini-batches
full_loss_every    = 32

compute_actual_batch        = True    # compute A, A_u, B, B_u on the actual training batch
compute_actual_batch_every  = 1     # actual-batch GBS at every step

compute_probe_batch         = True    # compute A, A_u, B, B_u averaged over probe_samples batches (u = u_avg)
compute_probe_batch_every   = 256   # probe-batch GBS + batch_sharpness every N steps

gbs_power_iters     = 50       # power iteration steps to find top eigenvector u
compute_u           = False     # False = skip A_u/B_u projections (saves ~2x per GBS step); lambda_max still computed normally

# --- Additional measurements ---
step_sharpness     = False     # single-batch Rayleigh quotient gHg/g^2
step_sharpness_every       = 16
final              = False     # extra lambda_max measurement after training ends
force_full_loss    = False     # periodic full-batch loss & accuracy logging
full_loss_warmup   = False     # full-batch loss every step for first ~128 steps after start

# --- Measurement tuning ---
cache_eigenvectors  = True     # warm-start eigenvector cache for faster eigenvalue computation
use_power_iteration = False    # True = power iteration; False = LOBPCG
num_eigenvalues     = 1        # number of top eigenvalues to compute (1 = just lambda_max)

# --- Checkpoints ---
checkpoint_every    = None     # save every N steps; None = auto (total_steps / 100)
save_only_last      = True    # True = skip intermediate checkpoints

# --- Continuation from a previous run ---
cont_run_id         = None     # run ID to resume from; None = fresh run
cont_step           = None     # step to resume from (must set both or neither)

# --- Seeds (all fixed for determinism) ---
seed               = 88881     # global torch/numpy/random seed
dataset_seed       = 888       # seed for dataset subsampling
init_seed          = 8888      # seed for weight initialization
data_ordering_seed = 42        # seed for epoch-wise data shuffling; None = non-deterministic


# =============================================
# CLI OVERRIDES: python config.py --lr 0.01 --gpu 1 --optimizer_params "{'beta': 0.9}"
# =============================================
import sys
import ast as _ast

_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    if _args[_i].startswith('--'):
        _key = _args[_i][2:]
        _val_str = _args[_i + 1]
        try:
            _val = _ast.literal_eval(_val_str)
        except (ValueError, SyntaxError):
            _val = _val_str
        if _key not in globals():
            raise ValueError(f"Unknown config key: {_key}")
        globals()[_key] = _val
        _i += 2
    else:
        _i += 1


# =============================================
# EVERYTHING BELOW RUNS THE EXPERIMENT
# =============================================

if __name__ == '__main__':
    # ----- Reproducibility -----
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----- Device -----
    if gpu == 'cpu':
        device = 'cpu'
    elif T.cuda.is_available():
        device = T.device(f'cuda:{gpu}' if gpu is not None else 'cuda')
    else:
        device = 'cpu'

    # ----- Validation -----
    if steps is not None and epochs is not None:
        raise ValueError("Set either epochs or steps, not both")

    if (cont_run_id is not None) != (cont_step is not None):
        raise ValueError("Both cont_run_id and cont_step must be set together")

    # ----- Measurement frequencies -----
    frequency_calculator.set_interval('full_batch_lambda_max', lambdamax_every)
    frequency_calculator.set_interval('batch_sharpness', compute_probe_batch_every)
    frequency_calculator.set_interval('actual_batch_gbs', compute_actual_batch_every)
    frequency_calculator.set_interval('probe_batch_gbs', compute_probe_batch_every)
    frequency_calculator.set_interval('step_sharpness', step_sharpness_every)
    frequency_calculator.set_interval('full_loss', full_loss_every)

    # ----- Measurements -----
    measurements = {name for name, enabled in [
        ('lmax', lambdamax),
        ('step_sharpness', step_sharpness),
        ('batch_sharpness', batch_sharpness),
        ('actual_batch_gbs', compute_actual_batch),
        ('probe_batch_gbs', compute_probe_batch),
        ('full_batch_gbs', compute_full_batch),
        ('final', final),
        ('full_loss_warmup', full_loss_warmup),
        ('full_loss', force_full_loss),
    ] if enabled}

    # ----- Loss Function -----
    if loss_type == 'mse':
        loss_fn = SquaredLoss()
    elif loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss()

    # ----- Dataset and Model -----
    dataset_presets = get_dataset_presets()
    model_presets = get_model_presets()

    data = prepare_dataset(dataset, DATASET_FOLDER, num_data, [], dataset_seed, loss_type=loss_type)

    params = model_presets[model]['params']
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']
    net = prepare_net(model_type=model_presets[model]['type'], params=params)

    if not no_init:
        initialize_net(net, scale=init_scale, seed=init_seed)

    # ----- Checkpoint Continuation -----
    step_to_start = 0
    epoch_to_start = 0
    if cont_run_id is not None and cont_step is not None:
        checkpoint_dir = get_checkpoint_dir_for_run(cont_run_id)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"Cannot find checkpoint directory for run ID: {cont_run_id}")

        checkpoint_info = find_closest_checkpoint_wandb(cont_step, checkpoint_dir=checkpoint_dir)
        if checkpoint_info is None:
            raise FileNotFoundError(f"No suitable checkpoint found for step {cont_step} in run {cont_run_id}")

        loaded_data = load_checkpoint_wandb(checkpoint_info, net)
        step_to_start = loaded_data['step']
        epoch_to_start = loaded_data['epoch']
        print(f"Loaded checkpoint from step {loaded_data['step']} (epoch {loaded_data['epoch']}) from run {cont_run_id}")

    # ----- Optimizer -----
    optimizer = create_optimizer(optimizer_name, net, lr, optimizer_params)

    # ----- Args namespace for storage internals -----
    args = SimpleNamespace(
        batch=batch_size, epochs=epochs, steps=steps, lr=lr, loss=loss_type,
        dataset=dataset, num_data=num_data, model=model,
        init_scale=init_scale, optimizer_name=optimizer_name,
        optimizer_params=optimizer_params, probe_samples=probe_samples,
        cont_run_id=cont_run_id, cont_step=cont_step,
    )

    # ----- Result Storage -----
    save_root = RES_FOLDER / results_subfolder if results_subfolder else RES_FOLDER
    save_root.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, save_root)

    # ----- Run ID -----
    run_id = generate_run_id()

    # ----- Checkpoint Cadence -----
    checkpoint_every_n_steps = checkpoint_every
    if len(measurements) == 0 or (len(measurements) == 1 and 'final' in measurements) or save_only_last:
        checkpoint_every_n_steps = 1_000_000_000

    # ----- Train -----
    train(
        net=net,
        optimizer=optimizer,
        data=data,
        max_epochs=epochs,
        max_steps=steps,
        batch_size=batch_size,
        save_to=run_folder,
        device=device,
        loss_fn=loss_fn,
        verbose=True,
        stop_loss=stop_loss,
        epoch_to_start=epoch_to_start,
        step_to_start=step_to_start,
        measurements=measurements,
        cache_eigenvectors=cache_eigenvectors,
        use_power_iteration=use_power_iteration,
        num_eigenvalues=num_eigenvalues,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        data_ordering_seed=data_ordering_seed,
        run_id=run_id,
        probe_samples=probe_samples,
        gbs_power_iters=gbs_power_iters,
        compute_u=compute_u,
    )
