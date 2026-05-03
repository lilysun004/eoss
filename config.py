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
results_subfolder  = '0318_alloptimizers_targeted_finebatch'      # save inside RES_FOLDER/results_subfolder/; None = RES_FOLDER directly

# --- Loss ---
loss_type          = 'mse'     # 'mse' (SquaredLoss) | 'ce' (CrossEntropyLoss) | 'logistic' (LogisticLoss; SST {-1,+1})

# --- Dataset ---
dataset            = 'cifar10' # 'cifar10' | 'cifar10_2cls' | 'cifar10_ez' | 'svhn' | 'fmnist' | 'cinic10' | 'imagenet32' | 'sst2'
num_data           = 8192      # number of training samples to use

# --- Model ---
model              = 'mlp'     # 'mlp' | 'mlp2' | 'mlp3' | 'mlp_s' | 'mlp_l' | 'mlp_silu' | 'mlp_tanh' | 'linear' | 'cnn' | 'resnet' | 'resnet_bn' | 'wrn' | 'wrn_no_bn' | 'vit' | 'sst_transformer'
init_scale         = 0.2       # weight initialization scale
no_init            = False     # True = skip custom weight initialization

# --- Measurements (True/False to enable/disable) ---
full_loss_every    = 32

batch_sharpness             = True      # E[gHg/g^2] averaged over probe_samples mini-batches
batch_sharpness_every       = 256

compute_probe_batch         = True    # compute A, B, GBS_out averaged over probe_samples batches
compute_probe_batch_every   = 256   # probe-batch GBS + batch_sharpness every N steps
power_iters                 = 50    # power iteration steps to find per-batch top eigenvector u

compute_distributions       = True  # save per-probe-batch arrays of ||g||, ||s||, g·s, s·u, g·u

compute_quantities_with_uB  = False  # False = skip per-batch power iteration + A_u/B_u/GBS_u/s_dot_u/g_dot_u

# --- Additional measurements ---
step_sharpness     = False     # single-batch Rayleigh quotient gHg/g^2
step_sharpness_every  = 16
final              = False     # extra lambda_max measurement after training ends
force_full_loss    = False     # periodic full-batch loss & accuracy logging
full_loss_warmup   = False     # full-batch loss every step for first ~128 steps after start

# --- Measurement tuning ---
cache_eigenvectors  = True     # warm-start eigenvector cache for faster eigenvalue computation
use_power_iteration = False    # True = power iteration; False = LOBPCG
num_eigenvalues     = 1        # number of top eigenvalues to compute (1 = just lambda_max)
measurement_batch_size_cap = None  # cap second-order measurement batches; None = automatic

# --- Projection histogram tracking ---
# Set both to enable tracking of <theta_t, v> projections at the edge of stability.
# CLI: python config.py --track_from 5000 --track_until 6000
track_from          = 20000    # step to start recording projections (None = disabled)
track_until         = 30000    # step to stop recording projections (None = disabled)
fixed_u             = False    # True = fix top Hessian eigenvec at track_from and reuse
projection_save_every = 100    # flush projections.npz every N tracked steps (0 = only at end)
track_stride        = 10       # record every Nth step within [track_from, track_until]
lobpcg_max_iters    = 20       # LOBPCG inner-loop cap for projection tracker top-5
lobpcg_reltol       = 0.02     # LOBPCG convergence tolerance for projection tracker top-5

# --- Measurement frequency ---
more_freq_measure   = False    # True = halve all measurement intervals

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
_overridden_keys = set()
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
        _overridden_keys.add(_key)
        _i += 2
    else:
        _i += 1

if model == 'vit':
    if 'no_init' not in _overridden_keys:
        no_init = True
    if 'measurement_batch_size_cap' not in _overridden_keys:
        measurement_batch_size_cap = 128

if model == 'sst_transformer':
    # Init handled inside SSTTransformer.__init__ (head zero-init, embeddings frozen);
    # initialize_net's branch is a no-op, so init_scale does not apply.
    if 'measurement_batch_size_cap' not in _overridden_keys:
        measurement_batch_size_cap = 128


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

    # ----- Measurement frequencies -----
    # more_freq_measure=True now halves intervals *only* inside the tracking
    # window [track_from, track_until]. Outside the window, base intervals apply.
    frequency_calculator.set_interval('full_batch_lambda_max', compute_probe_batch_every)
    frequency_calculator.set_interval('batch_sharpness',       batch_sharpness_every)
    frequency_calculator.set_interval('probe_batch_gbs',       compute_probe_batch_every)
    frequency_calculator.set_interval('distributions',         compute_probe_batch_every)
    frequency_calculator.set_interval('step_sharpness',        step_sharpness_every)
    frequency_calculator.set_interval('full_loss',             full_loss_every)
    frequency_calculator.set_window_halve(track_from, track_until, enabled=more_freq_measure)

    # ----- Measurements -----
    measurements = {'lmax'} | {name for name, enabled in [
        ('step_sharpness', step_sharpness),
        ('batch_sharpness', batch_sharpness),
        ('probe_batch_gbs', compute_probe_batch),
        ('final', final),
        ('full_loss_warmup', full_loss_warmup),
        ('full_loss', force_full_loss),
    ] if enabled}

    # ----- Loss Function -----
    if loss_type == 'mse':
        loss_fn = SquaredLoss()
    elif loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_type == 'logistic':
        from utils.nets import LogisticLoss
        loss_fn = LogisticLoss()

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

    # ----- Optimizer -----
    optimizer = create_optimizer(optimizer_name, net, lr, optimizer_params)

    # ----- Args namespace for storage internals -----
    args = SimpleNamespace(
        batch=batch_size, epochs=epochs, steps=steps, lr=lr, loss=loss_type,
        dataset=dataset, num_data=num_data, model=model,
        init_scale=init_scale, optimizer_name=optimizer_name,
        optimizer_params=optimizer_params, probe_samples=probe_samples,
    )

    # ----- Result Storage -----
    save_root = RES_FOLDER / results_subfolder if results_subfolder else RES_FOLDER
    save_root.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, save_root)

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
        measurements=measurements,
        cache_eigenvectors=cache_eigenvectors,
        use_power_iteration=use_power_iteration,
        num_eigenvalues=num_eigenvalues,
        data_ordering_seed=data_ordering_seed,
        probe_samples=probe_samples,
        power_iters=power_iters,
        compute_distributions=compute_distributions,
        compute_quantities_with_uB=compute_quantities_with_uB,
        track_from=track_from,
        track_until=track_until,
        fixed_u=fixed_u,
        projection_save_every=projection_save_every,
        track_stride=track_stride,
        lobpcg_max_iters=lobpcg_max_iters,
        lobpcg_reltol=lobpcg_reltol,
        measurement_batch_size_cap=measurement_batch_size_cap,
    )
