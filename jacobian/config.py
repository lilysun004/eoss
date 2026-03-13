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
from training import train


# -------------------------------------
# Paths (hardcoded — never uses $RESULTS)
# -------------------------------------
if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. "
                     "Use 'export DATASETS=/path/to/datasets'")

DATASET_FOLDER = Path(os.environ.get('DATASETS')).expanduser()
RES_FOLDER = Path(__file__).resolve().parent / 'results'


# =============================================
# CONFIGURE YOUR EXPERIMENT HERE
# =============================================

# --- Training ---
batch_size         = 64
steps              = 100_000
epochs             = None
lr                 = 0.005
optimizer_name     = 'SGD'
optimizer_params   = {}
stop_loss          = 0.00001
gpu                = 0

# --- Loss ---
loss_type          = 'mse'

# --- Dataset ---
dataset            = 'cifar10'
num_data           = 8192

# --- Model ---
model              = 'mlp'
init_scale         = 0.2
no_init            = False

# --- Jacobian / Lyapunov tracking ---
ema_taus                 = (1, 5, 10, 20)   # EMA time constants; rho_τ = exp(EMA(ell, α=1/τ))
full_loss_every          = 256            # log full-dataset loss every N steps

# --- Seeds ---
seed               = 88881
dataset_seed       = 888
init_seed          = 8888
data_ordering_seed = 42


# =============================================
# CLI OVERRIDES: python config.py --lr 0.01 --gpu 1
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

    # ----- Loss Function -----
    if loss_type == 'mse':
        loss_fn = SquaredLoss()
    elif loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # ----- Dataset and Model -----
    dataset_presets = get_dataset_presets()
    model_presets = get_model_presets()

    data = prepare_dataset(dataset, DATASET_FOLDER, num_data, [], dataset_seed,
                           loss_type=loss_type)

    params = model_presets[model]['params']
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']
    net = prepare_net(model_type=model_presets[model]['type'], params=params)

    if not no_init:
        initialize_net(net, scale=init_scale, seed=init_seed)

    # ----- Optimizer -----
    optimizer = create_optimizer(optimizer_name, net, lr, optimizer_params)

    # ----- Result Storage -----
    args = SimpleNamespace(
        batch=batch_size, epochs=epochs, steps=steps, lr=lr, loss=loss_type,
        dataset=dataset, num_data=num_data, model=model,
        init_scale=init_scale, optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
    )
    RES_FOLDER.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, RES_FOLDER)

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
        stop_loss=stop_loss,
        data_ordering_seed=data_ordering_seed,
        ema_taus=ema_taus,
        full_loss_every=full_loss_every,
    )
