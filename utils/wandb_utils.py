import os
import json
from pathlib import Path
import secrets
from typing import Any, Dict, Mapping

import numpy as np
import torch

try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when wandb missing
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


from .naming import sanitize_run_name_part as _sanitize_run_name_part
from .naming import compose_run_name as _compose_run_name


def is_wandb_available() -> bool:
    """Return whether the wandb library is importable."""
    return WANDB_AVAILABLE


def generate_run_id() -> str:
    """Generate a run identifier compatible with wandb directories."""

    if WANDB_AVAILABLE:
        if wandb.run:  # type: ignore[truthy-bool]
            return wandb.run.id  # type: ignore[return-value]
        return wandb.util.generate_id()  # type: ignore[attr-defined]

    return secrets.token_hex(4)


def init_wandb(args, step_to_start):
    """
    Initialize Weights & Biases for experiment tracking.
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is not available; call init_wandb only when enabled.")

    run_name = _compose_run_name(args)

    tags = []
    if hasattr(args, 'wandb_tag') and args.wandb_tag is not None:
        tags.append(args.wandb_tag)

    # Handle continuation from existing wandb run
    if hasattr(args, 'cont_run_id') and args.cont_run_id is not None:
        wandb_config = vars(args)
        wandb_config['fork_from'] = args.cont_run_id
        wandb_config['fork_step'] = step_to_start

        run = wandb.init(
            entity="xqsun-massachusetts-institute-of-technology",
            project="eoss",
            mode=os.getenv("WANDB_MODE", "online"),
            name=run_name,
            config=wandb_config,
            fork_from=f"{args.cont_run_id}?_step={step_to_start}",
            save_code=True,
            tags=tags if tags else None,
            notes=getattr(args, 'wandb_notes', None)
        )
        print(f"Resumed wandb run: {args.cont_run_id}")
    else:
        run = wandb.init(
            entity="xqsun-massachusetts-institute-of-technology",
            project="eoss",
            mode=os.getenv("WANDB_MODE", "online"),
            name=run_name,
            config=vars(args),
            save_code=True,
            tags=tags if tags else None,
            notes=getattr(args, 'wandb_notes', None)
        )
        print(f"Started new wandb run: {run.id}")

    # --- metric definitions so W&B plots look correct -------------
    wandb.define_metric("step")
    for m in [
        "batch_loss", "full_loss", "lambda_max",
        "step_sharpness", "batch_sharpn", "accuracy",
        "final_lambda_max", "final_full_loss", "final_accuracy",
    ]:
        wandb.define_metric(m, step_metric="step")

    return run


def _maybe_scalar(value: Any) -> Any:
    """Convert 0-D torch/np tensors to scalars for logging."""

    if isinstance(value, torch.Tensor) and value.dim() == 0:
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def log_metrics(metrics: Mapping[str, Any]) -> None:
    """Log a dictionary of metrics to wandb after basic sanitisation."""

    if not WANDB_AVAILABLE:
        return

    if metrics is None:
        return

    metrics_dict = dict(metrics)

    all_eigenvalues = metrics_dict.pop('all_eigenvalues', None)

    filtered_metrics: Dict[str, Any] = {}

    for key, value in metrics_dict.items():
        value = _maybe_scalar(value)

        if value is None:
            continue

        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue

        filtered_metrics[key] = value

    if all_eigenvalues is not None:
        if hasattr(all_eigenvalues, 'tolist'):
            eigenvalues = all_eigenvalues.tolist()
        elif isinstance(all_eigenvalues, (list, tuple)):
            eigenvalues = list(all_eigenvalues)
        else:
            eigenvalues = [all_eigenvalues]

        for idx, eigenval in enumerate(eigenvalues, start=1):
            eigenval = _maybe_scalar(eigenval)
            if eigenval is None:
                continue
            if isinstance(eigenval, (float, np.floating)) and np.isnan(eigenval):
                continue
            filtered_metrics[f"eigenvalue_{idx}"] = eigenval

    if filtered_metrics:
        wandb.log(filtered_metrics)


def save_checkpoint_wandb(model, optimizer, step, epoch, loss, run_id=None, save_every_n_steps=None):
    """
    Save model checkpoint to separate wandb_checkpoints directory organized by run ID.
    """
    if save_every_n_steps is not None and step % save_every_n_steps != 0:
        return None

    if run_id is None:
        if not WANDB_AVAILABLE or wandb.run is None:
            raise RuntimeError("run_id must be provided when wandb is disabled.")
        run_id = wandb.run.id

    wandb_dir = Path(os.environ.get("WANDB_DIR", "."))
    checkpoint_base_dir = wandb_dir / "wandb_checkpoints"
    checkpoint_dir = checkpoint_base_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'run_id': run_id
    }

    torch.save(checkpoint_data, checkpoint_path)

    # Update checkpoint metadata
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'checkpoints': []}

    checkpoint_info = {
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'filename': f"checkpoint_step_{step}.pt",
        'path': str(checkpoint_path)
    }

    metadata['checkpoints'] = [c for c in metadata['checkpoints'] if c['step'] != step]
    metadata['checkpoints'].append(checkpoint_info)

    metadata['checkpoints'].sort(key=lambda x: x['step'])

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return checkpoint_path


def find_closest_checkpoint_wandb(target_step, run_id=None, checkpoint_dir=None):
    """
    Find the checkpoint with step closest to (but not exceeding) target_step.
    """
    if checkpoint_dir is None:
        if run_id is None:
            if not WANDB_AVAILABLE or wandb.run is None:
                raise RuntimeError("run_id must be provided when wandb is disabled.")
            run_id = wandb.run.id

        checkpoint_base_dir = Path("wandb_checkpoints")
        checkpoint_dir = checkpoint_base_dir / run_id

        if not checkpoint_dir.exists():
            return None
    else:
        checkpoint_dir = Path(checkpoint_dir)

    metadata_path = checkpoint_dir / "checkpoint_metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    checkpoints = metadata.get('checkpoints', [])
    if not checkpoints:
        return None

    suitable_checkpoints = [c for c in checkpoints if c['step'] <= target_step]

    if not suitable_checkpoints:
        return None

    closest_checkpoint = max(suitable_checkpoints, key=lambda x: x['step'])

    return closest_checkpoint


def load_checkpoint_wandb(checkpoint_info, model, optimizer=None):
    """
    Load model and optimizer from wandb checkpoint.
    """
    checkpoint_path = Path(checkpoint_info['path'])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = next(model.parameters()).device
    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint_data['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    return {
        'step': checkpoint_data['step'],
        'epoch': checkpoint_data['epoch'],
        'loss': checkpoint_data['loss'],
        'run_id': checkpoint_data.get('run_id')
    }


def get_checkpoint_dir_for_run(run_id):
    """
    Get checkpoint directory for a specific run ID.
    """
    wandb_dir = Path(os.environ.get("WANDB_DIR", "."))
    checkpoint_base_dir = wandb_dir / "wandb_checkpoints"
    checkpoint_dir = checkpoint_base_dir / run_id

    if checkpoint_dir.exists():
        return checkpoint_dir

    return None
