import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


RESULTS_COLUMNS = [
    "epoch", "step", "batch_loss", "full_loss",
    "lambda_max", "step_sharpness", "batch_sharpness",
    "A_actual", "A_u_actual", "B_actual", "B_u_actual", "out_actual", "out_actual_u",
    "A_probe", "A_u_probe", "B_probe", "B_u_probe", "out_probe", "out_probe_u",
    "A_full", "A_u_full", "B_full", "B_u_full", "out_full", "out_full_u",
    "total_accuracy",
]


def get_welcome_string(args):
    """
    Returns the header of the log file (comment lines + CSV header row).
    """

    msg = f"""# Edge of Stochastic Stability.
# Dataset: {args.dataset}, model {args.model}, lr {args.lr}, batch size {args.batch}
# Arguments: {str(args)}
{','.join(RESULTS_COLUMNS)}"""
    return msg


def initialize_folders(args, results_folder):
    def generate_folder_name(args):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M_%S')
        opt_name = getattr(args, 'optimizer_name', 'SGD')
        opt_params = getattr(args, 'optimizer_params', {})
        parts = [f'{timestamp}_{opt_name}_lr{args.lr:g}_b{args.batch}']
        for k, v in opt_params.items():
            parts.append(f'{k}-{v}')
        config_name = '_'.join(parts)
        return config_name

    while True:
        config_name = generate_folder_name(args)
        runs_folder = results_folder / config_name
        if not runs_folder.exists():
            try:
                runs_folder.mkdir(parents=True, exist_ok=False)
            except:
                time.sleep(2)
                continue
            break
        else:
            time.sleep(2)
            continue



    model_save_path = runs_folder / 'checkpoints'
    model_save_path.mkdir(parents=True, exist_ok=True)


    results_file = runs_folder / 'results.txt'
    welcome_string = get_welcome_string(args)

    with open(results_file, 'w') as f:
        f.write(welcome_string + "\n")

    return runs_folder


def parse_folder_name(name):
    """Parse a run folder name into a dict of config values.

    Returns dict with keys: optimizer_name, lr, batch_size, and any
    optimizer params (e.g. beta, beta1, beta2, eps).
    """
    import re
    # Strip timestamp prefix
    stripped = re.sub(r'^\d{8}_\d{4}_\d+_', '', name)
    # Extract optimizer name (leading letters/hyphens before _lr)
    m = re.match(r'^([A-Za-z][-A-Za-z]*)_lr([\d.]+)_b(\d+)(.*)$', stripped)
    if not m:
        return {}
    opt_name, lr_str, batch_str, rest = m.groups()
    result = {
        'optimizer_name': opt_name,
        'lr': float(lr_str),
        'batch_size': int(batch_str),
    }
    # Parse trailing optimizer params like _beta-0.9_eps-1e-8
    for pm in re.finditer(r'_([a-zA-Z]\w*)-([^_]+)', rest):
        k, v = pm.group(1), pm.group(2)
        try:
            result[k] = float(v)
        except ValueError:
            result[k] = v
    return result


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    """
    Persist a mapping as JSON using a temporary file to avoid partial writes.

    Parameters
    ----------
    path : Path
        Destination file path.
    payload : Mapping[str, Any]
        Data to serialise as JSON.
    indent : int, optional
        Indentation level for pretty-printing. Default is 2.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
    tmp_path.replace(path)
