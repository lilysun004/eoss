import time
from datetime import datetime
from pathlib import Path


RESULTS_COLUMNS = [
    "epoch", "step", "batch_loss", "full_loss",
    "ell", "rho_1", "rho_10", "rho_100",
    "total_accuracy",
]


def get_welcome_string(args):
    msg = (
        f"# Jacobian Spectral Radius Experiment.\n"
        f"# Dataset: {args.dataset}, model {args.model}, lr {args.lr}, batch size {args.batch}\n"
        f"# Arguments: {str(args)}\n"
        f"{','.join(RESULTS_COLUMNS)}"
    )
    return msg


def initialize_folders(args, results_folder):
    def generate_folder_name(args):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M_%S')
        opt_name = getattr(args, 'optimizer_name', 'SGD')
        opt_params = getattr(args, 'optimizer_params', {})
        parts = [f'{timestamp}_{opt_name}_lr{args.lr:g}_b{args.batch}']
        for k, v in opt_params.items():
            parts.append(f'{k}-{v}')
        return '_'.join(parts)

    while True:
        config_name = generate_folder_name(args)
        runs_folder = results_folder / config_name
        if not runs_folder.exists():
            try:
                runs_folder.mkdir(parents=True, exist_ok=False)
            except Exception:
                time.sleep(2)
                continue
            break
        else:
            time.sleep(2)
            continue

    results_file = runs_folder / 'results.txt'
    welcome_string = get_welcome_string(args)
    with open(results_file, 'w') as f:
        f.write(welcome_string + "\n")

    return runs_folder
