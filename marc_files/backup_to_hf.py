"""
Backup script: upload EoSS experiment results, bert embedding cache,
and uniquely trained models to HuggingFace under marcwalden/.

Uses upload_folder() per sweep so all files land in one commit — avoids
the free-tier 1000 requests/5-min rate limit that kills per-file uploads.

Run AFTER logging in:
  /n/home06/mwalden/.conda/envs/eoss/bin/python -c "
    from huggingface_hub import login; login()"

Then:
  /n/home06/mwalden/.conda/envs/eoss/bin/python marc_files/backup_to_hf.py

Progress is logged to marc_files/backup_to_hf.log.
Idempotent: HF's upload_folder skips files whose SHA matches what's already
on the hub, so re-running only uploads missing/changed files.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/n/home06/mwalden/eoss/marc_files/backup_to_hf.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

from huggingface_hub import HfApi, create_repo

api = HfApi()
HF_USER = "marcwalden"

# ---------------------------------------------------------------------------
# 1.  EoSS experiment results  →  marcwalden/eoss-results  (dataset)
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results")
EOSS_DATASET_REPO = f"{HF_USER}/eoss-results"

# Sweeps to upload (skip: wandb, smoke_*, marc_projection_test, progressive_distill)
RESULT_SWEEPS = [
    "SST_opt_batch_sweep",
    "marc_cnn_sweep_fixed_u",
    "marc_cnn_sweep_fixed_u_n16384",
    "marc_vit_sweep",
    "MLP_sweep",
    "marc_batch_sweep",
    "tangent_drift_cnn",
    "tangent_drift_cnn_optsweep",
    "tangent_drift_mlp_optsweep",
    "tangent_drift_sst_optsweep",
    "tangent_drift_vit_optsweep",
    "qwen_probe4",
]

# Only keep these file types; skip rendered histograms, checkpoints, etc.
ALLOW_PATTERNS = ["*.npz", "*.txt", "*.json", "*.csv", "*.pt"]

# ---------------------------------------------------------------------------
# 2.  BERT embedding projection cache  →  same dataset repo
# ---------------------------------------------------------------------------
BERT_EMB = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/bert_emb_proj64.pt")

# ---------------------------------------------------------------------------
# 3.  Fine-tuned models  →  individual HF model repos
# ---------------------------------------------------------------------------
MODELS_ROOT = Path("/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models")
FINE_TUNED_MODELS = [
    "gemma-3-270m-distill-sftlr1e-4-seed1",
    "gemma-3-270m-progdistill-sftlr1e-4-seed1-round_1600",
    "qwen2.5-0.5b-distill-sftlr1e-5-seed1",
    "qwen2.5-0.5b-progdistill-sftlr1e-6-seed1",
]


def ensure_repo(repo_id: str, repo_type: str = "model") -> None:
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
        log.info(f"Repo ready: {repo_id}")
    except Exception as e:
        log.warning(f"create_repo {repo_id}: {e}")


def upload_results() -> None:
    """Upload per-run data files for each sweep using upload_folder().

    One upload_folder() call per sweep = one commit regardless of file count,
    so we stay well under the free-tier 1000 API-requests/5-min limit.
    Already-uploaded files whose SHA matches are skipped automatically.
    """
    ensure_repo(EOSS_DATASET_REPO, repo_type="dataset")

    # bert embedding cache — a single .pt file sitting outside any sweep folder
    if BERT_EMB.exists():
        log.info(f"Uploading {BERT_EMB.name} ({BERT_EMB.stat().st_size / 1e6:.1f} MB)")
        try:
            api.upload_file(
                path_or_fileobj=str(BERT_EMB),
                path_in_repo=BERT_EMB.name,
                repo_id=EOSS_DATASET_REPO,
                repo_type="dataset",
            )
            log.info(f"  done: {BERT_EMB.name}")
        except Exception as e:
            log.error(f"  FAILED {BERT_EMB.name}: {e}")

    for sweep in RESULT_SWEEPS:
        sweep_dir = RESULTS_ROOT / sweep
        if not sweep_dir.exists():
            log.warning(f"SKIP (not found): {sweep_dir}")
            continue

        n_runs = sum(1 for d in sweep_dir.iterdir() if d.is_dir())
        size_mb = sum(f.stat().st_size for f in sweep_dir.rglob("*")
                      if f.is_file() and any(f.match(p) for p in ALLOW_PATTERNS)) / 1e6
        log.info(f"Uploading sweep {sweep}: {n_runs} runs, ~{size_mb:.0f} MB")
        try:
            api.upload_folder(
                folder_path=str(sweep_dir),
                path_in_repo=sweep,
                repo_id=EOSS_DATASET_REPO,
                repo_type="dataset",
                allow_patterns=ALLOW_PATTERNS,
            )
            log.info(f"  done: {sweep}")
        except Exception as e:
            log.error(f"  FAILED {sweep}: {e}")

    log.info("Results upload complete.")


def upload_models() -> None:
    """Upload fine-tuned model directories to individual HF model repos."""
    for model_name in FINE_TUNED_MODELS:
        src = MODELS_ROOT / model_name
        if not src.exists():
            log.warning(f"SKIP (not found): {src}")
            continue
        repo_id = f"{HF_USER}/{model_name}"
        ensure_repo(repo_id, repo_type="model")
        size_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e6
        log.info(f"Uploading model {model_name} ({size_mb:.0f} MB) → {repo_id}")
        try:
            api.upload_folder(
                folder_path=str(src),
                repo_id=repo_id,
                repo_type="model",
            )
            log.info(f"  done: {repo_id}")
        except Exception as e:
            log.error(f"  FAILED {repo_id}: {e}")

    log.info("Model uploads complete.")


if __name__ == "__main__":
    log.info("=== EoSS HuggingFace backup (v2 — batched) ===")
    log.info("Step 1: experiment results → marcwalden/eoss-results")
    upload_results()
    log.info("=== Backup complete ===")
