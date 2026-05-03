#!/bin/bash
#SBATCH --job-name=sst_optbatch
#SBATCH --output=logs/sst_optbatch_%a_%j.out
#SBATCH --error=logs/sst_optbatch_%a_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-23

# ============================================================================
# SST-2 SSTTransformer bimodality sweep — 6 optimizers × 4 batch sizes = 24 cells
# Cell layout mirrors marc_files/sweep_optimizers/mlp_sweep_requeue.sh.
# Run with: sbatch marc_files/sweep_optimizers/sweep_sst_opt_batch_mit.sh
#
# >>> EDIT BEFORE FIRST USE (lines marked EDIT) <<<
#   - DATASETS / RESULTS env vars (point at your MIT scratch / project dirs)
#   - REPO_DIR (where you cloned eoss on MIT)
#   - LRs per optimizer (table below) if defaults don't land near EoS for SST
# ============================================================================

eval "$(conda shell.bash hook)"
conda activate eoss

export PYTHONUNBUFFERED=1

# EDIT: point these at your MIT data + results paths
export DATASETS="${DATASETS:-$HOME/eoss_data}"      # bert-base-uncased + GLUE/sst2 cache live under here
export RESULTS="${RESULTS:-$HOME/eoss_results}"

# EDIT: path to the eoss repo on MIT
REPO_DIR="${REPO_DIR:-$HOME/eoss}"
cd "$REPO_DIR"

mkdir -p logs

# Cell table: (optimizer, lr, optimizer_params, batch_size).
# Steps + tracking window are uniform (50k / 40k–50k) per the project default.
# LRs taken from mlp_sweep_requeue.sh. Adjust per-optimizer if SST doesn't show
# bimodality at these values (SST + SGDM β=0.5 anchored at lr≈0.02 in the
# marcwalden1 reference, so lr=0.002 with β=0.9 should be in the right ballpark).
case "$SLURM_ARRAY_TASK_ID" in
  0)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="8"    ;;
  1)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="32"   ;;
  2)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="128"  ;;
  3)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="1024" ;;
  4)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="8"    ;;
  5)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="32"   ;;
  6)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="128"  ;;
  7)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="1024" ;;
  8)  OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="8"    ;;
  9)  OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="32"   ;;
  10) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="128"  ;;
  11) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="1024" ;;
  12) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="8"    ;;
  13) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="32"   ;;
  14) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="128"  ;;
  15) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="1024" ;;
  16) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="8"    ;;
  17) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="32"   ;;
  18) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="128"  ;;
  19) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="1024" ;;
  20) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="8"    ;;
  21) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="32"   ;;
  22) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="128"  ;;
  23) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="1024" ;;
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

STEPS=50000
TRACK_FROM=40000
TRACK_UNTIL=50000

echo "SST opt×batch task $SLURM_ARRAY_TASK_ID -> $OPT lr=$LR batch=$BATCH"

CMD=(python config.py
    --dataset sst2
    --model sst_transformer
    --loss_type mse
    --num_data 8192
    --optimizer_name "$OPT"
    --lr "$LR"
    --batch_size "$BATCH"
    --steps "$STEPS"
    --stop_loss None
    --track_from "$TRACK_FROM"
    --track_until "$TRACK_UNTIL"
    --more_freq_measure True
    --fixed_u True
    --results_subfolder SST_opt_batch_sweep)

if [ -n "$PARAMS" ]; then
    CMD+=(--optimizer_params "$PARAMS")
fi

"${CMD[@]}"
