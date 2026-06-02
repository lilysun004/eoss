#!/bin/bash
#SBATCH -J qwen_probe4
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --array=0-3
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/qwen_probe4_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/qwen_probe4_%a_%j.err

# 4-cell probe for Qwen2.5-0.5B on SST-2, SGD-Momentum beta=0.5 batch=32.
# Uses base Qwen2.5-0.5B (already downloaded; switch to Instruct after download).
# EoSS condition (beta=0.5): batch_sharpness = 1/lr.
# Covers 2 log-decades to find where lr*BS/2 -> 1.
#
# Submit: env -u SBATCH_PARTITION -u SBATCH_ACCOUNT sbatch marc_files/sweep_optimizers/probe_qwen_4cell.sh

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

# ID   lr        (target BS = 1/lr for beta=0.5)
#  0   1e-4      -> BS target ~10000
#  1   5e-4      -> BS target ~2000
#  2   1e-3      -> BS target ~1000
#  3   5e-3      -> BS target ~200
LRS=(1e-4 5e-4 1e-3 5e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Qwen probe4 task $SLURM_ARRAY_TASK_ID -> lr=$LR"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --dataset qwen_sst2 \
    --model qwen_classifier \
    --qwen_model_path /n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/models/Qwen2.5-0.5B \
    --loss_type ce \
    --label_smoothing 0.1 \
    --num_data 8192 \
    --batch_size 32 \
    --optimizer_name SGD-Momentum \
    --optimizer_params "{'beta': 0.5}" \
    --lr "$LR" \
    --steps 10000 \
    --track_from 8000 \
    --track_until 10000 \
    --fixed_u True \
    --results_subfolder qwen_probe4
