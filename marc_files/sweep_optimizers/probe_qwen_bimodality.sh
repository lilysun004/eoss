#!/bin/bash
#SBATCH -J qwen_probe
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --array=0-5
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/qwen_probe_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/qwen_probe_%a_%j.err

# Short probe to find EoSS lr for Qwen2.5-0.5B-Instruct on SST-2.
# SGD-Momentum beta=0.5, batch=32, 6 LR values spanning 1e-4 to 5e-3.
# EoSS condition: batch_sharpness = 2*(1-beta)/lr = 1/lr.
# Judge each run by lr*batch_sharpness/2 -> 1 (see CLAUDE.md).
# Pick the LR closest to 1 for the full bimodality sweep.
#
# Submit: env -u SBATCH_PARTITION -u SBATCH_ACCOUNT sbatch marc_files/sweep_optimizers/probe_qwen_bimodality.sh

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

# ID   lr
#  0   1e-4
#  1   2e-4
#  2   5e-4
#  3   1e-3
#  4   2e-3
#  5   5e-3
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Qwen probe task $SLURM_ARRAY_TASK_ID -> SGD-Momentum lr=$LR batch=32"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --dataset qwen_sst2 \
    --model qwen_classifier \
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
    --results_subfolder qwen_probe
