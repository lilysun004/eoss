#!/bin/bash
#SBATCH -J qwen_bimod
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --array=0-9
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/qwen_bimod_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/qwen_bimod_%a_%j.err

# Qwen2.5-0.5B-Instruct bimodality sweep on SST-2.
# SGD-Momentum beta=0.5, batch=32, 10 LR values.
# EoSS condition (beta=0.5): batch_sharpness = 2*(1-0.5)/lr = 1/lr.
# Adjust LRS based on probe_qwen_bimodality.sh results before submitting.
#
# Submit: env -u SBATCH_PARTITION -u SBATCH_ACCOUNT sbatch marc_files/sweep_optimizers/sweep_qwen_bimodality.sh

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

# Placeholder LR grid — replace with probe-guided values after running probe script.
# Current range assumes batch_sharpness ~ 500-5000, targeting lr*BS/2 -> 1.
#  ID     lr
#   0   5e-5
#   1   8e-5
#   2   1e-4
#   3   2e-4
#   4   3e-4
#   5   5e-4
#   6   7e-4
#   7   1e-3
#   8   2e-3
#   9   5e-3
LRS=(5e-5 8e-5 1e-4 2e-4 3e-4 5e-4 7e-4 1e-3 2e-3 5e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Qwen bimodality task $SLURM_ARRAY_TASK_ID -> SGD-Momentum lr=$LR batch=32"

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
    --steps 50000 \
    --track_from 40000 \
    --track_until 50000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder Qwen_SST_sweep
