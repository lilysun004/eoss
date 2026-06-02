#!/bin/bash
#SBATCH -J ce_batch_probe
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/ce_batch_probe_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/ce_batch_probe_%A_%a.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# EoSS is a STOCHASTIC criterion: batch_sharpness = E[g_b H_b g_b/|g_b|^2] pins at
# 2/lr, while lmax sits below. At b=128 the CE runs sat at batch_sharp ratio 0.50-0.67
# (too close to full-batch for the stochastic gap to develop). The canonical MSE EoSS
# cell (CNN SGD lr=0.02 b=32) shows batch_sharp ratio 1.10 with lmax only 0.74.
# Lever: smaller batch raises batch_sharpness toward 2/lr. Sweep batch at fixed
# eps=0.1, lr=0.03; criterion is lr*batch_sharpness/2 -> 1.0 (NOT lmax).
BATCHES=(8 16 32 64)
B=${BATCHES[$SLURM_ARRAY_TASK_ID]}

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr 0.03 --batch_size $B \
    --steps 20000 --track_from 19999 --track_until 20000 \
    --results_subfolder ce_batch_probe_b${B}
