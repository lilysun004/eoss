#!/bin/bash
#SBATCH -J ce_blr_matrix
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-14
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/ce_blr_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/ce_blr_%A_%a.err
source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Find the EoSS lr (lr*batch_sharpness/2 ~= 1.0) for each batch size, eps=0.1.
# 3 candidate lrs per batch (~sqrt scaling: larger batch -> higher lr). No curvature
# scan; lmax & batch_sharpness logged throughout. Winner per batch -> full curv run.
case "$SLURM_ARRAY_TASK_ID" in
  0)  B=8;    LR=0.02 ;;  1)  B=8;    LR=0.03 ;;  2)  B=8;    LR=0.05 ;;
  3)  B=32;   LR=0.04 ;;  4)  B=32;   LR=0.07 ;;  5)  B=32;   LR=0.10 ;;
  6)  B=128;  LR=0.07 ;;  7)  B=128;  LR=0.12 ;;  8)  B=128;  LR=0.18 ;;
  9)  B=1024; LR=0.12 ;; 10)  B=1024; LR=0.20 ;; 11)  B=1024; LR=0.30 ;;
  12) B=8192; LR=0.15 ;; 13) B=8192; LR=0.30 ;; 14) B=8192; LR=0.50 ;;
esac

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr $LR --batch_size $B \
    --steps 20000 --track_from 19999 --track_until 20000 \
    --results_subfolder ce_blr_b${B}_lr${LR}
