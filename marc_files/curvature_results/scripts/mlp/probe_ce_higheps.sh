#!/bin/bash
#SBATCH -J ce_higheps
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-1
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/ce_higheps_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/ce_higheps_%A_%a.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Sustained edge-ratio rose monotonically with smoothing (eps 0.05->0.56, 0.1->0.88).
# Test whether higher eps pushes the plateau to ratio ~= 1.0. lr=0.07 (the eps=0.1 peak).
case "$SLURM_ARRAY_TASK_ID" in
  0) EPS=0.15 ;;
  1) EPS=0.20 ;;
esac

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing $EPS \
    --optimizer_name SGD --lr 0.07 --batch_size 128 \
    --steps 20000 --track_from 19999 --track_until 20000 \
    --results_subfolder ce_higheps_eps${EPS}_lr0.07
