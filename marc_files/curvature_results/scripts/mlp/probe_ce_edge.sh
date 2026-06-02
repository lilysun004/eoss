#!/bin/bash
#SBATCH -J ce_edge_probe
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/ce_edge_probe_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/ce_edge_probe_%A_%a.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Edge-finding probe for CE+label-smoothing. The eps=0.1 plateaus all settled BELOW
# the EoS threshold (lr*lmax/2 ~= 0.69-0.81), i.e. sub-critical -> no period-2 -> weak
# Fig-29 signal. Goal: park at ratio ~= 1.0 *sustained*. Two levers: push LR higher at
# eps=0.1, or loosen smoothing to eps=0.05 so sharpness can climb to the edge.
# No curvature/tracking (fast, ~2-3 min); lmax logged every ~256 steps is all we need.
#   idx 0: eps=0.05 lr=0.02     idx 1: eps=0.05 lr=0.03
#   idx 2: eps=0.10 lr=0.07     idx 3: eps=0.10 lr=0.10
case "$SLURM_ARRAY_TASK_ID" in
  0) EPS=0.05; LR=0.02 ;;
  1) EPS=0.05; LR=0.03 ;;
  2) EPS=0.10; LR=0.07 ;;
  3) EPS=0.10; LR=0.10 ;;
esac

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing $EPS \
    --optimizer_name SGD --lr $LR --batch_size 128 \
    --steps 20000 --track_from 19999 --track_until 20000 \
    --results_subfolder ce_edge_probe_eps${EPS}_lr${LR}
