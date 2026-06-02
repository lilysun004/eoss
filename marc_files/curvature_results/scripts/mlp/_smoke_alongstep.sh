#!/bin/bash
#SBATCH -J smoke_alongstep
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:15:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/smoke_alongstep_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/smoke_alongstep_%j.err
source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss
/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr 0.03 --batch_size 8 \
    --steps 400 --track_from 200 --track_until 400 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 20 \
    --results_subfolder smoke_alongstep
