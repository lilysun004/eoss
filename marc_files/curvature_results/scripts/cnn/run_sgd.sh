#!/bin/bash
#SBATCH -J curv_cnn
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 02:30:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_cnn_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_cnn_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn --dataset cifar10 --num_data 16384 \
    --optimizer_name SGD --lr 0.02 --batch_size 32 \
    --steps 35000 --track_from 30000 --track_until 35000 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_cnn
