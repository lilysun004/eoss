#!/bin/bash
#SBATCH -J eb01_cnn_lr0.04
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 05:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/eb01_cnn_lr0.04_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/eb01_cnn_lr0.04_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/early_bimodality_scan/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn \
    --dataset cifar10 \
    --num_data 16384 \
    --optimizer_name SGD \
    --lr 0.04 \
    --batch_size 32 \
    --steps 3000 \
    --stop_loss None \
    --track_from 300 \
    --track_until 2300 \
    --track_stride 2 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder early_scan_cnn
