#!/bin/bash
#SBATCH -J e806_mlp_lr0.1
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/e806_mlp_lr0.1_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/e806_mlp_lr0.1_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/early_bimodality_scan_8k/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp \
    --dataset cifar10 \
    --num_data 8192 \
    --optimizer_name SGD \
    --lr 0.1 \
    --batch_size 32 \
    --steps 8000 \
    --stop_loss None \
    --track_from 0 \
    --track_until 8000 \
    --track_stride 2 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder early_scan_mlp_8k
