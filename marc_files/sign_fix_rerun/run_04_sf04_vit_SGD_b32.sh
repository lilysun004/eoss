#!/bin/bash
#SBATCH -J sf04_vit_SGD_b32
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/sf04_vit_SGD_b32_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/sf04_vit_SGD_b32_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model vit \
    --dataset cifar10 \
    --num_data 8192 \
    --optimizer_name SGD \
    --lr 0.0035 \
    --batch_size 32 \
    --optimizer_params "{}" \
    --loss_type mse \
    --steps 150000 \
    --track_from 145000 \
    --track_until 150000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_vit_sweep_signfix
