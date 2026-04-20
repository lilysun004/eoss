#!/bin/bash
#SBATCH -J eoss_vit_sgdm
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/vit_sgdm_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/vit_sgdm_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model vit \
    --loss_type mse \
    --no_init True \
    --optimizer_name SGD-Momentum \
    --lr 0.007 \
    --optimizer_params "{'beta': 0.5}" \
    --batch_size 64 \
    --measurement_batch_size_cap 128 \
    --steps 50000 \
    --track_from 40000 \
    --track_until 50000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_vit_sgdm
