#!/bin/bash
#SBATCH -J sf02_mlp_Muon_b128
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/sf02_mlp_Muon_b128_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/sf02_mlp_Muon_b128_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp \
    --dataset cifar10 \
    --num_data 8192 \
    --optimizer_name Muon \
    --lr 0.001 \
    --batch_size 128 \
    --optimizer_params "{'momentum': 0.9}" \
    --loss_type mse \
    --steps 80000 \
    --track_from 70000 \
    --track_until 80000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder MLP_sweep_signfix
