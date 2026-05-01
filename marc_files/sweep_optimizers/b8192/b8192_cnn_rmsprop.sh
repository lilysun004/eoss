#!/bin/bash
#SBATCH -J cnn_rms_b8192
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/cnn_rms_b8192_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/cnn_rms_b8192_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn \
    --optimizer_name RMSProp \
    --lr 0.00003 \
    --optimizer_params "{'beta2': 0.99}" \
    --batch_size 8192 \
    --steps 80000 \
    --track_from 70000 \
    --track_until 80000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_cnn_sweep_fixed_u
