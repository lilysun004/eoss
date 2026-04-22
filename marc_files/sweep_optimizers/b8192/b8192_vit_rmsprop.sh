#!/bin/bash
#SBATCH -J vit_rms_b8192
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/vit_rms_b8192_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/vit_rms_b8192_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model vit \
    --no_init False \
    --measurement_batch_size_cap 1024 \
    --optimizer_name RMSProp \
    --lr 0.00002 \
    --optimizer_params "{'beta2': 0.99}" \
    --batch_size 8192 \
    --steps 100000 \
    --track_from 95000 \
    --track_until 100000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_vit_sweep
