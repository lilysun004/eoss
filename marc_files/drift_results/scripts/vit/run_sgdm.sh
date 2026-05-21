#!/bin/bash
#SBATCH -J tdv_sgdm
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/tdv_sgdm_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/tdv_sgdm_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model vit --dataset cifar10 \
    --measurement_batch_size_cap 1024 \
    --optimizer_name SGD-Momentum --lr 0.007 --batch_size 128 \
    --optimizer_params "{'beta': 0.5}" \
    --steps 150000 --track_from 145000 --track_until 150000 \
    --more_freq_measure True \
    --track_stride 10 --top_k_track 30 --cat3_m 10 --fixed_u True \
    --results_subfolder tangent_drift_vit_optsweep
