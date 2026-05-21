#!/bin/bash
#SBATCH -J td_rmsprop
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/td_rmsprop_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/td_rmsprop_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn --dataset cifar10 --num_data 16384 \
    --optimizer_name RMSProp --lr 0.00003 --batch_size 128 \
    --optimizer_params "{'beta2': 0.99}" \
    --steps 75000 --track_from 70000 --track_until 75000 \
    --track_stride 10 --top_k_track 30 --fixed_u True \
    --results_subfolder tangent_drift_cnn_optsweep
