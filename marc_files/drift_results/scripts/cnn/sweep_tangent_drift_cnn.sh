#!/bin/bash
#SBATCH -J tangent_cnn
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/tangent_cnn_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/tangent_cnn_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn \
    --dataset cifar10 \
    --num_data 8192 \
    --optimizer_name SGD \
    --lr 0.02 \
    --batch_size 32 \
    --steps 35000 \
    --track_from 30000 \
    --track_until 35000 \
    --track_stride 10 \
    --top_k_track 15 \
    --fixed_u True \
    --results_subfolder tangent_drift_cnn
