#!/bin/bash
#SBATCH -J sst_smoke
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/sst_smoke_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/sst_smoke_%j.err

# Smoke test: validates SST integration end-to-end on GPU including the
# projection tracker / LOBPCG path on SSTTransformer. Short tracking window.

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --dataset sst2 \
    --model sst_transformer \
    --loss_type mse \
    --num_data 8192 \
    --batch_size 16 \
    --optimizer_name SGD-Momentum \
    --optimizer_params "{'beta': 0.5}" \
    --lr 0.02 \
    --steps 1000 \
    --stop_loss None \
    --track_from 800 \
    --track_until 1000 \
    --track_stride 5 \
    --fixed_u True \
    --results_subfolder smoke_sst
