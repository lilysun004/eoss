#!/bin/bash
#SBATCH -J eoss_mlp_sgd
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/mlp_sgd_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/mlp_sgd_%j.err

source ~/.bashrc
conda activate eoss

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

python config.py \
    --model mlp \
    --optimizer_name SGD \
    --lr 0.02 \
    --results_subfolder marc_projection_test
