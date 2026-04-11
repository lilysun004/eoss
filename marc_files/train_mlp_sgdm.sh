#!/bin/bash
#SBATCH -J eoss_mlp_sgdm
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/mlp_sgdm_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/mlp_sgdm_%j.err

source ~/.bashrc
conda activate eoss || source ~/.conda/envs/eoss/etc/conda/activate.d/*.sh 2>/dev/null

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp \
    --optimizer_name SGD-Momentum \
    --lr 0.002 \
    --optimizer_params "{'beta': 0.9}" \
    --results_subfolder marc_projection_test
