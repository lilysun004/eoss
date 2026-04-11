#!/bin/bash
#SBATCH -J eoss_plot_sweep
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/plot_sweep_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/plot_sweep_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python marc_files/plot_batch_sweep.py
