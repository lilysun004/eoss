#!/bin/bash
#SBATCH -J sf08_sst_transformer_SGD_b1024
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/sf08_sst_transformer_SGD_b1024_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/sf08_sst_transformer_SGD_b1024_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model sst_transformer \
    --dataset sst2 \
    --num_data 8192 \
    --optimizer_name SGD \
    --lr 0.02 \
    --batch_size 1024 \
    --optimizer_params "{}" \
    --loss_type mse \
    --steps 50000 \
    --track_from 40000 \
    --track_until 50000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder SST_opt_batch_sweep_signfix
