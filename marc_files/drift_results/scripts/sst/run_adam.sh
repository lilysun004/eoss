#!/bin/bash
#SBATCH -J tds_adam
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/tds_adam_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/tds_adam_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/home06/mwalden/eoss

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --dataset sst2 --model sst_transformer --loss_type mse \
    --num_data 8192 \
    --optimizer_name Adam --lr 0.00003 --batch_size 128 \
    --optimizer_params "{'beta1': 0.9, 'beta2': 0.99}" \
    --steps 50000 --track_from 40000 --track_until 50000 \
    --stop_loss None \
    --more_freq_measure True \
    --track_stride 10 --top_k_track 30 --cat3_m 10 --fixed_u True \
    --results_subfolder tangent_drift_sst_optsweep
