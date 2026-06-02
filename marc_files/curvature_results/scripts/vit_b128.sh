#!/bin/bash
#SBATCH -J curv_vit_b128
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --constraint=h100|h200
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_vit_b128_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_vit_b128_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results

cd /n/home06/mwalden/eoss

# Single ViT cell: SGD lr=0.0035 b=128 150k steps, track 145k–150k.
# Matches marc_files/drift_results/scripts/vit/run_sgd.sh.
/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model vit --dataset cifar10 \
    --measurement_batch_size_cap 1024 \
    --optimizer_name SGD --lr 0.0035 --batch_size 128 \
    --steps 150000 --track_from 145000 --track_until 150000 \
    --stop_loss None \
    --more_freq_measure True \
    --track_stride 10 --top_k_track 30 --cat3_m 10 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_vit
