#!/bin/bash
#SBATCH -J cnn_signfix
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/cnn_signfix_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/cnn_signfix_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

# Re-run of the flagship "strongest bimodality" CNN cell (SGD lr=0.02 b=32, num_data=16384,
# track 30000-40000) with the LOBPCG sign-alignment fix applied (utils/measure.py:
# _run_lobpcg_with_operator now flips eigenvectors to match the warm-start sign).
# Identical config to marc_cnn_sweep_fixed_u_n16384/*SGD_lr0.02_b32* except for the fix
# and a distinct results_subfolder so it doesn't overwrite the original (contaminated) run.
/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn \
    --num_data 16384 \
    --optimizer_name SGD \
    --lr 0.02 \
    --batch_size 32 \
    --steps 40000 \
    --stop_loss None \
    --track_from 30000 \
    --track_until 40000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_cnn_sweep_signfix_n16384
