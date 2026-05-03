#!/bin/bash
#SBATCH -J sst_bimod
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --array=0-9
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/sst_bimod_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/sst_bimod_%a_%j.err

# SST-2 SSTTransformer bimodality sweep.
# Mirrors marcwalden1/edge-of-stochastic-stability/slurm_scripts/sst_transformer.sh:
#   10 LRs, log-spaced so 1/lr spans [16, 773]; tok_emb frozen; expected batch
#   sharpness 2(1-beta)/lr = 1/lr at momentum beta=0.5.
# Adapted to this repo's tracking conventions (track final 10k of 200k steps).

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

#  ID  1/lr    lr
#   0   16   0.06180
#   1   25   0.04000
#   2   38   0.02607
#   3   58   0.01700
#   4   90   0.01108
#   5  138   0.00722
#   6  212   0.00471
#   7  325   0.00307
#   8  500   0.00200
#   9  773   0.00129
LRS=(0.06180 0.04000 0.02607 0.01700 0.01108 0.00722 0.00471 0.00307 0.00200 0.00129)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "SST bimodality task $SLURM_ARRAY_TASK_ID -> SGD-Momentum lr=$LR batch=16 (1/lr target BS)"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --dataset sst2 \
    --model sst_transformer \
    --loss_type mse \
    --num_data 8192 \
    --batch_size 16 \
    --optimizer_name SGD-Momentum \
    --optimizer_params "{'beta': 0.5}" \
    --lr "$LR" \
    --steps 200000 \
    --stop_loss None \
    --track_from 190000 \
    --track_until 200000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder SST_sweep
