#!/bin/bash
#SBATCH -J cnn_sgd
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --array=0-3
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/cnn_sgd_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/cnn_sgd_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

BATCH_SIZES=(8 32 128 1024)
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID -> batch size $BATCH"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model cnn \
    --optimizer_name SGD \
    --lr 0.02 \
    --batch_size $BATCH \
    --steps 40000 \
    --track_from 30000 \
    --track_until 40000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_cnn_sweep_fixed_u
