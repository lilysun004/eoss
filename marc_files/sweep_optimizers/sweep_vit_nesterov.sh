#!/bin/bash
#SBATCH -J vit_nest
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --array=0-4
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/vit_nest_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/vit_nest_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

BATCH_SIZES=(8 32 128 1024 8192)
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID → batch size $BATCH"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model vit \
    --no_init False \
    --measurement_batch_size_cap 1024 \
    --optimizer_name SGD-Nesterov \
    --lr 0.007 \
    --optimizer_params "{'beta': 0.5}" \
    --batch_size $BATCH \
    --steps 150000 \
    --track_from 145000 \
    --track_until 150000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_vit_sweep
