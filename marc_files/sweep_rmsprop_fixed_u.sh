#!/bin/bash
#SBATCH -J eoss_rmsprop_fu
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 16:00:00
#SBATCH --array=0-4
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/rmsprop_fu_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/rmsprop_fu_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

BATCH_SIZES=(8 32 128 1024 8192)
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID → batch size $BATCH"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp \
    --optimizer_name RMSProp \
    --lr 0.00003 \
    --optimizer_params "{'beta2': 0.99}" \
    --batch_size $BATCH \
    --steps 80000 \
    --track_from 70000 \
    --track_until 80000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder marc_optimizer_sweep_fixed_u
