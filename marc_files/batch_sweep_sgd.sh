#!/bin/bash
#SBATCH -J eoss_bsweep
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH --array=0-31
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/bsweep_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/bsweep_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

BATCH_SIZES=(2 3 4 5 6 7 8 11 13 16 19 22 27 32 38 45 54 64 90 128 180 256 362 512 640 1024 1448 2048 2896 4096 5792 8192)
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID → batch size $BATCH"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp \
    --optimizer_name SGD \
    --lr 0.02 \
    --batch_size $BATCH \
    --steps 40000 \
    --results_subfolder marc_batch_sweep
