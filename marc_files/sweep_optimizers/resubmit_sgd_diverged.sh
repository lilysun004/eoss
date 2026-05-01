#!/bin/bash
#SBATCH -J sgd_resub
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --array=0-1
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/sgd_resub_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/sgd_resub_%a_%j.err

# Re-run the two SGD cells from MLP_sweep that diverged with lr=0.02.
# Lower lr to 0.005 (kept other settings identical so the data fits the existing sweep).

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

case "$SLURM_ARRAY_TASK_ID" in
  0) BATCH="128"  ;;
  1) BATCH="8192" ;;
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "SGD resubmit task $SLURM_ARRAY_TASK_ID -> SGD lr=0.005 batch=$BATCH"

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp \
    --num_data 8192 \
    --optimizer_name SGD \
    --lr 0.005 \
    --batch_size "$BATCH" \
    --steps 40000 \
    --stop_loss None \
    --track_from 30000 \
    --track_until 40000 \
    --more_freq_measure True \
    --fixed_u True \
    --results_subfolder MLP_sweep
