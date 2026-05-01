#!/bin/bash
#SBATCH -J mlp_b8192
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 16:00:00
#SBATCH --array=0-5
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/mlp_b8192_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/mlp_b8192_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

case "$SLURM_ARRAY_TASK_ID" in
  0) OPT="SGD";          LR="0.02";    PARAMS="";                              STEPS="40000"; TRACK_FROM="35000"; TRACK_UNTIL="40000" ;;
  1) OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 STEPS="40000"; TRACK_FROM="35000"; TRACK_UNTIL="40000" ;;
  2) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 STEPS="40000"; TRACK_FROM="35000"; TRACK_UNTIL="40000" ;;
  3) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; STEPS="80000"; TRACK_FROM="75000"; TRACK_UNTIL="80000" ;;
  4) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               STEPS="80000"; TRACK_FROM="75000"; TRACK_UNTIL="80000" ;;
  5) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             STEPS="80000"; TRACK_FROM="75000"; TRACK_UNTIL="80000" ;;
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "MLP b8192 task $SLURM_ARRAY_TASK_ID -> $OPT"

CMD=(/n/home06/mwalden/.conda/envs/eoss/bin/python config.py
    --model mlp
    --num_data 8192
    --optimizer_name "$OPT"
    --lr "$LR"
    --batch_size 8192
    --steps "$STEPS"
    --stop_loss None
    --track_from "$TRACK_FROM"
    --track_until "$TRACK_UNTIL"
    --more_freq_measure True
    --fixed_u True
    --results_subfolder MLP_sweep)

if [ -n "$PARAMS" ]; then
    CMD+=(--optimizer_params "$PARAMS")
fi

"${CMD[@]}"
