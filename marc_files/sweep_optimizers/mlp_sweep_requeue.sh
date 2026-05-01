#!/bin/bash
#SBATCH -J mlp_sweep
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 16:00:00
#SBATCH --array=0-23
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/mlp_sweep_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/mlp_sweep_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

case "$SLURM_ARRAY_TASK_ID" in
  0)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="8";    STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  1)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="32";   STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  2)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="128";  STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  3)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH="1024"; STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  4)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="8";    STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  5)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="32";   STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  6)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="128";  STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  7)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="1024"; STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  8)  OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="8";    STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  9)  OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="32";   STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  10) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="128";  STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  11) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH="1024"; STEPS="40000"; TRACK_FROM="30000"; TRACK_UNTIL="40000" ;;
  12) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="8";    STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  13) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="32";   STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  14) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="128";  STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  15) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH="1024"; STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  16) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="8";    STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  17) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="32";   STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  18) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="128";  STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  19) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH="1024"; STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  20) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="8";    STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  21) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="32";   STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  22) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="128";  STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  23) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH="1024"; STEPS="80000"; TRACK_FROM="70000"; TRACK_UNTIL="80000" ;;
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "MLP sweep task $SLURM_ARRAY_TASK_ID -> $OPT batch size $BATCH"

CMD=(/n/home06/mwalden/.conda/envs/eoss/bin/python config.py
    --model mlp
    --num_data 8192
    --optimizer_name "$OPT"
    --lr "$LR"
    --batch_size "$BATCH"
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
