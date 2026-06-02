#!/bin/bash
#SBATCH -J curv_cnn_arr
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH --array=0-28
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_cnn_arr_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_cnn_arr_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results

cd /n/home06/mwalden/eoss

# 29 cells. Skipped: CNN SGD b=32 lr=0.02 num_data=16384 (already done at
# curvature_results/curvature_failure_cnn/20260523_1547_08_SGD_lr0.02_b32).
#
# All CNN runs use num_data=16384 (per CLAUDE.md convention — needed for
# the network to remain at EoS through the tracking window).
#
# Each optimizer uses its drift_results LR, constant across batches:
#   SGD lr=0.02; SGD-Momentum/Nesterov lr=0.002; Adam/RMSProp lr=0.00003;
#   Muon lr=0.001
#
# Steps / track-window per optimizer:
#   - SGD / SGD-Momentum / SGD-Nesterov: 35k steps, track 30k–35k
#   - Adam / RMSProp / Muon: 75k steps, track 70k–75k
case "$SLURM_ARRAY_TASK_ID" in
  # SGD (4 cells: b=32 done)
  0)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=8;    STEPS=35000; TF=30000; TU=35000 ;;
  1)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=128;  STEPS=35000; TF=30000; TU=35000 ;;
  2)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=1024; STEPS=35000; TF=30000; TU=35000 ;;
  3)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=8192; STEPS=35000; TF=30000; TU=35000 ;;
  # SGD-Momentum (5 cells)
  4)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8;    STEPS=35000; TF=30000; TU=35000 ;;
  5)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=32;   STEPS=35000; TF=30000; TU=35000 ;;
  6)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=128;  STEPS=35000; TF=30000; TU=35000 ;;
  7)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=1024; STEPS=35000; TF=30000; TU=35000 ;;
  8)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8192; STEPS=35000; TF=30000; TU=35000 ;;
  # SGD-Nesterov (5 cells)
  9)  OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8;    STEPS=35000; TF=30000; TU=35000 ;;
  10) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=32;   STEPS=35000; TF=30000; TU=35000 ;;
  11) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=128;  STEPS=35000; TF=30000; TU=35000 ;;
  12) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=1024; STEPS=35000; TF=30000; TU=35000 ;;
  13) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8192; STEPS=35000; TF=30000; TU=35000 ;;
  # Adam (5 cells)
  14) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=8;    STEPS=75000; TF=70000; TU=75000 ;;
  15) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=32;   STEPS=75000; TF=70000; TU=75000 ;;
  16) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=128;  STEPS=75000; TF=70000; TU=75000 ;;
  17) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=1024; STEPS=75000; TF=70000; TU=75000 ;;
  18) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=8192; STEPS=75000; TF=70000; TU=75000 ;;
  # RMSProp (5 cells)
  19) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=8;    STEPS=75000; TF=70000; TU=75000 ;;
  20) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=32;   STEPS=75000; TF=70000; TU=75000 ;;
  21) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=128;  STEPS=75000; TF=70000; TU=75000 ;;
  22) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=1024; STEPS=75000; TF=70000; TU=75000 ;;
  23) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=8192; STEPS=75000; TF=70000; TU=75000 ;;
  # Muon (5 cells)
  24) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=8;    STEPS=75000; TF=70000; TU=75000 ;;
  25) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=32;   STEPS=75000; TF=70000; TU=75000 ;;
  26) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=128;  STEPS=75000; TF=70000; TU=75000 ;;
  27) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=1024; STEPS=75000; TF=70000; TU=75000 ;;
  28) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=8192; STEPS=75000; TF=70000; TU=75000 ;;
  *)  echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "curv_cnn[$SLURM_ARRAY_TASK_ID] -> $OPT lr=$LR b=$BATCH steps=$STEPS"

CMD=(/n/home06/mwalden/.conda/envs/eoss/bin/python config.py
    --model cnn --dataset cifar10 --num_data 16384
    --optimizer_name "$OPT"
    --lr "$LR"
    --batch_size "$BATCH"
    --steps "$STEPS"
    --stop_loss None
    --track_from "$TF"
    --track_until "$TU"
    --more_freq_measure True
    --track_stride 10 --top_k_track 30 --cat3_m 10 --fixed_u True
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50
    --results_subfolder curvature_failure_cnn)

if [ -n "$PARAMS" ]; then
    CMD+=(--optimizer_params "$PARAMS")
fi

"${CMD[@]}"
