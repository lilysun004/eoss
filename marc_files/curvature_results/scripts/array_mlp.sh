#!/bin/bash
#SBATCH -J curv_mlp_arr
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --array=0-28
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_arr_%a_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_arr_%a_%j.err

source ~/.bashrc
conda activate eoss || true

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results

cd /n/home06/mwalden/eoss

# 29 cells. Skipped: MLP SGD b=128 lr=0.005 (already done at
# curvature_results/curvature_failure_mlp/20260522_2025_52_SGD_lr0.005_b128).
#
# LR convention:
#   - SGD lr=0.02 for b∈{8,32,1024}, lr=0.005 for b=8192 (per user note;
#     b=128 cell at lr=0.005 already done so omitted from array)
#   - Other optimizers: their drift_results lr, constant across batches
#
# Steps / track-window per optimizer:
#   - SGD / SGD-Momentum / SGD-Nesterov: 40k steps, track 35k–40k
#   - Adam / RMSProp / Muon: 80k steps, track 75k–80k
case "$SLURM_ARRAY_TASK_ID" in
  # SGD (3 cells: b=128 done)
  0)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=8;    STEPS=40000; TF=35000; TU=40000 ;;
  1)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=32;   STEPS=40000; TF=35000; TU=40000 ;;
  2)  OPT="SGD";          LR="0.02";    PARAMS="";                              BATCH=1024; STEPS=40000; TF=35000; TU=40000 ;;
  3)  OPT="SGD";          LR="0.005";   PARAMS="";                              BATCH=8192; STEPS=40000; TF=35000; TU=40000 ;;
  # SGD-Momentum (5 cells)
  4)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8;    STEPS=40000; TF=35000; TU=40000 ;;
  5)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=32;   STEPS=40000; TF=35000; TU=40000 ;;
  6)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=128;  STEPS=40000; TF=35000; TU=40000 ;;
  7)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=1024; STEPS=40000; TF=35000; TU=40000 ;;
  8)  OPT="SGD-Momentum"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8192; STEPS=40000; TF=35000; TU=40000 ;;
  # SGD-Nesterov (5 cells)
  9)  OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8;    STEPS=40000; TF=35000; TU=40000 ;;
  10) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=32;   STEPS=40000; TF=35000; TU=40000 ;;
  11) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=128;  STEPS=40000; TF=35000; TU=40000 ;;
  12) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=1024; STEPS=40000; TF=35000; TU=40000 ;;
  13) OPT="SGD-Nesterov"; LR="0.002";   PARAMS="{'beta': 0.9}";                 BATCH=8192; STEPS=40000; TF=35000; TU=40000 ;;
  # Adam (5 cells)
  14) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=8;    STEPS=80000; TF=75000; TU=80000 ;;
  15) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=32;   STEPS=80000; TF=75000; TU=80000 ;;
  16) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=128;  STEPS=80000; TF=75000; TU=80000 ;;
  17) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=1024; STEPS=80000; TF=75000; TU=80000 ;;
  18) OPT="Adam";         LR="0.00003"; PARAMS="{'beta1': 0.9, 'beta2': 0.99}"; BATCH=8192; STEPS=80000; TF=75000; TU=80000 ;;
  # RMSProp (5 cells)
  19) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=8;    STEPS=80000; TF=75000; TU=80000 ;;
  20) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=32;   STEPS=80000; TF=75000; TU=80000 ;;
  21) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=128;  STEPS=80000; TF=75000; TU=80000 ;;
  22) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=1024; STEPS=80000; TF=75000; TU=80000 ;;
  23) OPT="RMSProp";      LR="0.00003"; PARAMS="{'beta2': 0.99}";               BATCH=8192; STEPS=80000; TF=75000; TU=80000 ;;
  # Muon (5 cells)
  24) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=8;    STEPS=80000; TF=75000; TU=80000 ;;
  25) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=32;   STEPS=80000; TF=75000; TU=80000 ;;
  26) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=128;  STEPS=80000; TF=75000; TU=80000 ;;
  27) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=1024; STEPS=80000; TF=75000; TU=80000 ;;
  28) OPT="Muon";         LR="0.001";   PARAMS="{'momentum': 0.9}";             BATCH=8192; STEPS=80000; TF=75000; TU=80000 ;;
  *)  echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "curv_mlp[$SLURM_ARRAY_TASK_ID] -> $OPT lr=$LR b=$BATCH steps=$STEPS"

CMD=(/n/home06/mwalden/.conda/envs/eoss/bin/python config.py
    --model mlp --dataset cifar10 --num_data 8192
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
    --results_subfolder curvature_failure_mlp)

if [ -n "$PARAMS" ]; then
    CMD+=(--optimizer_params "$PARAMS")
fi

"${CMD[@]}"
