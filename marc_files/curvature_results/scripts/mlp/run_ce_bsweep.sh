#!/bin/bash
#SBATCH -J curv_ce_bsweep
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --array=0-4
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_ce_bsweep_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_ce_bsweep_%A_%a.err
source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Full curvature (Fig 29 + stochastic along-step) sweep over batch size for MLP CE,
# eps=0.1, each at its EoSS lr (batch_sharpness ratio ~= 1.0, found by probe matrix).
# Demonstrates the EoSS->EoS crossover: b=8 strongly stochastic (lmax_ratio 0.08,
# bsharp 1.04), b=128 crossover (lmax~=bsharp~=0.88), b>=1024 deterministic EoS
# (lmax~=bsharp~=1.0). The new along-step scan uses the per-batch Hessian H_B along
# the realized step direction (the EoSS instability axis); compare to the along-u
# scan (full-batch top eigvec) — they should be decoupled at small b, aligned at large b.
case "$SLURM_ARRAY_TASK_ID" in
  0) B=8;    LR=0.03 ;;
  1) B=32;   LR=0.10 ;;
  2) B=128;  LR=0.18 ;;
  3) B=1024; LR=0.12 ;;
  4) B=8192; LR=0.15 ;;
esac

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr $LR --batch_size $B \
    --steps 20000 --track_from 15000 --track_until 20000 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_mlp_ce_bsweep
