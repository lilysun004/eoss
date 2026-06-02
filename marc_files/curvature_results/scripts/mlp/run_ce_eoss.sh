#!/bin/bash
#SBATCH -J curv_ce_eoss
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH --array=0-1
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_ce_eoss_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_ce_eoss_%A_%a.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Full curvature (Fig 29) runs for MLP CE at the EoSS config found via batch
# sweep: eps=0.1, b=8 -> lr*batch_sharpness/2 = 1.06 (AT EoSS), lmax ratio only
# 0.08 (textbook EoSS: batch sharpness pinned at 2/eta, full-batch lmax far below).
# EoSS proximity judged by BATCH SHARPNESS, not lmax (see memory). Two LRs to
# bracket. Caveat: at b=8 the full-batch top eigvec is low-curvature, so the
# period-2 signal may not live along u — check the top-eigvec projections.
LRS=(0.03 0.05)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr $LR --batch_size 8 \
    --steps 20000 --track_from 15000 --track_until 20000 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_mlp_ce_eoss
