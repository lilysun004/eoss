#!/bin/bash
#SBATCH -J curv_mlp_ce_edge
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_ce_edge_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_ce_edge_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Full curvature (Fig 29) run for MLP CE + label smoothing, at the best edge config
# found by the probes: eps=0.1, lr=0.07. Earlier eps=0.1/lr=0.03 sat sub-critically
# (ratio 0.71 -> no period-2, weak failure signal). The edge probe showed the
# ratio-vs-lr curve peaks near lr=0.07 (sustained lr*lmax/2 ~= 0.88, lmax ~= 25);
# eps=0.05 was WORSE (0.56) and lr=0.10 overshot then fell back. Tracking window
# 15k-20k sits in the sustained 0.88 plateau where period-2 should be present.
/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr 0.07 --batch_size 128 \
    --steps 20000 --track_from 15000 --track_until 20000 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_mlp_ce_edge
