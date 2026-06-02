#!/bin/bash
#SBATCH -J curv_mlp_ce_ls
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_ce_ls_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_ce_ls_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Full curvature (Fig 29) run for MLP cross-entropy WITH label smoothing eps=0.1.
# Plain CE never sustained EoS (curvature collapsed at every LR). With eps=0.1 the
# logits are bounded -> finite min -> sustained lmax plateau. The LR probe
# (ce_ls_probe) showed lr=0.03 holds a stable edge plateau (lr*lmax/2 ~= 0.77,
# lmax ~= 50) across steps ~7k-20k, with an early overshoot to 1.82 confirming the
# EoS mechanism engages. Tracking window 15k-20k sits squarely in that plateau.
/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr 0.03 --batch_size 128 \
    --steps 20000 --track_from 15000 --track_until 20000 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_mlp_ce_ls
