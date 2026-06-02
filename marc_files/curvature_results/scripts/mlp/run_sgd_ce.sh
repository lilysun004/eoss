#!/bin/bash
#SBATCH -J curv_mlp_ce
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 02:30:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_ce_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/curv_mlp_ce_%j.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Cross-entropy variant of the MLP curvature cell. Identical hyperparameters to
# run_sgd.sh (SGD lr=0.005 b=128, track 35k-40k) — only the loss changes, so the
# MSE and CE runs are directly comparable. CE has no finite interpolating minimum
# (logits diverge, GN curvature collapses as the model gets confident), so if
# lambda_max has already declined by step 35k the tracking window will need to be
# moved earlier on the next run.
/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce \
    --optimizer_name SGD --lr 0.005 --batch_size 128 \
    --steps 40000 --track_from 35000 --track_until 40000 \
    --track_stride 10 --top_k_track 5 --fixed_u True \
    --curv_n_alphas 13 --curv_n_betas 9 --curv_beta_scale 2.0 --curv_every 50 \
    --results_subfolder curvature_failure_mlp_ce
