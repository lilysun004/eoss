#!/bin/bash
#SBATCH -J ce_ls_probe
#SBATCH -p kempner_requeue
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/ce_ls_probe_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/ce_ls_probe_%A_%a.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Diagnostic LR sweep for MLP CE *with label smoothing eps=0.1*. Plain CE never
# sustained EoS at any LR (grazed lr*lmax/2~=1.0 for ~1-2k steps then curvature
# collapsed). Label smoothing bounds the logits -> finite min -> sustained lmax
# plateau, the same reason MSE sits at EoS. These probes re-map where lmax pins at
# 2/lr now that the collapse is suppressed. No curvature scan, degenerate tracking
# window => ~2-3 min each; lmax logged every ~256 steps regardless. Winner gets the
# full curvature run with the window placed on the plateau.
LRS=(0.01 0.02 0.03 0.05)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce --label_smoothing 0.1 \
    --optimizer_name SGD --lr $LR --batch_size 128 \
    --steps 20000 --track_from 19999 --track_until 20000 \
    --results_subfolder ce_ls_probe_lr${LR}
