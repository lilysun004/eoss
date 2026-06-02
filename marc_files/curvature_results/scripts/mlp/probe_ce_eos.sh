#!/bin/bash
#SBATCH -J ce_eos_probe
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/ce_eos_probe_%A_%a.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/ce_eos_probe_%A_%a.err

source ~/.bashrc
conda activate eoss || true
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/home06/mwalden/eoss/marc_files/curvature_results
cd /n/home06/mwalden/eoss

# Diagnostic LR sweep for the MLP CE cell. At lr=0.005 CE never reached EoS
# (peaked at lr*lmax/2 = 0.58, then curvature collapsed). These probes raise the
# LR so the stability threshold 2/lr drops into the curvature CE can sustain.
# No curvature scan and a degenerate tracking window (track_from==track_until-1)
# so these are FAST (~2-3 min) — lmax is logged every ~256 steps regardless, which
# is all we need to locate where each LR pins at EoS (lr*lmax/2 ~= 1.0). The winner
# gets a full curvature run with the window placed correctly afterward.
LRS=(0.01 0.02 0.03 0.04)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

/n/home06/mwalden/.conda/envs/eoss/bin/python config.py \
    --model mlp --dataset cifar10 --num_data 8192 \
    --loss_type ce \
    --optimizer_name SGD --lr $LR --batch_size 128 \
    --steps 20000 --track_from 19999 --track_until 20000 \
    --results_subfolder ce_eos_probe_lr${LR}
