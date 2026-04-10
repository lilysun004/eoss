#!/bin/bash
#SBATCH -J eoss_plot_hist
#SBATCH -p kempner_h100
#SBATCH -A kempner_kdbrantley_lab
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /n/home06/mwalden/eoss/marc_files/logs/plot_hist_%j.out
#SBATCH -e /n/home06/mwalden/eoss/marc_files/logs/plot_hist_%j.err

source ~/.bashrc
conda activate eoss

export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results

cd /n/home06/mwalden/eoss

for run_folder in "$RESULTS/marc_projection_test"/*/; do
    npz="$run_folder/projections.npz"
    if [ -f "$npz" ]; then
        echo "Plotting $run_folder"
        python plot_histograms.py "$run_folder"
    else
        echo "Skipping $run_folder (no projections.npz)"
    fi
done
