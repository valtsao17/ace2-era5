#!/bin/bash
#SBATCH --job-name=monthly_raw_rc
#SBATCH --partition=chaney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/vt55/ace2/outputs/lag_may/regen_logs/monthly_raw_rc_%j.log

# Monthly RAW heat-extreme freq vs Kendall tau vs modified-Kendall z (Jun/Jul/Aug),
# built from the fresh 92-day caches (no run re-reads). CONUS, all/land/sea legend.
set -euo pipefail
cd /home/vt55/ace2
echo "[$(date)] host=$(hostname)  START monthly raw freq-vs-rankcorr"
/home/vt55/mm/envs/ace2-fresh/bin/python scripts/monthly_raw_freq_rankcorr.py
echo "[$(date)] DONE — freq_vs_rankcorr_conus_{jun,jul,aug}.png"
