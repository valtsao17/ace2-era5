#!/bin/bash
#SBATCH --job-name=era5_precip_jja
#SBATCH --partition=chaney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vt55/ace2/outputs/lag_may/regen_logs/era5_precip_jja_%j.log
set -uo pipefail
cd /home/vt55/ace2
echo "[$(date)] host=$(hostname) START ERA5 JJA precip fetch 2002-2016"
/home/vt55/mm/envs/ace2-fresh/bin/python scripts/fetch_era5_precip_jja.py --years 1980-2001
echo "[$(date)] DONE"
