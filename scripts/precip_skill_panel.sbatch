#!/bin/bash
#SBATCH --job-name=precip_skill_panel
#SBATCH --partition=chaney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/vt55/ace2/outputs/lag_may/regen_logs/precip_skill_panel_%j.log
set -uo pipefail
cd /home/vt55/ace2
echo "[$(date)] host=$(hostname) START precip skill panel"
/home/vt55/mm/envs/ace2-fresh/bin/python scripts/precip_skill_panel.py --years 2002-2016
echo "[$(date)] DONE"
