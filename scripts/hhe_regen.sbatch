#!/bin/bash
#SBATCH --job-name=hhe_regen
#SBATCH --partition=chaney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/vt55/ace2/outputs/lag_may/regen_logs/hhe_regen_%j.log

# Rebuild bias-corrected ACE2 HHE frequency from the extended (full-JJA, 92-day)
# inference runs, then replot the boxed panels. Runs on a compute node via SLURM
# so it survives login-session disconnect.
set -euo pipefail
cd /home/vt55/ace2
PY=/home/vt55/mm/envs/ace2-fresh/bin/python

echo "[$(date)] host=$(hostname)  START HHE regen (force cache + bias)"
$PY scripts/hhe_ace2_biascorr.py --force-cache --force-bias
echo "[$(date)] HHE freq rebuilt; replotting boxed panels"
$PY scripts/extreme_freq_boxed_panels.py
echo "[$(date)] DONE — fig1_raw_extreme_freq_panels.png + fig2_hhe_freq_panels.png"
