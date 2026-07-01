#!/bin/bash
#SBATCH --job-name=hhe_metrics
#SBATCH --partition=chaney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vt55/ace2/outputs/lag_may/regen_logs/hhe_metrics_%j.log

# Regenerate the TRUE-HHE metric figures on the new 92-day data, with the
# all/land/sea score breakdown. Chained (afterok) on the HHE-freq regen job so
# it sees the rebuilt jja_hi_freq_ace2.nc, bias_fields.nc and ace2_daily_cache.
# (Raw side intentionally left on existing data per user choice.)
set -euo pipefail
cd /home/vt55/ace2
PY=/home/vt55/mm/envs/ace2-fresh/bin/python

echo "[$(date)] host=$(hostname)  HHE seasonal skill (tau/tau_p) from new freq"
$PY scripts/hhe_seasonal_skill.py
echo "[$(date)] HHE 2x2 skill panel (recompute mod-z + day-level precision/recall)"
$PY scripts/hhe_skill_panel.py
echo "[$(date)] HHE 3-panel: freq vs rank-correlation"
$PY scripts/hhe_freq_vs_rankcorr_panel.py
echo "[$(date)] DONE — skill_panel_combined_hhe.png + hhe_freq_vs_rankcorr_conus.png"
