#!/bin/bash
# Detached HHE-from-new-data pipeline: rebuild bias-corrected ACE2 HHE frequency
# from the extended (full-JJA, 92-day) inference runs, then replot the boxed
# panels. Launched with setsid+nohup so it survives cluster disconnect.
set -euo pipefail
cd /home/vt55/ace2
# self-redirect all output to the pipeline log (so launcher needs no redirect)
exec > outputs/lag_may/regen_logs/hhe_newdata_pipeline.log 2>&1
PY=/home/vt55/mm/envs/ace2-fresh/bin/python

echo "[$(date)] START regen HHE (force cache + bias) from extended runs"
$PY scripts/hhe_ace2_biascorr.py --force-cache --force-bias
echo "[$(date)] HHE freq rebuilt; replotting boxed panels (fig1 raw + fig2 HHE)"
$PY scripts/extreme_freq_boxed_panels.py
echo "[$(date)] DONE — fig1_raw_extreme_freq_panels.png, fig2_hhe_freq_panels.png"
