#!/bin/bash
# Full DJF postprocess pipeline: combine → seasonal skill plot
# Runs after all inference (1981-2015) is complete.
set -e

PYTHON=/home/vt55/mm/envs/ace2-fresh/bin/python3
ROOT=/home/vt55/ace2
LOG=$ROOT/logs/djf_postprocess_pipeline.log

echo "=== DJF postprocess pipeline started $(date) ===" | tee -a $LOG

echo "--- Step 1: combine_lag_nov_djf.py (all years 1981-2015) ---" | tee -a $LOG
$PYTHON $ROOT/scripts/combine_lag_nov_djf.py 2>&1 | tee -a $LOG

echo "--- Step 2: seasonal_djf_skill.py (fetch ERA5, compute tau, plot) ---" | tee -a $LOG
$PYTHON $ROOT/scripts/seasonal_djf_skill.py 2>&1 | tee -a $LOG

echo "=== Pipeline complete $(date) ===" | tee -a $LOG
