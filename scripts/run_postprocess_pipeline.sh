#!/bin/bash
# Full postprocessing pipeline for 1980-2002 JJA lag ensemble.
# Safe to run in screen/nohup — all steps are idempotent (skip if output exists).

set -euo pipefail

PYTHON=/home/vt55/ace2/.conda/envs/ace2/bin/python
SCRIPTS=/home/vt55/ace2/scripts
YEARS="1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002"
LOG_DIR=/home/vt55/ace2/outputs/lag_may/logs
mkdir -p "$LOG_DIR"

echo "=== Pipeline started: $(date) ===" | tee -a "$LOG_DIR/pipeline.log"

echo "--- Step 1: combine_lag_may_daily ---" | tee -a "$LOG_DIR/pipeline.log"
$PYTHON "$SCRIPTS/combine_lag_may_daily.py" --years "$YEARS" \
    2>&1 | tee -a "$LOG_DIR/combine.log"
echo "combine done: $(date)" | tee -a "$LOG_DIR/pipeline.log"

echo "--- Step 2: postprocess_jja_lag ---" | tee -a "$LOG_DIR/pipeline.log"
$PYTHON "$SCRIPTS/postprocess_jja_lag.py" --years "$YEARS" \
    2>&1 | tee -a "$LOG_DIR/postprocess.log"
echo "postprocess done: $(date)" | tee -a "$LOG_DIR/pipeline.log"

echo "--- Step 3: sst_teleconnection_jja ---" | tee -a "$LOG_DIR/pipeline.log"
$PYTHON "$SCRIPTS/sst_teleconnection_jja.py" --years "$YEARS" \
    2>&1 | tee -a "$LOG_DIR/sst.log"
echo "sst done: $(date)" | tee -a "$LOG_DIR/pipeline.log"

echo "=== Pipeline complete: $(date) ===" | tee -a "$LOG_DIR/pipeline.log"
