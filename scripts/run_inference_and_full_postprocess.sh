#!/bin/bash
# Continues inference from 2003-2016 (resumes partial 2003 via --skip-existing),
# then combines those years, then re-runs the FULL 1980-2016 postprocess pipeline
# (thresholds recomputed with --force since all years are now available).

set -euo pipefail

PYTHON=/home/vt55/ace2/.conda/envs/ace2/bin/python
ROOT=/home/vt55/ace2
SCRIPTS=$ROOT/scripts
LOG_DIR=$ROOT/outputs/lag_may/logs
mkdir -p "$LOG_DIR"

NEW_YEARS="2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016"
ALL_YEARS="1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016"

echo "=== Inference+full-postprocess pipeline started: $(date) ===" | tee -a "$LOG_DIR/pipeline2.log"

echo "--- Step 1: inference 2003-2016 (skip-existing, 4 GPUs) ---" | tee -a "$LOG_DIR/pipeline2.log"
$PYTHON "$SCRIPTS/run_lag_inference_may.py" \
    --years "$NEW_YEARS" \
    --n-gpus 4 \
    --skip-existing \
    2>&1 | tee -a "$LOG_DIR/inference_2003_2016.log"
echo "inference done: $(date)" | tee -a "$LOG_DIR/pipeline2.log"

echo "--- Step 2: combine 2003-2016 ---" | tee -a "$LOG_DIR/pipeline2.log"
$PYTHON "$SCRIPTS/combine_lag_may_daily.py" --years "$NEW_YEARS" \
    2>&1 | tee -a "$LOG_DIR/combine_2003_2016.log"
echo "combine done: $(date)" | tee -a "$LOG_DIR/pipeline2.log"

echo "--- Step 3: full postprocess 1980-2016 (--force to recompute LOO with all years) ---" | tee -a "$LOG_DIR/pipeline2.log"
$PYTHON "$SCRIPTS/postprocess_jja_lag.py" --years "$ALL_YEARS" --force \
    2>&1 | tee -a "$LOG_DIR/postprocess_full.log"
echo "postprocess done: $(date)" | tee -a "$LOG_DIR/pipeline2.log"

echo "--- Step 4: sst teleconnection 1980-2016 ---" | tee -a "$LOG_DIR/pipeline2.log"
$PYTHON "$SCRIPTS/sst_teleconnection_jja.py" --years "$ALL_YEARS" --force \
    2>&1 | tee -a "$LOG_DIR/sst_full.log"
echo "sst done: $(date)" | tee -a "$LOG_DIR/pipeline2.log"

echo "=== Pipeline complete: $(date) ===" | tee -a "$LOG_DIR/pipeline2.log"
