#!/bin/bash
# Waits for ace2_inference2016 to finish, then reruns postprocess + SST
# with updated plotting code (uses cached thresholds/ERA5 data — no --force).

set -euo pipefail

PYTHON=/home/vt55/ace2/.conda/envs/ace2/bin/python
SCRIPTS=/home/vt55/ace2/scripts
LOG_DIR=/home/vt55/ace2/outputs/lag_may/logs
ALL_YEARS="1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016"

echo "Waiting for ace2_inference2016 screen to finish..." | tee -a "$LOG_DIR/replot.log"
while screen -ls | grep -q "ace2_inference2016"; do
    sleep 60
done
echo "Pipeline done. Starting replot: $(date)" | tee -a "$LOG_DIR/replot.log"

echo "--- Replot: postprocess_jja_lag ---" | tee -a "$LOG_DIR/replot.log"
$PYTHON "$SCRIPTS/postprocess_jja_lag.py" --years "$ALL_YEARS" \
    2>&1 | tee -a "$LOG_DIR/replot_postprocess.log"
echo "postprocess replot done: $(date)" | tee -a "$LOG_DIR/replot.log"

echo "--- Replot: sst_teleconnection_jja ---" | tee -a "$LOG_DIR/replot.log"
$PYTHON "$SCRIPTS/sst_teleconnection_jja.py" --years "$ALL_YEARS" \
    2>&1 | tee -a "$LOG_DIR/replot_sst.log"
echo "sst replot done: $(date)" | tee -a "$LOG_DIR/replot.log"

echo "=== Replot complete: $(date) ===" | tee -a "$LOG_DIR/replot.log"
