#!/usr/bin/env bash
# Orchestrate IC and forcing file generation for 1990-2000, one year at a time.
# Each subprocess is short-lived, so external kill limits don't accumulate.

set -euo pipefail

PY="/home/jovyan/ace2-era5/.conda/envs/ace2/bin/python"
SCRIPT="/home/jovyan/hiro_ace_clean_v4/scripts/make_lag_data_1990_2000.py"
IC_DIR="/home/jovyan/ace2_lag_data/initial_conditions"
FORC_DIR="/home/jovyan/ace2_lag_data/forcing_data_ace2era5"
LOG_DIR="/home/jovyan/hiro_ace_clean_v4/outputs/lag_10yr"
MAX_RETRIES=10

nc_valid() {
    "$PY" -c "
import netCDF4 as nc, sys
try:
    ds = nc.Dataset('$1', 'r')
    n = len(ds.variables)
    ds.close()
    sys.exit(0 if n > 3 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

run_with_retry() {
    local label="$1"; shift
    local out_file="$1"; shift
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if nc_valid "$out_file"; then
            echo "[SKIP] $label (already complete)"
            return 0
        fi
        echo "[RUN ] $label (attempt $((retries+1))/$MAX_RETRIES)"
        "$PY" "$SCRIPT" "$@" >> "${LOG_DIR}/lag_data_prep.log" 2>&1 || true
        if nc_valid "$out_file"; then
            echo "[DONE] $label"
            return 0
        fi
        retries=$((retries + 1))
        echo "[RETRY] $label incomplete, waiting 30s ..."
        sleep 30
    done
    echo "[FAIL] $label after $MAX_RETRIES retries"
    return 1
}

echo "=== IC files (1990-2000) ===" | tee -a "${LOG_DIR}/lag_data_prep.log"
for year in 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000; do
    run_with_retry "IC $year" "${IC_DIR}/ic_lag_${year}1101_25m.nc" --ic-year "$year"
done

echo "=== Forcing files (1991-2001) ===" | tee -a "${LOG_DIR}/lag_data_prep.log"
for year in 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001; do
    run_with_retry "Forcing $year" "${FORC_DIR}/forcing_${year}.nc" --forcing-year "$year"
done

echo "=== All done ===" | tee -a "${LOG_DIR}/lag_data_prep.log"
