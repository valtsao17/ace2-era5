#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu_node
#SBATCH --job-name="enso_tmp2m_corr"
#SBATCH --output="/home/vt55/ace2/slurm_logs/%x_%j.log"

# ENSO/Nino3.4 2m-temperature correlation maps for ERA5 vs ACE2.
# This uses ACE2 TMP2m by default, avoiding the prescribed surface_temperature
# field that made the old SST maps identical.
#
# Submit, using existing ACE2 runs with TMP2m already saved:
#   sbatch scripts/run_enso_sst_correlation.sbatch
#
# If the run files do not contain TMP2m, rerun/resume inference:
#   RUN_INFERENCE=1 sbatch scripts/run_enso_sst_correlation.sbatch

set -euo pipefail

ROOT=${ROOT:-/home/vt55/ace2}
PY=${PY:-/home/vt55/mm/envs/ace2-fresh/bin/python}
YEARS=${YEARS:-1980:2016}
RUNS_ROOT=${RUNS_ROOT:-$ROOT/outputs/lag_may/runs}
FORCING_DIR=${FORCING_DIR:-$ROOT/data/lag_data/forcing_data_ace2era5}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/lag_may/enso_tmp2m_correlation}
RUN_INFERENCE=${RUN_INFERENCE:-0}
N_GPUS=${N_GPUS:-4}
INDEX_SOURCE=${INDEX_SOURCE:-self}
ERA5_SST_VAR=${ERA5_SST_VAR:-TMP2m}
ACE2_SST_VAR=${ACE2_SST_VAR:-TMP2m}
WRITER_NAMES=${WRITER_NAMES:-TMP2m,Q2m,PRESsfc,UGRD10m,VGRD10m}

cd "$ROOT"
mkdir -p "$ROOT/slurm_logs" "$ROOT/tmp/matplotlib" "$OUT_DIR"

echo "[$(date)] START ENSO TMP2m correlation"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "YEARS=$YEARS"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "FORCING_DIR=$FORCING_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "INDEX_SOURCE=$INDEX_SOURCE"
echo "ERA5_SST_VAR=$ERA5_SST_VAR"
echo "ACE2_SST_VAR=$ACE2_SST_VAR"

if [ "$RUN_INFERENCE" = "1" ]; then
  echo "=== Rerun/resume ACE2 May inference with TMP2m saved ==="
  "$PY" scripts/run_lag_inference_may.py \
    --root "$ROOT" \
    --years "$YEARS" \
    --n-steps 500 \
    --n-gpus "$N_GPUS" \
    --forward-steps-in-memory 40 \
    --skip-existing \
    --writer-names "$WRITER_NAMES"
else
  echo "=== RUN_INFERENCE=0, using existing ACE2 member outputs ==="
fi

ERA5_ARGS=(--era5-forcing-dir "$FORCING_DIR" --era5-sst-var "$ERA5_SST_VAR")
if [ -n "${ERA5_SST_NC:-}" ]; then
  ERA5_ARGS=(--era5-sst-nc "$ERA5_SST_NC" --era5-sst-var "$ERA5_SST_VAR")
fi

MPLCONFIGDIR="$ROOT/tmp/matplotlib" "$PY" scripts/enso_sst_correlation_map.py \
  --root "$ROOT" \
  --years "$YEARS" \
  "${ERA5_ARGS[@]}" \
  --ace2-runs-root "$RUNS_ROOT" \
  --ace2-sst-var "$ACE2_SST_VAR" \
  --mask-source-forcing-dir "$FORCING_DIR" \
  --no-ocean-mask \
  --index-source "$INDEX_SOURCE" \
  --output-dir "$OUT_DIR"

echo "[$(date)] DONE"
echo "Outputs: $OUT_DIR"
