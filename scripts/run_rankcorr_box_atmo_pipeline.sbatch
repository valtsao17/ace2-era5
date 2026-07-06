#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu_node
#SBATCH --exclude=node1,node2,node3,node4,node5,node6,node7,node8,node9,node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20
#SBATCH --exclusive
#SBATCH --job-name="rankcorr_atmo"
#SBATCH --output="/home/vt55/ace2/slurm_logs/%x_%j.log"

# Re-run May ACE2 inference with the compact atmospheric-variable writer list
# needed for Figure-7-style mechanism maps, using 4 GPUs in parallel. Then
# rebuild the dry RAW and HHE SST-box products and regress JJA atmospheric
# fields onto the discovered SST boxes.
#
# Submit:
#   sbatch scripts/run_rankcorr_box_atmo_pipeline.sbatch
#
# Optional overrides:
#   ROOT=/home/vt55/ace2 PY=/path/to/python YEARS=all sbatch ...

set -euo pipefail

ROOT=${ROOT:-/home/vt55/ace2}
PY=${PY:-/home/vt55/mm/envs/ace2-fresh/bin/python}
YEARS=${YEARS:-all}
RUN_INFERENCE=${RUN_INFERENCE:-1}
SKIP_ERA5_ATMO=${SKIP_ERA5_ATMO:-1}
WRITER_NAMES=${WRITER_NAMES:-TMP2m,Q2m,PRESsfc,UGRD10m,VGRD10m,air_temperature_2,eastward_wind_2,northward_wind_2,air_temperature_6,specific_total_water_6,eastward_wind_6,northward_wind_6,surface_temperature}

cd "$ROOT"
mkdir -p "$ROOT/slurm_logs" "$ROOT/outputs/lag_may/logs"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-manual} START rankcorr-atmo pipeline"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "YEARS=$YEARS"
echo "RUN_INFERENCE=$RUN_INFERENCE"
echo "SKIP_ERA5_ATMO=$SKIP_ERA5_ATMO"
echo "WRITER_NAMES=$WRITER_NAMES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

ATMO_ERA5_ARGS=()
if [ "$SKIP_ERA5_ATMO" = "1" ]; then
  ATMO_ERA5_ARGS=(--skip-era5)
fi

run_inference() {
  local ngpus="$1"
  local fsim="$2"
  echo "    run_lag_inference_may.py --n-gpus $ngpus --forward-steps-in-memory $fsim"
  "$PY" scripts/run_lag_inference_may.py \
    --root "$ROOT" \
    --years "$YEARS" \
    --n-steps 500 \
    --n-gpus "$ngpus" \
    --forward-steps-in-memory "$fsim" \
    --skip-existing \
    --writer-names "$WRITER_NAMES"
}

echo "=== Step 1: ACE2 inference with atmo_fig7 variables + surface_temperature, 4 GPUs ==="
if [ "$RUN_INFERENCE" = "0" ]; then
  echo "[$(date)] RUN_INFERENCE=0 -> skipping inference and using available member outputs"
else
  if ! run_inference 4 40; then
    echo "[$(date)] WARNING: 4-GPU pass left failed members; retrying missing outputs with 2 GPUs and smaller chunks"
    if ! run_inference 2 20; then
      echo "[$(date)] WARNING: 2-GPU retry still left failed members; retrying serial with smallest chunks"
      if ! run_inference 1 10; then
        echo "[$(date)] WARNING: inference still has failed members; continuing with available members"
      fi
    fi
  fi
fi

echo "=== Step 2: dry RAW seasonal frequency and rank-corr SST box ==="
"$PY" scripts/combine_lag_may_daily.py --years "$YEARS" --allow-partial
"$PY" scripts/postprocess_jja_lag.py --years "$YEARS"
"$PY" scripts/seasonal_jja_skill.py
"$PY" scripts/sst_raw_rankcorr_box_pipeline.py

echo "=== Step 3: dry RAW Figure-7-style atmospheric regression ==="
"$PY" scripts/sst_rankcorr_box_atmo_regression.py \
  --box-summary "$ROOT/outputs/lag_may/sst_raw_rankcorr_box/sst_raw_rankcorr_box_summary.json" \
  --box-key highest_corr_box_era5 \
  --tag dry_rankcorr_box \
  --years "$YEARS" \
  "${ATMO_ERA5_ARGS[@]}"

echo "=== Step 4: HHE absolute-threshold frequencies and HHE SST box ==="
"$PY" scripts/heat_index_era5.py --years "$YEARS" --force
"$PY" scripts/hhe_ace2_biascorr.py --years "$YEARS" --force-cache --force-bias
"$PY" scripts/sst_hhe_fig2box_corr.py

echo "=== Step 5: HHE Figure-7-style atmospheric regression ==="
"$PY" scripts/sst_rankcorr_box_atmo_regression.py \
  --box-summary "$ROOT/outputs/lag_may/sst_teleconnection_hhe_lagged/sst_hhe_fig2box_summary.json" \
  --box-key highest_corr_box_era5 \
  --tag hhe_fig2box \
  --years "$YEARS" \
  "${ATMO_ERA5_ARGS[@]}"

echo "[$(date)] DONE rankcorr-atmo pipeline"
echo "Dry outputs: $ROOT/outputs/lag_may/sst_rankcorr_box_atmo/dry_rankcorr_box"
echo "HHE outputs: $ROOT/outputs/lag_may/sst_rankcorr_box_atmo/hhe_fig2box"
