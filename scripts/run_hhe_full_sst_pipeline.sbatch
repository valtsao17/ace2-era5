#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu_node
#SBATCH --exclude=node1,node2,node3,node4,node5,node6,node7,node8,node9,node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20
#SBATCH --exclusive
#SBATCH --job-name="hhe_full_sst"
#SBATCH --output="/home/vt55/ace2/slurm_logs/%x_%j.log"

# Full HHE/SST pipeline:
#   1. optionally rerun May inference with surface_temperature written
#   2. rebuild HHE target frequencies/fig2
#   3. compute Figure 4 and/or Figure 6 style SST products using:
#        ERA5 side  -> ERA5_SST_NC
#        model side -> ACE2/SPEAR inference outputs in outputs/lag_may/runs
#
# Submit example:
#   ERA5_SST_NC=/path/to/full_era5_sst.nc RUN_INFERENCE=0 sbatch scripts/run_hhe_full_sst_pipeline.sbatch

set -euo pipefail

ROOT=${ROOT:-/home/vt55/ace2}
PY=${PY:-/home/vt55/mm/envs/ace2-fresh/bin/python}
YEARS=${YEARS:-1995:2022}
ERA5_YEARS=${ERA5_YEARS:-}
MODEL_YEARS=${MODEL_YEARS:-}
HHE_ERA5_YEARS=${ERA5_YEARS:-$YEARS}
HHE_MODEL_YEARS=${MODEL_YEARS:-$YEARS}
RUN_INFERENCE=${RUN_INFERENCE:-0}
RUN_REBUILD_HHE=${RUN_REBUILD_HHE:-1}
RUNS_ROOT=${RUNS_ROOT:-$ROOT/outputs/lag_may/runs}
ERA5_SST_NC=${ERA5_SST_NC:-auto}
ONLY=${ONLY:-all}

WRITER_NAMES=${WRITER_NAMES:-TMP2m,Q2m,PRESsfc,UGRD10m,VGRD10m,surface_temperature}

cd "$ROOT"
mkdir -p "$ROOT/slurm_logs" "$ROOT/outputs/lag_may/logs" "$ROOT/tmp/matplotlib"

if [ "$ERA5_SST_NC" = "auto" ]; then
  for candidate in \
    "$ROOT/outputs/lag_may/postprocess_jja/sst_teleconnection/era5_jja_sst_mean.nc" \
    "$ROOT/outputs/lag_may/sst_teleconnection_sliding7d/era5_jja_sst_mean.nc"; do
    if [ -s "$candidate" ]; then
      ERA5_SST_NC="$candidate"
      break
    fi
  done
fi
if [ "$ERA5_SST_NC" = "auto" ] || [ ! -s "$ERA5_SST_NC" ]; then
  echo "ERROR: ERA5 SST cache not found. Set ERA5_SST_NC=/path/to/era5_jja_sst_mean.nc" >&2
  echo "Checked:" >&2
  echo "  $ROOT/outputs/lag_may/postprocess_jja/sst_teleconnection/era5_jja_sst_mean.nc" >&2
  echo "  $ROOT/outputs/lag_may/sst_teleconnection_sliding7d/era5_jja_sst_mean.nc" >&2
  exit 2
fi

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-manual} START HHE full-SST pipeline"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "YEARS=$YEARS"
echo "ERA5_YEARS=${ERA5_YEARS:-<default YEARS>}"
echo "MODEL_YEARS=${MODEL_YEARS:-<default YEARS>}"
echo "HHE_ERA5_YEARS=$HHE_ERA5_YEARS"
echo "HHE_MODEL_YEARS=$HHE_MODEL_YEARS"
echo "RUN_INFERENCE=$RUN_INFERENCE"
echo "RUN_REBUILD_HHE=$RUN_REBUILD_HHE"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "ERA5_SST_NC=$ERA5_SST_NC"
echo "ONLY=$ONLY"
echo "WRITER_NAMES=$WRITER_NAMES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

if [ "$RUN_INFERENCE" = "1" ]; then
  echo "=== Step 1: rerun/resume ACE2 inference with surface_temperature ==="
  if ! "$PY" scripts/run_lag_inference_may.py \
      --root "$ROOT" \
      --years "$YEARS" \
      --n-steps 500 \
      --n-gpus 4 \
      --forward-steps-in-memory 40 \
      --skip-existing \
      --writer-names "$WRITER_NAMES"; then
    echo "[$(date)] WARNING: 4-GPU pass failed; retrying with 2 GPUs"
    "$PY" scripts/run_lag_inference_may.py \
      --root "$ROOT" \
      --years "$YEARS" \
      --n-steps 500 \
      --n-gpus 2 \
      --forward-steps-in-memory 20 \
      --skip-existing \
      --writer-names "$WRITER_NAMES"
  fi
else
  echo "=== Step 1: RUN_INFERENCE=0, using existing inference outputs ==="
fi

if [ "$RUN_REBUILD_HHE" = "1" ]; then
  echo "=== Step 2: rebuild HHE frequencies and Figure 2 ==="
  "$PY" scripts/heat_index_era5.py --years "$HHE_ERA5_YEARS" --force
  "$PY" scripts/hhe_ace2_biascorr.py --years "$HHE_MODEL_YEARS" --force-cache --force-bias
  "$PY" scripts/extreme_freq_boxed_panels.py
else
  echo "=== Step 2: RUN_REBUILD_HHE=0, using existing HHE frequency files ==="
fi

echo "=== Step 3: Figure 4/Figure 6 with full SST and fixed boxes ==="
ERA5_SST_NC="$ERA5_SST_NC" \
MODEL_SST_RUNS_ROOT="$RUNS_ROOT" \
YEARS="$YEARS" \
ERA5_YEARS="$ERA5_YEARS" \
MODEL_YEARS="$MODEL_YEARS" \
MODEL_LABEL="${MODEL_LABEL:-ACE2}" \
TAG="${TAG:-hhe_full_sst_seus_tna}" \
ONLY="$ONLY" \
PYTHON="$PY" \
MPLCONFIGDIR="$ROOT/tmp/matplotlib" \
bash scripts/run_hhe_full_sst_pipeline.sh

echo "[$(date)] DONE HHE full-SST pipeline"
echo "Outputs: $ROOT/outputs/lag_may/extreme_sst_pipeline/${TAG:-hhe_full_sst_seus_tna}"
