#!/bin/bash
#SBATCH --job-name=sst_rel_hhe
#SBATCH --partition=gpu_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/vt55/ace2/slurm_logs/%x_%j.log

# Post-process all completed SST-causality simulations with the repo's relative
# HHE definition.  This job launches no ACE2 inference.
#
# Event definition:
#   daily HI > one frozen historical ACE2 grid-cell/calendar-day p90
#   threshold pool = +/-7 calendar days, all 1980-2022 years and members
#   default inputs = Jia-style bias-corrected Tmax and RHmin
#
# Submit from /home/vt55/ace2:
#   sbatch scripts/run_sst_causal_relative_hhe_test.sbatch

set -euo pipefail

ROOT=${ROOT:-/home/vt55/ace2}
PY=${PY:-/home/vt55/mm/envs/ace2-fresh/bin/python}
EXP=${EXP:-$ROOT/outputs/lag_may/sst_causal_v2}
BASE_YEAR=${BASE_YEAR:-2000}
THRESHOLD_YEARS=${THRESHOLD_YEARS:-1980:2022}
MODEL_HI_INPUTS=${MODEL_HI_INPUTS:-bias-corrected}
THRESHOLD_JOBS=${THRESHOLD_JOBS:-4}
MIN_MEMBERS=${MIN_MEMBERS:-20}
FORCE_THRESHOLD=${FORCE_THRESHOLD:-0}
FORCE_EVALUATION=${FORCE_EVALUATION:-0}
FORCE_FREQUENCY_CACHE=${FORCE_FREQUENCY_CACHE:-0}
BUILD_MISSING_HISTORICAL_CACHE=${BUILD_MISSING_HISTORICAL_CACHE:-1}

HISTORICAL_RUNS=${HISTORICAL_RUNS:-$ROOT/outputs/lag_may/runs}
HHE_DIR=${HHE_DIR:-$ROOT/outputs/lag_may/heat_index_era5}
DAILY_CACHE=${DAILY_CACHE:-$HHE_DIR/ace2_daily_cache}
BIAS_NC=${BIAS_NC:-$HHE_DIR/bias_fields.nc}
THRESHOLD_DIR=${THRESHOLD_DIR:-$EXP/relative_hhe_threshold}
THRESHOLD_NC=${THRESHOLD_NC:-$THRESHOLD_DIR/ace2_hi_p90_pm7d_1980_2022_${MODEL_HI_INPUTS}.nc}
RELATIVE_EVAL_DIR=${RELATIVE_EVAL_DIR:-$EXP/evaluation_relative_hhe}
FREQUENCY_CACHE=${FREQUENCY_CACHE:-$EXP/relative_hhe_frequency_cache}
ABSOLUTE_EVAL_DIR=${ABSOLUTE_EVAL_DIR:-$EXP/evaluation}
COMPARISON_DIR=${COMPARISON_DIR:-$RELATIVE_EVAL_DIR/comparison}
LAND_MASK_FORCING=${LAND_MASK_FORCING:-$EXP/inputs/control/forcing/forcing_${BASE_YEAR}.nc}

cd "$ROOT"
mkdir -p "$ROOT/slurm_logs" "$ROOT/tmp/matplotlib" "$ROOT/tmp/joblib" \
  "$DAILY_CACHE" "$THRESHOLD_DIR" "$RELATIVE_EVAL_DIR" "$FREQUENCY_CACHE" "$COMPARISON_DIR"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$ROOT/tmp/matplotlib}
export JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-$ROOT/tmp/joblib}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date)] host=$(hostname) START relative-HHE SST sensitivity test"
echo "EXP=$EXP"
echo "threshold years=$THRESHOLD_YEARS"
echo "model HI inputs=$MODEL_HI_INPUTS"
echo "threshold=$THRESHOLD_NC"
echo "relative evaluations=$RELATIVE_EVAL_DIR"

# Relative HHE cannot be reconstructed from the compact absolute-HHE
# evaluation alone; the saved TMP2m/Q2m/PRESsfc trajectories must still exist.
REQUIRED_RUN_TAGS=(
  control_mar control_apr control_may
  persistent_mar_global persistent_apr_global persistent_may_global
  evolving_may_aug_global
  evolving_may_aug_tropical_pacific evolving_may_aug_north_pacific
  evolving_may_aug_tropical_atlantic evolving_may_aug_north_atlantic
)
for RUN_TAG in "${REQUIRED_RUN_TAGS[@]}"; do
  RUN_YEAR_DIR="$EXP/runs/$RUN_TAG/$BASE_YEAR"
  if [[ ! -d "$RUN_YEAR_DIR" ]]; then
    echo "ERROR: missing saved trajectory directory: $RUN_YEAR_DIR" >&2
    exit 2
  fi
  RUN_COUNT=$(find "$RUN_YEAR_DIR" -type f -name autoregressive_predictions.nc | wc -l)
  if (( RUN_COUNT < MIN_MEMBERS )); then
    echo "ERROR: $RUN_TAG has only $RUN_COUNT saved trajectories; $MIN_MEMBERS required." >&2
    exit 2
  fi
done

# The threshold builder consumes the compact historical daily caches rather
# than the raw trajectories.  Build them only when they are genuinely absent.
FIRST_CACHE=$(find "$DAILY_CACHE" -maxdepth 1 -type f -name 'daily_y*_mem*.nc' -print -quit 2>/dev/null || true)
CACHE_INCOMPLETE=0
if [[ "$THRESHOLD_YEARS" == *:* ]]; then
  START_YEAR=${THRESHOLD_YEARS%%:*}
  END_YEAR=${THRESHOLD_YEARS##*:}
  for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
    COUNT=$(find "$DAILY_CACHE" -maxdepth 1 -type f -name "daily_y${YEAR}_mem*.nc" 2>/dev/null | wc -l)
    if (( COUNT < MIN_MEMBERS )); then
      echo "historical daily cache incomplete for $YEAR: $COUNT/$MIN_MEMBERS required"
      CACHE_INCOMPLETE=1
    fi
  done
fi
if [[ -z "$FIRST_CACHE" || "$CACHE_INCOMPLETE" == 1 || ( "$MODEL_HI_INPUTS" == "bias-corrected" && ! -s "$BIAS_NC" ) ]]; then
  if [[ "$BUILD_MISSING_HISTORICAL_CACHE" != 1 ]]; then
    echo "ERROR: missing historical ACE2 daily cache and/or bias file." >&2
    echo "Set BUILD_MISSING_HISTORICAL_CACHE=1 or run hhe_ace2_biascorr.py first." >&2
    exit 2
  fi
  echo "[$(date)] building missing historical daily cache/bias fields"
  HHE_ARGS=(--years "$THRESHOLD_YEARS")
  if [[ "$CACHE_INCOMPLETE" == 1 || ! -s "$BIAS_NC" ]]; then
    HHE_ARGS+=(--force-bias)
  fi
  RUNS_ROOT="$HISTORICAL_RUNS" "$PY" scripts/hhe_ace2_biascorr.py "${HHE_ARGS[@]}"
fi

if [[ "$FORCE_THRESHOLD" == 1 ]]; then
  THRESHOLD_ARGS=(--out "$THRESHOLD_NC")
elif [[ -s "$THRESHOLD_NC" ]]; then
  THRESHOLD_ARGS=()
  echo "using existing frozen threshold: $THRESHOLD_NC"
else
  THRESHOLD_ARGS=(--out "$THRESHOLD_NC")
fi

if (( ${#THRESHOLD_ARGS[@]} )); then
  echo "[$(date)] building frozen historical ACE2 relative-HHE threshold"
  "$PY" scripts/build_ace2_relative_hhe_threshold.py \
    --years "$THRESHOLD_YEARS" \
    --daily-cache-dir "$DAILY_CACHE" \
    --bias-nc "$BIAS_NC" \
    --model-hi-inputs "$MODEL_HI_INPUTS" \
    --percentile 90 \
    --window 7 \
    --min-members "$MIN_MEMBERS" \
    --jobs "$THRESHOLD_JOBS" \
    "${THRESHOLD_ARGS[@]}"
fi

if [[ ! -s "$THRESHOLD_NC" ]]; then
  echo "ERROR: threshold was not created: $THRESHOLD_NC" >&2
  exit 2
fi
if [[ ! -s "$LAND_MASK_FORCING" ]]; then
  echo "ERROR: missing land-mask forcing file: $LAND_MASK_FORCING" >&2
  exit 2
fi

evaluate_experiment() {
  local HIGH_TAG=$1
  local LOW_TAG=$2
  local INIT_MONTH=$3
  local HIGH_LABEL=$4
  local LOW_LABEL=$5
  local OUT_NC="$RELATIVE_EVAL_DIR/${HIGH_TAG}_${BASE_YEAR}.nc"
  local OUT_JSON="${OUT_NC%.nc}.json"

  if [[ "$FORCE_EVALUATION" != 1 && -s "$OUT_NC" && -s "$OUT_JSON" ]]; then
    echo "using existing relative-HHE evaluation: $OUT_NC"
    return
  fi
  if [[ ! -d "$EXP/runs/$HIGH_TAG/$BASE_YEAR" ]]; then
    echo "ERROR: missing perturbation run: $EXP/runs/$HIGH_TAG/$BASE_YEAR" >&2
    exit 2
  fi
  if [[ ! -d "$EXP/runs/$LOW_TAG/$BASE_YEAR" ]]; then
    echo "ERROR: missing control run: $EXP/runs/$LOW_TAG/$BASE_YEAR" >&2
    exit 2
  fi

  local CACHE_ARGS=()
  if [[ "$FORCE_FREQUENCY_CACHE" == 1 ]]; then
    CACHE_ARGS+=(--force-cache)
  fi
  echo "[$(date)] relative-HHE evaluation: $HIGH_TAG minus $LOW_TAG"
  "$PY" scripts/evaluate_sst_composite_causal_relative.py \
    --high-runs "$EXP/runs/$HIGH_TAG" \
    --low-runs "$EXP/runs/$LOW_TAG" \
    --high-cache-tag "$HIGH_TAG" \
    --low-cache-tag "$LOW_TAG" \
    --cache-dir "$FREQUENCY_CACHE" \
    --threshold-nc "$THRESHOLD_NC" \
    --bias-nc "$BIAS_NC" \
    --model-hi-inputs "$MODEL_HI_INPUTS" \
    --high-label "$HIGH_LABEL" \
    --low-label "$LOW_LABEL" \
    --years "$BASE_YEAR" \
    --members all \
    --min-members "$MIN_MEMBERS" \
    --min-jja-days 90 \
    --init-month "$INIT_MONTH" \
    --init-day 1 \
    --index-box 23 38 260 283 \
    --land-mask-forcing "$LAND_MASK_FORCING" \
    --out "$OUT_NC" \
    "${CACHE_ARGS[@]}"
}

CONTROL_LABEL="Climatological SST control"
evaluate_experiment \
  persistent_mar_global control_mar 3 \
  "Climatology + persistent March SST anomaly" "$CONTROL_LABEL"
evaluate_experiment \
  persistent_apr_global control_apr 4 \
  "Climatology + persistent April SST anomaly" "$CONTROL_LABEL"
evaluate_experiment \
  persistent_may_global control_may 5 \
  "Climatology + persistent May SST anomaly" "$CONTROL_LABEL"
evaluate_experiment \
  evolving_may_aug_global control_may 5 \
  "Climatology + evolving May-August SST anomalies" "$CONTROL_LABEL"
evaluate_experiment \
  evolving_may_aug_tropical_pacific control_may 5 \
  "Climatology + evolving Tropical Pacific SST anomalies" "$CONTROL_LABEL"
evaluate_experiment \
  evolving_may_aug_north_pacific control_may 5 \
  "Climatology + evolving North Pacific SST anomalies" "$CONTROL_LABEL"
evaluate_experiment \
  evolving_may_aug_tropical_atlantic control_may 5 \
  "Climatology + evolving Tropical Atlantic SST anomalies" "$CONTROL_LABEL"
evaluate_experiment \
  evolving_may_aug_north_atlantic control_may 5 \
  "Climatology + evolving North Atlantic SST anomalies" "$CONTROL_LABEL"

echo "[$(date)] comparing relative and absolute HHE treatment effects"
"$PY" scripts/compare_sst_causal_hhe_definitions.py \
  --absolute-dir "$ABSOLUTE_EVAL_DIR" \
  --relative-dir "$RELATIVE_EVAL_DIR" \
  --base-year "$BASE_YEAR" \
  --out-dir "$COMPARISON_DIR"

echo "[$(date)] COMPLETE relative-HHE SST sensitivity test"
echo "threshold:  $THRESHOLD_NC"
echo "evaluations: $RELATIVE_EVAL_DIR"
echo "comparison:  $COMPARISON_DIR/relative_vs_absolute_hhe_sensitivity.png"
