#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# Required: observed/full ERA5 SST file. It may be yearly JJA SST with a year
# dimension, or time-resolved SST that this pipeline will average over JJA.
: "${ERA5_SST_NC:?Set ERA5_SST_NC=/path/to/full_era5_sst.nc}"

# Model/hindcast SST source. Preferred for ACE2/SPEAR inference output:
#   MODEL_SST_RUNS_ROOT=/home/vt55/ace2/outputs/lag_may/runs
# This reads {year}/member_XX/autoregressive_predictions.nc and averages JJA
# surface_temperature over time and members. You can alternatively pass one
# model SST NetCDF or one file per year.
MODEL_SST_RUNS_ROOT="${MODEL_SST_RUNS_ROOT:-}"
MODEL_SST_NC="${MODEL_SST_NC:-}"
MODEL_SST_YEARLY_PATTERN="${MODEL_SST_YEARLY_PATTERN:-}"

MODEL_SST_ARGS=()
MODEL_SST_VAR_DEFAULT=""
if [[ -n "$MODEL_SST_RUNS_ROOT" ]]; then
  MODEL_SST_ARGS+=(--sst-model-runs-root "$MODEL_SST_RUNS_ROOT")
  MODEL_SST_VAR_DEFAULT="surface_temperature"
elif [[ -n "$MODEL_SST_YEARLY_PATTERN" ]]; then
  MODEL_SST_ARGS+=(--sst-model-yearly-pattern "$MODEL_SST_YEARLY_PATTERN")
  MODEL_SST_VAR_DEFAULT="surface_temperature"
elif [[ -n "$MODEL_SST_NC" ]]; then
  MODEL_SST_ARGS+=(--sst-model-nc "$MODEL_SST_NC")
else
  echo "Set MODEL_SST_RUNS_ROOT=/path/to/runs, MODEL_SST_NC=/path/to/model_sst.nc, or MODEL_SST_YEARLY_PATTERN='..._{year}.nc'" >&2
  exit 2
fi

MASK_ARGS=()
if [[ "${MASK_SST_OCEAN:-1}" != "0" ]]; then
  MASK_ARGS+=(--mask-sst-ocean)
fi

YEAR_ARGS=()
if [[ -n "${ERA5_YEARS:-}" ]]; then
  YEAR_ARGS+=(--era5-years "$ERA5_YEARS")
fi
if [[ -n "${MODEL_YEARS:-}" ]]; then
  YEAR_ARGS+=(--model-years "$MODEL_YEARS")
fi

mkdir -p "$ROOT/tmp/matplotlib"

MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/tmp/matplotlib}" "$PYTHON" "$ROOT/scripts/extreme_sst_pipeline.py" \
  --tag "${TAG:-hhe_full_sst_seus_tna}" \
  --only "${ONLY:-all}" \
  --years "${YEARS:-1995:2022}" \
  "${YEAR_ARGS[@]}" \
  --target-name "${TARGET_NAME:-HHE}" \
  --model-label "${MODEL_LABEL:-ACE2}" \
  --target-scale "${TARGET_SCALE:-100}" \
  --target-era5-nc "${TARGET_ERA5_NC:-$ROOT/outputs/lag_may/heat_index_era5/jja_hi_freq_era5.nc}" \
  --target-era5-var "${TARGET_ERA5_VAR:-era5_hi_freq}" \
  --target-model-nc "${TARGET_MODEL_NC:-$ROOT/outputs/lag_may/heat_index_era5/jja_hi_freq_ace2.nc}" \
  --target-model-var "${TARGET_MODEL_VAR:-ace2_hi_freq}" \
  --sst-era5-nc "$ERA5_SST_NC" \
  --sst-era5-var "${ERA5_SST_VAR:-}" \
  "${MODEL_SST_ARGS[@]}" \
  --sst-model-var "${MODEL_SST_VAR:-$MODEL_SST_VAR_DEFAULT}" \
  "${MASK_ARGS[@]}" \
  --index-box 23 38 260 283 \
  --tna-box 0 23 280 325 \
  --corr-extent 140 -35 -30 70 \
  --reg-extent -130 -60 23 50 \
  --corr-fdr-q-era5 0.05 \
  --corr-fdr-q-model 0.15 \
  --reg-fdr-q-era5 0.05 \
  --reg-fdr-q-model 0.05 \
  --reg-vmin "${REG_VMIN:--2}" \
  --reg-vmax "${REG_VMAX:-10}"
