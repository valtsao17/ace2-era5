#!/bin/bash
#SBATCH --job-name=sst_v2_prepare
#SBATCH --partition=chaney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/vt55/ace2/slurm_logs/%x_%j.log

# CPU-side prerequisite for the ACE2 SST causal simulations. This creates the
# ACE2-ranked SST composites, 1980-2022 climatology, March/April/May lag ICs,
# shared climatological control, and the three persistent plus one evolving
# global perturbation input sets. It does not launch ACE2 inference.
set -euo pipefail

ROOT=${ROOT:-/home/vt55/ace2}
PY=${PY:-/home/vt55/mm/envs/ace2-fresh/bin/python}
FORCING=${FORCING:-$ROOT/data/lag_data/forcing_data_ace2era5}
ICS=${ICS:-$ROOT/data/lag_data/initial_conditions}
EXP=${EXP:-$ROOT/outputs/lag_may/sst_causal_v2}
YEARS=${YEARS:-1980:2022}
BASE_YEAR=${BASE_YEAR:-2000}
ACE2_BILLING_PROJECT=${ACE2_BILLING_PROJECT:-ace2-ic-download}
HHE_FILE=${HHE_FILE:-$ROOT/outputs/lag_may/heat_index_era5/jja_hi_freq_ace2.nc}
PREP_MONTHS=${PREP_MONTHS:-3,4,5}
PREP_EVOLVING=${PREP_EVOLVING:-1}
REBUILD_COMPOSITE=${REBUILD_COMPOSITE:-0}
REBUILD_CLIMATOLOGY=${REBUILD_CLIMATOLOGY:-0}

cd "$ROOT"
mkdir -p "$EXP" "$ROOT/slurm_logs" "$ROOT/tmp/matplotlib"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$ROOT/tmp/matplotlib}
export ACE2_BILLING_PROJECT

echo "[$(date)] host=$(hostname) START SST causal v2 preparation"
echo "years=$YEARS base_year=$BASE_YEAR months=$PREP_MONTHS evolving=$PREP_EVOLVING"

if [[ ! -s "$HHE_FILE" ]]; then
  echo "ERROR: missing ACE2 HHE file: $HHE_FILE" >&2
  exit 2
fi
for YEAR in $(seq 1980 2022); do
  FILE="$FORCING/forcing_${YEAR}.nc"
  if [[ ! -s "$FILE" ]]; then
    echo "ERROR: missing ACE2 forcing file: $FILE" >&2
    exit 2
  fi
done

COMPOSITE="$EXP/ace2_hhe_sst_composites_mar_aug.nc"
CLIMATOLOGY="$EXP/sst_climatology_1980_2022_on_2000.nc"

if [[ -s "$COMPOSITE" && "$REBUILD_COMPOSITE" != 1 ]]; then
  echo "using existing composite: $COMPOSITE"
else
  "$PY" scripts/ace2_sst_causal_v2.py composite \
    --ace2-hhe-nc "$HHE_FILE" \
    --ace2-hhe-var ace2_hi_freq \
    --forcing-dir "$FORCING" \
    --years "$YEARS" \
    --strict-years \
    --tail-fraction 0.20 \
    --index-box 23 38 260 283 \
    --out "$COMPOSITE"
fi

if [[ -s "$CLIMATOLOGY" && "$REBUILD_CLIMATOLOGY" != 1 ]]; then
  echo "using existing climatology: $CLIMATOLOGY"
else
  "$PY" scripts/ace2_sst_causal_v2.py climatology \
    --forcing-dir "$FORCING" \
    --years "$YEARS" \
    --template-year "$BASE_YEAR" \
    --out "$CLIMATOLOGY"
fi

IFS=',' read -r -a MONTHS <<< "$PREP_MONTHS"
for M in "${MONTHS[@]}"; do
  if [[ "$M" != 3 && "$M" != 4 && "$M" != 5 ]]; then
    echo "ERROR: PREP_MONTHS accepts only comma-separated 3, 4, and 5; got $M" >&2
    exit 2
  fi
  "$PY" scripts/make_lag_data_2017_2022.py \
    --years "$BASE_YEAR" --init-month "$M" --init-day 1 \
    --only ic --ic-dir "$ICS" --forcing-dir "$FORCING" \
    --billing-project "$ACE2_BILLING_PROJECT"
  case "$M" in
    3) MON=mar ;;
    4) MON=apr ;;
    5) MON=may ;;
  esac
  TAG="persistent_${MON}_global"
  ARM_ROOT="$EXP/inputs/$TAG"
  ARM_IC="$ARM_ROOT/initial_conditions/ic_lag_${BASE_YEAR}$(printf '%02d' "$M")01_25m.nc"
  if [[ -s "$ARM_ROOT/manifest.json" && -s "$ARM_ROOT/forcing/forcing_${BASE_YEAR}.nc" && -s "$ARM_IC" ]]; then
    echo "using existing completed arm: $TAG"
  else
    "$PY" scripts/ace2_sst_causal_v2.py prepare \
      --composite "$COMPOSITE" \
      --climatology "$CLIMATOLOGY" \
      --forcing-dir "$FORCING" --ic-dir "$ICS" --out-root "$EXP" \
      --tag "$TAG" --base-year "$BASE_YEAR" \
      --init-month "$M" --mode persistent --precursor-month "$M" \
      --basin global --overwrite
  fi
done

if [[ "$PREP_EVOLVING" == 1 ]]; then
  TAG=evolving_may_aug_global
  ARM_ROOT="$EXP/inputs/$TAG"
  ARM_IC="$ARM_ROOT/initial_conditions/ic_lag_${BASE_YEAR}0501_25m.nc"
  if [[ -s "$ARM_ROOT/manifest.json" && -s "$ARM_ROOT/forcing/forcing_${BASE_YEAR}.nc" && -s "$ARM_IC" ]]; then
    echo "using existing completed arm: $TAG"
  else
    "$PY" scripts/ace2_sst_causal_v2.py prepare \
      --composite "$COMPOSITE" \
      --climatology "$CLIMATOLOGY" \
      --forcing-dir "$FORCING" --ic-dir "$ICS" --out-root "$EXP" \
      --tag "$TAG" --base-year "$BASE_YEAR" \
      --init-month 5 --mode evolving --basin global --overwrite
  fi
fi

echo "[$(date)] COMPLETE SST causal v2 preparation"
find "$EXP/inputs" -maxdepth 2 -type d -print | sort
