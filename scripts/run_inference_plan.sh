#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -n "${PY:-}" ]; then
  PYTHON="$PY"
else
  PYTHON=/home/jovyan/ace2-era5/.conda/envs/ace2/bin/python
fi

PLAN=outputs/targetdate_smoke/plans/target_date_inference_plan.csv

if [ ! -f "$PLAN" ]; then
  echo "Missing plan: $PLAN"
  echo "Run: bash scripts/run_plan.sh"
  exit 1
fi

"$PYTHON" scripts/run_ace2s_inference_plan.py \
  --plan "$PLAN" \
  --work-root outputs/targetdate_predictions \
  --members 25 \
  --chunk-size 5 \
  --skip-existing
