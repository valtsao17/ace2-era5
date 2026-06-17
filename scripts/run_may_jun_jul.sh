#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -n "${PY:-}" ]; then
  PYTHON="$PY"
elif [ -x "$HOME/.conda/envs/hiro-ace-clean/bin/python" ]; then
  PYTHON="$HOME/.conda/envs/hiro-ace-clean/bin/python"
elif [ -x "/home/jovyan/ace2-era5/.conda/envs/ace2/bin/python" ]; then
  PYTHON="/home/jovyan/ace2-era5/.conda/envs/ace2/bin/python"
else
  PYTHON="$(command -v python)"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHON=$PYTHON"

run_target() {
  local CONFIG="$1"
  local PLAN_PATH="$2"

  echo ""
  echo "=== Generating plan: $CONFIG ==="
  "$PYTHON" -m hiro_ace_pipeline.cli plan --config "$CONFIG"

  echo ""
  echo "=== Running inference: $PLAN_PATH ==="
  "$PYTHON" scripts/run_ace2s_inference_plan.py \
    --plan "$PLAN_PATH" \
    --work-root outputs/targetdate_predictions \
    --members 25 \
    --chunk-size 5 \
    --skip-existing
}

run_target configs/targetdate_may.yaml outputs/targetdate_may/plans/target_date_inference_plan.csv
run_target configs/targetdate_jun.yaml outputs/targetdate_jun/plans/target_date_inference_plan.csv
run_target configs/targetdate_jul.yaml outputs/targetdate_jul/plans/target_date_inference_plan.csv

echo ""
echo "All three target dates complete."
