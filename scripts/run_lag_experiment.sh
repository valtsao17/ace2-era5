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
echo ""

# ── Step 1: Extract lag IC file from ERA5 zarr ────────────────────────────────
IC_FILE="/home/jovyan/ace2_lag_data/initial_conditions/ic_lag_20140201_25m.nc"
if [ -f "$IC_FILE" ]; then
  echo "=== Step 1: lag IC already exists, skipping ==="
else
  echo "=== Step 1: Extracting lag IC from ERA5 zarr ==="
  "$PYTHON" scripts/make_lag_ic.py
fi

# ── Step 2: Run 25 lag member inference jobs ───────────────────────────────────
echo ""
echo "=== Step 2: Running 25-member lag inference ==="
"$PYTHON" scripts/run_lag_inference.py \
  --python "$PYTHON" \
  --skip-existing \
  --forward-steps-in-memory 40

# ── Step 3: Combine member outputs ────────────────────────────────────────────
echo ""
echo "=== Step 3: Combining lag member outputs ==="
"$PYTHON" scripts/combine_lag_predictions.py

# ── Step 4: Postprocess ───────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Postprocessing ==="
"$PYTHON" scripts/run_lag_postprocess.py

echo ""
echo "=== Lag experiment complete ==="
