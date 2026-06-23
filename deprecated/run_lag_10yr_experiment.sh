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
LOGDIR="$PROJECT_ROOT/logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHON=$PYTHON"
echo ""

# ── Step 1: Convert forcing files 1980-1990 to ACE2-ERA5 format ───────────────
echo "=== Step 1: Convert forcing files ==="
"$PYTHON" scripts/convert_forcing_ace2era5.py 2>&1 | tee -a "$LOGDIR/lag_10yr_${TS}.log"

# ── Step 2: Extract lag IC files for Nov 1, 1980-1989 ─────────────────────────
echo ""
echo "=== Step 2: Extract lag IC files ==="
"$PYTHON" scripts/make_lag_ics_nov.py 2>&1 | tee -a "$LOGDIR/lag_10yr_${TS}.log"

# ── Step 3: Run 250 inference jobs ────────────────────────────────────────────
echo ""
echo "=== Step 3: Run 250-member lag inference (10 years × 25 members) ==="
"$PYTHON" scripts/run_lag_inference_nov.py \
  --python "$PYTHON" \
  --skip-existing \
  --forward-steps-in-memory 40 \
  2>&1 | tee -a "$LOGDIR/lag_10yr_${TS}.log"

# ── Step 4: Combine member outputs ────────────────────────────────────────────
echo ""
echo "=== Step 4: Combine and extract target date snapshots ==="
"$PYTHON" scripts/combine_lag_nov.py 2>&1 | tee -a "$LOGDIR/lag_10yr_${TS}.log"

# ── Step 5: Rank correlation analysis and figures ─────────────────────────────
echo ""
echo "=== Step 5: Rank correlation analysis ==="
"$PYTHON" scripts/rank_corr_analysis.py 2>&1 | tee -a "$LOGDIR/lag_10yr_${TS}.log"

echo ""
echo "=== 10-year lag experiment complete ==="
echo "Log: $LOGDIR/lag_10yr_${TS}.log"
