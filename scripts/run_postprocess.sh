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
"$PYTHON" -m hiro_ace_pipeline.cli postprocess --config configs/smoke_targetdate.yaml --force-figures
