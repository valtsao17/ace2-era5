#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v mamba >/dev/null 2>&1; then
  mamba env create -f environment.yml || mamba env update -f environment.yml
else
  conda env create -f environment.yml || conda env update -f environment.yml
fi

echo "Activate with: conda activate hiro-ace-clean"
