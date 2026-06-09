#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/targetdate_postprocess_$(date -u +%Y%m%dT%H%M%SZ).log"

nohup "$PROJECT_ROOT/scripts/run_postprocess.sh" > "$LOG" 2>&1 &

echo "Started target-date postprocess"
echo "PID: $!"
echo "LOG: $LOG"
echo "Follow with: tail -f $LOG"
