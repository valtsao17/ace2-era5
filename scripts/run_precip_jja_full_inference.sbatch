#!/bin/bash
#SBATCH --job-name=precip_jja_full
#SBATCH --partition=gpu_node
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/vt55/ace2/outputs/lag_may/regen_logs/precip_jja_full_%j.log

# Full-range ACE2 PRATEsfc JJA inference for 1980-2016 (37 yrs x 25 members,
# 500 steps), DISK-SAFE and RESUMABLE:
#   * a year whose 25 CONUS-JJA member caches already exist is SKIPPED outright
#     (so the already-finished 2002-2016 — whose bulky raw dirs were deleted —
#      is never re-run, and a requeued/resubmitted job picks up where it left off);
#   * otherwise run inference -> extract compact CONUS JJA precip cache ->
#     delete the bulky raw run dir before moving on (home is ~98% full).
# SLURM makes this disconnect-proof; --requeue + the skip check make it
# crash/preemption-proof: just `sbatch` it again to resume.
# NOT set -e: a single bad year must not abort the whole sweep.
set -uo pipefail
cd /home/vt55/ace2
PY=/home/vt55/mm/envs/ace2-fresh/bin/python
RUNS=outputs/lag_may/precip_jja/runs
CACHE=outputs/lag_may/precip_jja/ace2_precip_cache
mkdir -p "$RUNS" "$CACHE"

echo "[$(date)] host=$(hostname) job=$SLURM_JOB_ID START precip JJA FULL inference 1980-2016"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

for Y in $(seq 1980 2016); do
  HAVE=$(ls "$CACHE"/precip_y${Y}_mem*.nc 2>/dev/null | wc -l)
  if [ "$HAVE" -eq 25 ]; then
    echo "=== YEAR $Y: already cached 25/25 -> SKIP ==="
    continue
  fi

  echo "=== YEAR $Y: inference (4 GPUs) ==="
  $PY scripts/run_lag_inference_may.py \
      --years "$Y" --writer-names PRATEsfc --n-steps 500 --n-gpus 4 \
      --output-dir "$RUNS" --skip-existing

  echo "=== YEAR $Y: extract CONUS JJA daily precip ==="
  $PY scripts/extract_precip_jja.py --year "$Y" --runs-root "$RUNS"

  NC=$(ls "$CACHE"/precip_y${Y}_mem*.nc 2>/dev/null | wc -l)
  if [ "$NC" -eq 25 ]; then
    echo "=== YEAR $Y: 25/25 cached -> deleting raw run dir $RUNS/$Y ==="
    rm -rf "${RUNS:?}/$Y"
  else
    echo "=== YEAR $Y: only $NC/25 cached -> KEEPING raw run dir for inspection ==="
  fi
  echo "[$(date)] disk: $(df -h /home | tail -1 | awk '{print $4" free ("$5")"}')"
done

echo "[$(date)] DONE precip JJA FULL inference+extract"
TOT=$(for Y in $(seq 1980 2016); do ls "$CACHE"/precip_y${Y}_mem*.nc 2>/dev/null | wc -l | tr -d '\n'; echo -n " "; done)
echo "per-year member counts 1980..2016: $TOT"
