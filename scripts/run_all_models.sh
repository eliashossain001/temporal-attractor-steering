#!/usr/bin/env bash
# Run the full TAS pipeline for every model in MODELS, one at a time.
# Each stage is idempotent; restarting after a failure resumes where it
# left off.
#
# Usage:
#     bash scripts/run_all_models.sh                       # subset
#     DATA=data/large RUNS=runs/large bash scripts/run_all_models.sh

set -uo pipefail

DATA="${DATA:-data/subset}"
RUNS="${RUNS:-runs/main}"
MODELS=("${MODELS[@]:-qwen-2.5-1.5b qwen-2.5-7b mistral-7b-v0.3 llama-3.1-8b}")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for MODEL in "${MODELS[@]}"; do
  log "==========================================================="
  log "MODEL: $MODEL"
  log "==========================================================="
  python -m temporal_conflict.cli tas \
    --model "$MODEL" \
    --data "$DATA" \
    --runs-root "$RUNS" \
    || log "[FAIL] $MODEL"
done

log "ALL MODELS DONE"
