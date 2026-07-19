#!/bin/bash
# Sequential free-gen for the gated/large models: delete prior HF cache (disk-limited),
# then run. Assumes Qwen runs already done.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for M in mistral-7b-v0.3 llama-3.1-8b; do
  echo "=== cleaning HF cache before $M ==="
  rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null || true
  df -h / | awk 'NR==2{print "  disk free "$4}'
  echo "=== free-gen: $M ==="
  python3 scripts/evaluate_free_generation.py --model "$M" --device cuda:0
done
echo "=== ALL remaining models done ==="
