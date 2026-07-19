#!/bin/bash
set -e; cd "$(dirname "$0")/.."; export PYTHONPATH=src
for M in mistral-7b-v0.3 llama-3.1-8b; do
  rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null || true
  echo "=== free-gen (fixed parser): $M ==="
  python3 scripts/evaluate_free_generation.py --model "$M" --device cuda:0
done
echo "=== mistral+llama rerun done ==="
