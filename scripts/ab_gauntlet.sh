#!/usr/bin/env bash
# Full A/B gauntlet: probe then full replay per arm, sequentially.
# Usage: bash scripts/ab_gauntlet.sh [arm1 arm2 ...]   (default: all challenger arms)
set -u
cd /home/oknight/src/hermes-trader
PY=.venv/bin/python
ARMS=("${@:-frognano-4b-q8 spark-x25-4b-q8 qwen25-coder-7b-q8 neohorse-9b-q8 k2-horizon-7b-q8 deepseek-v4pro-q4}")
if [ $# -eq 0 ]; then
  ARMS=(frognano-4b-q8 spark-x25-4b-q8 qwen25-coder-7b-q8 neohorse-9b-q8 k2-horizon-7b-q8)
fi
for arm in "${ARMS[@]}"; do
  echo "=== [$arm] probe (5) $(date -u +%H:%M:%S) ==="
  $PY scripts/llm_ab_replay.py replay --arm "$arm" --limit 5 || { echo "[$arm] PROBE FAILED — skipping arm"; continue; }
  $PY scripts/llm_ab_replay.py audit --arm "$arm" | head -12
  echo "=== [$arm] full replay $(date -u +%H:%M:%S) ==="
  $PY scripts/llm_ab_replay.py replay --arm "$arm" --concurrency 2 || echo "[$arm] replay errors (see results jsonl)"
  $PY scripts/llm_ab_replay.py audit --arm "$arm" | head -6
done
echo "=== gauntlet done $(date -u) ==="
$PY scripts/llm_ab_replay.py score
$PY scripts/llm_ab_replay.py counterfactual
