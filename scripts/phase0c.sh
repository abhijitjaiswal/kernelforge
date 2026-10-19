#!/bin/sh
# Phase 0c: MLX cache cap, then an alternating bf16 vs q8 speed test (cold p1, staged, cache cap 0.5 GB).
PY=$HOME/kf-venv/bin/python
run() {
  exp=$1; opts=$2; cool=$3; shift 3
  echo "== $exp cooldown ${cool}s $(date)"; sleep $cool
  top -l 1 -n 0 | grep -E 'CPU usage|PhysMem'; sysctl vm.swapusage
  $PY bench/run.py --exp $exp --adapter flux2_klein --opts "$opts" --prompts p1 --runs 1 --no-warmup "$@"
}
BF='{"weights":"bf16","lifecycle":"staged","cache_limit_gb":0.5}'
Q8='{"weights":"q8","lifecycle":"staged","cache_limit_gb":0.5}'
run exp-009 "$BF" 180 --ref baselines/bf16_free_te
run exp-010 '{"weights":"q8","lifecycle":"staged","tile_vae":true,"cache_limit_gb":0.5}' 180 --ref baselines/q8_free_te
for i in 1 2 3; do
  run exp-011-bf16-$i "$BF" 150 --ref baselines/bf16_free_te
  run exp-011-q8-$i "$Q8" 150 --ref baselines/q8_free_te
done
