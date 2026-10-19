#!/bin/sh
# Phase 0b: memory lifecycle, cold single image per process (p1), memtrace on. 180 s cool-down.
PY=$HOME/kf-venv/bin/python
run() {
  exp=$1; opts=$2; shift 2
  echo "== $exp cooldown 180s $(date)"; sleep 180
  top -l 1 -n 0 | grep -E 'CPU usage|PhysMem'; sysctl vm.swapusage
  $PY bench/run.py --exp $exp --adapter flux2_klein --opts "$opts" --prompts p1 --runs 1 --no-warmup "$@"
}
run exp-005 '{"weights":"q8","lifecycle":"free_te"}' --ref baselines/q8_free_te
run exp-006 '{"weights":"q8","lifecycle":"staged"}' --ref baselines/q8_free_te
run exp-007 '{"weights":"bf16","lifecycle":"staged"}' --ref baselines/bf16_free_te
run exp-008 '{"weights":"q8","lifecycle":"staged","tile_vae":true}' --ref baselines/q8_free_te
