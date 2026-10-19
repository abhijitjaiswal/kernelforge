#!/bin/sh
# Phase 0 baseline batch, run on m5-test from ~/kernelforge. 120 s cool-down before each config.
PY=$HOME/kf-venv/bin/python
run() {  # run <exp> <opts-json> <extra args...>
  exp=$1; opts=$2; shift 2
  echo "== $exp cooldown 120s $(date)"; sleep 120
  top -l 1 -n 3 -o cpu | sed -n '4p;7p' ; sysctl vm.swapusage
  $PY bench/run.py --exp $exp --adapter flux2_klein --opts "$opts" "$@"
}
run exp-001 '{"weights":"q8","lifecycle":"free_te"}' --runs 5 --save-baseline baselines/q8_free_te
run exp-002 '{"weights":"q8","lifecycle":"stock"}'   --runs 5 --ref baselines/q8_free_te
run exp-003 '{"weights":"bf16","lifecycle":"free_te"}' --runs 3 --save-baseline baselines/bf16_free_te --ref baselines/q8_free_te
run exp-004 '{"weights":"q8","lifecycle":"free_te","tile_vae":true}' --runs 5 --ref baselines/q8_free_te
