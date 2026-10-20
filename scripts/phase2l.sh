#!/bin/sh
# Phase 2 (1) cont.: lagged barriers for compiled streaming (cold SSD). Reference: block-compiled resident (R1).
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
for i in 1 2; do
  echo "== exp-054-rblockc-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-054-rblockc-$i --mode resident_blockc --runs 2 --ref runs/exp-053-rblockc-1
  for lag in 1 2; do
    echo "== exp-054-lag$lag-$i"; cool
    $PY -m adapters.flux2_klein.stream_bench --exp exp-054-lag$lag-$i --mode prefetch_blockc --lag $lag --runs 2 --purge-steps --ref runs/exp-053-rblockc-1
  done
done
