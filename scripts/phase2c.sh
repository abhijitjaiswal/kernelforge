#!/bin/sh
# Phase 2 (b) cold-SSD test: purge the page cache before every denoise step. Prediction recorded in NOTES.md first.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
$PY -m adapters.flux2_klein.stream_bench --exp exp-052-disk --disk
for i in 1 2; do
  echo "== exp-052-barrier-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-052-barrier-$i --mode barrier --runs 2 --ref runs/exp-051-resident-1
  for mode in sync prefetch; do
    echo "== exp-052-$mode-cold-$i"; cool
    $PY -m adapters.flux2_klein.stream_bench --exp exp-052-$mode-cold-$i --mode $mode --runs 2 --purge-steps --ref runs/exp-051-resident-1
  done
done
