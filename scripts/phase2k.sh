#!/bin/sh
# Phase 2 (1): per-block compiled streaming (timebox 1 day). Speed reference = mflux whole-transformer compile (resident).
# Numerics reference (R1) = resident with the same per-block compilation. Cold SSD (purge before every step) for streaming.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
for i in 1 2; do
  echo "== exp-053-rcomp-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-053-rcomp-$i --mode resident_compiled --runs 2
  echo "== exp-053-rblockc-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-053-rblockc-$i --mode resident_blockc --runs 2 --ref runs/exp-053-rcomp-1
  echo "== exp-053-pblockc-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-053-pblockc-$i --mode prefetch_blockc --runs 2 --purge-steps --ref runs/exp-053-rblockc-1
done
