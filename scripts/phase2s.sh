#!/bin/sh
# Phase 2 footprint (b): transformer weight streaming, 2-block residency. Deliverable: prefetch-overlap measurement.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
[ -f models/klein-blocks-bf16/024.safetensors ] || $PY -m adapters.flux2_klein.stream_bench --prep
du -sh models/klein-blocks-bf16
$PY -m adapters.flux2_klein.stream_bench --exp exp-050 --disk
for i in 1 2; do
  for mode in resident sync prefetch; do
    echo "== exp-051-$mode-$i"; cool
    $PY -m adapters.flux2_klein.stream_bench --exp exp-051-$mode-$i --mode $mode --runs 2 --ref runs/exp-051-resident-1
  done
done
