#!/bin/sh
# Full exact pipeline: streamed text encoder + per-block compiled lag-1 streamed transformer + chunked VAE, cold SSD.
# Reference (R1): exp-054-lag1-1 (same pipeline, resident text encoder) -> the image must be bit-identical.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
for i in 1 2; do
  echo "== exp-061-full-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-061-full-$i --mode prefetch_blockc --lag 1 --te-stream --runs 2 --purge-steps --ref runs/exp-054-lag1-1
done
