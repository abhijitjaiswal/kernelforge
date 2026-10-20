#!/bin/sh
# Final pipeline (exp-061 config) on p2 and p3 for the report grid and per-prompt PSNR vs the bf16 baseline.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
for p in p2 p3; do
  echo "== exp-062-full-$p"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-062-full-$p --prompt $p --mode prefetch_blockc --lag 1 --te-stream --runs 2 --purge-steps
done
