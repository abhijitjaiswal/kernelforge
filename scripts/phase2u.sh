#!/bin/sh
# (1) cold single image: baseline (exp-011 config) vs final pipeline, alternating 3 rounds.
# (2) p2 uncompiled (eager) resident reference for the final pipeline's p2 output.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
for i in 1 2 3; do
  echo "== exp-065-base-$i"; cool
  $PY bench/run.py --exp exp-065-base-$i --adapter flux2_klein --opts '{"weights":"bf16","lifecycle":"staged","cache_limit_gb":0.5}' --prompts p1 --runs 1 --no-warmup --ref baselines/bf16_free_te
  echo "== exp-065-final-$i"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-065-final-$i --mode prefetch_blockc --lag 1 --te-stream --runs 0 --purge-steps --ref runs/exp-061-full-1
done
echo "== exp-066-eager-p2"; cool
$PY -m adapters.flux2_klein.stream_bench --exp exp-066-eager-p2 --prompt p2 --mode resident --runs 2 --ref runs/exp-062-full-p2
