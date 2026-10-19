#!/bin/sh
# Phase 1a: exp-010 x3, tiled images for the quality grid, alternating stage-bench bf16 vs q8, then quality.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; top -l 1 -n 0 | grep -E 'CPU usage|PhysMem'; sysctl vm.swapusage; }
TILE='{"weights":"q8","lifecycle":"staged","tile_vae":true,"cache_limit_gb":0.5}'
for i in 1 2 3; do
  echo "== exp-010-rep-$i"; cool
  $PY bench/run.py --exp exp-010-rep-$i --adapter flux2_klein --opts "$TILE" --prompts p1 --runs 1 --no-warmup --ref baselines/q8_free_te
done
echo "== exp-013"; cool
$PY bench/run.py --exp exp-013 --adapter flux2_klein --opts '{"weights":"q8","lifecycle":"free_te","tile_vae":true,"cache_limit_gb":0.5}' --prompts p1,p2,p3 --runs 1 --ref baselines/q8_free_te
for i in 1 2 3; do
  for w in bf16 q8; do
    echo "== exp-015-$w-$i"; cool
    $PY bench/stages.py --exp exp-015-$w-$i --adapter flux2_klein --opts "{\"weights\":\"$w\",\"lifecycle\":\"staged\",\"cache_limit_gb\":0.5}" --prompt p1 --runs 5 --ref baselines/${w}_free_te
  done
done
echo "== quality"
$PY bench/quality.py --cand baselines/q8_free_te --ref baselines/bf16_free_te --out runs/exp-003/quality --label "q8 vs bf16 (exp-003)"
$PY bench/quality.py --cand runs/exp-013 --ref baselines/bf16_free_te --out runs/exp-013/quality_vs_bf16 --label "q8 + tiled VAE vs bf16 baseline"
$PY bench/quality.py --cand runs/exp-013 --ref baselines/q8_free_te --out runs/exp-013/quality_vs_q8 --label "tiled VAE vs untiled (both q8)"
