#!/bin/sh
# Phase 1b: exact low-memory VAE decode. Latents for p2/p3, VAE-only decode variants on p1-p3, then
# full-pipeline A/B (stock vs chunked decode) and the 8 GB cap simulation.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
BF='"weights":"bf16","lifecycle":"staged","cache_limit_gb":0.5'
for p in p2 p3; do
  echo "== exp-016-lat-$p"; cool 120
  $PY bench/stages.py --exp exp-016-lat-$p --adapter flux2_klein --opts "{$BF}" --prompt $p --runs 1 --ref baselines/bf16_free_te
done
lat() { case $1 in p1) echo runs/exp-012-smoke/latents.safetensors;; *) echo runs/exp-016-lat-$1/latents.safetensors;; esac; }
VT="$PY -m adapters.flux2_klein.vae_tools decode --weights bf16 --runs 3 --ref baselines/bf16_free_te"
for p in p1 p2 p3; do
  echo "== VAE-only $p"; cool 60
  $VT --exp exp-017-stock-$p     --prompt $p --latents $(lat $p) --impl stock
  cool 60
  $VT --exp exp-018-barriers-$p  --prompt $p --latents $(lat $p) --impl barriers
  cool 60
  $VT --exp exp-019-chunk128-$p  --prompt $p --latents $(lat $p) --impl chunked --impl-opts '{"rows":128}'
done
for r in 64 256; do
  echo "== VAE-only rows $r"; cool 60
  $VT --exp exp-020-chunk$r-p1 --prompt p1 --latents $(lat p1) --impl chunked --impl-opts "{\"rows\":$r}"
done
for i in 1 2 3; do
  echo "== exp-021-stock-$i"; cool
  $PY bench/run.py --exp exp-021-stock-$i --adapter flux2_klein --opts "{$BF}" --prompts p1 --runs 1 --no-warmup --ref baselines/bf16_free_te
  echo "== exp-021-chunk-$i"; cool
  $PY bench/run.py --exp exp-021-chunk-$i --adapter flux2_klein --opts "{$BF,\"vae_impl\":\"chunked\"}" --prompts p1 --runs 1 --no-warmup --ref baselines/bf16_free_te
done
for cap in 8 6; do
  echo "== exp-022-cap$cap"; cool
  $PY bench/run.py --exp exp-022-cap$cap --adapter flux2_klein --opts "{$BF,\"vae_impl\":\"chunked\",\"mem_limit_gb\":$cap}" --prompts p1 --runs 1 --no-warmup --ref baselines/bf16_free_te
done
