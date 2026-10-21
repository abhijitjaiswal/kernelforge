#!/bin/sh
# Wan DiT block streaming (frozen engine via adapter), 8 steps, stop after denoise, alternating A/B, cache cap 0.5 GB.
PY=$HOME/kf-video-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-120}; sysctl vm.swapusage; }
[ -f models/wan21-dit-blocks/block29.safetensors ] || $PY -m adapters.wan.dit_stream prep
B="$PY -m adapters.wan.bench_wan --prompt v1 --width 832 --height 480 --frames 33 --steps 8 --seed 42 --context-ref baselines_video/wan21/v1_text_context.safetensors --stop-after-denoise --cache-limit-gb 0.5"
for i in 1 2; do
  for m in stock blockc stream; do
    echo "== exp-126-$m-$i"; cool
    $B --exp exp-126-$m-$i --dit $m
  done
done
