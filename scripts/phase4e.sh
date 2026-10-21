#!/bin/sh
# Composed-run divergence isolation: 50 steps, streamed TE (exact), stop after denoise, final latents vs the frozen stock latents.
PY=$HOME/kf-video-venv/bin/python
B="$PY -m adapters.wan.bench_wan --prompt v1 --width 832 --height 480 --frames 33 --steps 50 --seed 42 --te-stream --stop-after-denoise --save-noise --cache-limit-gb 0.5"
for m in stock blockc; do
  echo "== exp-130-$m"; $PY bench/cooldown.py --min 120
  $B --exp exp-130-$m --dit $m
  $PY bench/latent_compare.py runs/exp-130-$m/latents.safetensors baselines_video/wan21/v1_latents.safetensors
done
