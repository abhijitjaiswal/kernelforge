#!/bin/sh
# Wan VAE decoder: frozen-pattern tests from frozen latents (decode only). Watchdog 8 GB default.
PY=$HOME/kf-video-venv/bin/python
L=baselines_video/wan21/v1_latents.safetensors
cool() { $PY bench/cooldown.py --min ${1:-90}; sysctl vm.swapusage; }
mkdir -p runs/ref-v1 && cp baselines_video/wan21/v1_frames_stock.npy runs/ref-v1/frames.npy
echo "== exp-106-tiled"; cool 120
$PY -m adapters.wan.vae_tools --exp exp-106-tiled --latents $L --impl tiled_auto --ref runs/ref-v1
for impl in untiled barriers profile; do
  echo "== exp-107-$impl-pre3"; cool
  $PY -m adapters.wan.vae_tools --exp exp-107-$impl-pre3 --latents $L --impl $impl --latent-frames 3 --ref runs/exp-107-untiled-pre3
done
echo "== exp-108-barriers-full"; cool 120
$PY -m adapters.wan.vae_tools --exp exp-108-barriers-full --latents $L --impl barriers --ref runs/ref-v1
