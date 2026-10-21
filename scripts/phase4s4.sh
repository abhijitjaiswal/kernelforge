#!/bin/sh
# Wan streaming decoder v4 (option 1: direct per-frame windows). Verify bit-identity, then quotable capped full run.
PY=$HOME/kf-video-venv/bin/python
L=baselines_video/wan21/v1_latents.safetensors
cool() { $PY bench/cooldown.py --min ${1:-60}; sysctl vm.swapusage; }
VT="$PY -m adapters.wan.vae_tools --latents $L --impl stream --rows 96 --direct --cache-limit-gb 0.5"
echo "== exp-118-v4-pre3"; cool
$VT --exp exp-118-v4-pre3 --latent-frames 3 --ref runs/exp-107-untiled-pre3
echo "== exp-118-v4-pre4"; cool
$VT --exp exp-118-v4-pre4 --latent-frames 4 --ref runs/exp-109-untiled-pre4
echo "== exp-119-v4-full"; cool 120
$VT --exp exp-119-v4-full --ref runs/ref-v1 --prefix-ref runs/exp-109-untiled-pre4
