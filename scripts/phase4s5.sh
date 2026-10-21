#!/bin/sh
# v5: direct windows + direct upsample strips. Bit-identity on prefixes, then quotable capped full run, then profile.
PY=$HOME/kf-video-venv/bin/python
L=baselines_video/wan21/v1_latents.safetensors
cool() { $PY bench/cooldown.py --min ${1:-60}; sysctl vm.swapusage; }
VT="$PY -m adapters.wan.vae_tools --latents $L --rows 96 --direct --cache-limit-gb 0.5"
echo "== exp-121-v5-pre3"; cool
$VT --exp exp-121-v5-pre3 --impl stream --latent-frames 3 --ref runs/exp-107-untiled-pre3
echo "== exp-121-v5-pre4"; cool
$VT --exp exp-121-v5-pre4 --impl stream --latent-frames 4 --ref runs/exp-109-untiled-pre4
echo "== exp-122-v5-full"; cool 120
$VT --exp exp-122-v5-full --impl stream --ref runs/ref-v1 --prefix-ref runs/exp-109-untiled-pre4
echo "== exp-123-v5-profile"; cool
$VT --exp exp-123-v5-profile --impl stream_profile --latent-frames 2
