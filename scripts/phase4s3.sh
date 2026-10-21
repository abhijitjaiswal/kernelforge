#!/bin/sh
# Wan streaming decoder v3: grid-aligned upsampler strips (Winograd-safe). Verify bit-identity and memory.
PY=$HOME/kf-video-venv/bin/python
L=baselines_video/wan21/v1_latents.safetensors
cool() { $PY bench/cooldown.py --min ${1:-60}; sysctl vm.swapusage; }
VT="$PY -m adapters.wan.vae_tools --latents $L"
echo "== exp-115-v3-pre3"; cool
$VT --exp exp-115-v3-pre3 --impl stream --rows 96 --latent-frames 3 --ref runs/exp-107-untiled-pre3
echo "== exp-115-v3-pre4"; cool
$VT --exp exp-115-v3-pre4 --impl stream --rows 96 --latent-frames 4 --ref runs/exp-109-untiled-pre4
echo "== exp-115-v3-full"; cool
$VT --exp exp-115-v3-full --impl stream --rows 96 --ref runs/ref-v1 --prefix-ref runs/exp-109-untiled-pre4
