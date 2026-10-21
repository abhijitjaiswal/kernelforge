#!/bin/sh
# Wan exact streaming decoder v2 (temporal state + height-chunked CausalConv3d + row-chunked upsample Conv2d via frozen conv_rows).
PY=$HOME/kf-video-venv/bin/python
L=baselines_video/wan21/v1_latents.safetensors
cool() { $PY bench/cooldown.py --min ${1:-60}; sysctl vm.swapusage; }
VT="$PY -m adapters.wan.vae_tools --latents $L"
echo "== exp-112-sr-pre3"; cool
$VT --exp exp-112-sr-pre3 --impl stream --rows 96 --latent-frames 3 --ref runs/exp-107-untiled-pre3
echo "== exp-112-sr-pre4"; cool
$VT --exp exp-112-sr-pre4 --impl stream --rows 96 --latent-frames 4 --ref runs/exp-109-untiled-pre4
for n in 6 9; do
  echo "== exp-113-sr-pre$n"; cool
  $VT --exp exp-113-sr-pre$n --impl stream --rows 96 --latent-frames $n --ref runs/ref-v1 --prefix-ref runs/exp-109-untiled-pre4
done
