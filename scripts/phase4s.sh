#!/bin/sh
# Wan exact streaming decoder (temporal streaming + height chunking, fp32) from frozen latents.
PY=$HOME/kf-video-venv/bin/python
L=baselines_video/wan21/v1_latents.safetensors
cool() { $PY bench/cooldown.py --min ${1:-60}; sysctl vm.swapusage; }
VT="$PY -m adapters.wan.vae_tools --latents $L"
echo "== exp-109-stream-pre3 (temporal only)"; cool
$VT --exp exp-109-stream-pre3 --impl stream --latent-frames 3 --ref runs/exp-107-untiled-pre3
echo "== exp-109-streamrows-pre3 (temporal + height 96 rows)"; cool
$VT --exp exp-109-streamrows-pre3 --impl stream --rows 96 --latent-frames 3 --ref runs/exp-107-untiled-pre3
echo "== exp-109-untiled-pre4 (stock reference attempt, 16 frames)"; cool
$VT --exp exp-109-untiled-pre4 --impl untiled --latent-frames 4
for n in 6 9; do
  echo "== exp-110-streamrows-pre$n (memory vs length)"; cool
  $VT --exp exp-110-streamrows-pre$n --impl stream --rows 96 --latent-frames $n --ref runs/ref-v1 --prefix-ref runs/exp-107-untiled-pre3
done
if [ -f runs/exp-109-untiled-pre4/frames.npy ]; then
  echo "== exp-111-prefix16 check on full stream"; cool
  $VT --exp exp-111-streamrows-full-p16 --impl stream --rows 96 --ref runs/ref-v1 --prefix-ref runs/exp-109-untiled-pre4
fi
