#!/bin/sh
# On m5-test: after exp-131, pause >= 5 min until thermal is Nominal, then stock tiled decode of the true-stock latents (exp-130)
# and frame comparison against the composed pipeline's video (exp-127).
cd "$HOME/kernelforge" || exit 1
PY=$HOME/kf-video-venv/bin/python
until grep -q "^EXIT=" runs/jobs/phase4f.log; do sleep 15; done
$PY bench/cooldown.py --min 300
$PY -m adapters.wan.vae_tools --exp exp-132-stock-decode --latents runs/exp-130-stock/latents.safetensors --impl tiled_auto \
  --ref runs/exp-127-composed --swap-abort-gb 8
