#!/bin/sh
# Runs on m5-test (no dependency on the Mac): wait for exp-130-stock's comparison, stop phase4e before its blockc arm, run exp-131.
cd "$HOME/kernelforge" || exit 1
PY=$HOME/kf-video-venv/bin/python
until grep -q "^latents runs/exp-130-stock" runs/jobs/phase4e.log; do sleep 15; done
/opt/homebrew/bin/tmux kill-session -t kf-phase4e; pkill -f "exp-130-blockc"; sleep 3
$PY bench/latent_compare.py runs/exp-130-stock/latents.safetensors runs/exp-127-composed/latents.safetensors
$PY bench/cooldown.py --min 120
$PY -m adapters.wan.bench_wan --exp exp-131-ctxref-stream --prompt v1 --width 832 --height 480 --frames 33 --steps 50 --seed 42 \
  --context-ref baselines_video/wan21/v1_text_context.safetensors --dit stream --stop-after-denoise --cache-limit-gb 0.5
$PY bench/latent_compare.py runs/exp-131-ctxref-stream/latents.safetensors baselines_video/wan21/v1_latents.safetensors
