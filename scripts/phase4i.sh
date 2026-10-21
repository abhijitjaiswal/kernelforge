#!/bin/sh
# A3-d dynamic skip (threshold 0.022 on the block-stack change rate), hardest prompt first: neon (v3), then waves (v2).
cd "$HOME/kernelforge" || exit 1
PY=$HOME/kf-video-venv/bin/python
B="$PY -m adapters.wan.bench_wan --width 832 --height 480 --frames 33 --steps 50 --seed 42 --te-stream --dit stream --vae-stream --cache-limit-gb 0.5"
for p in v3 v2; do
  echo "== exp-a3d-$p"; $PY bench/cooldown.py --min 120
  $B --exp exp-a3d-$p --prompt $p --cache-schedule dyn --cache-threshold 0.022
  $HOME/kf-venv/bin/python bench/video_gate.py runs/exp-a3d-$p runs/exp-ref-$p --latents-ref runs/exp-ref-$p --out runs/exp-a3d-$p/gate.json
done
