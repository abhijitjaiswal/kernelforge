#!/bin/sh
# Caching A2 on the composed low-memory pipeline (v1), its quality gate, then exact v2/v3 references for the sweep.
cd "$HOME/kernelforge" || exit 1
PY=$HOME/kf-video-venv/bin/python
B="$PY -m adapters.wan.bench_wan --width 832 --height 480 --frames 33 --steps 50 --seed 42 --te-stream --dit stream --vae-stream --cache-limit-gb 0.5"
echo "== exp-134-a2"; $PY bench/cooldown.py --min 120
$B --exp exp-134-a2 --prompt v1 --cache-schedule a2
$HOME/kf-venv/bin/python bench/video_gate.py runs/exp-134-a2 runs/exp-127-composed --latents-ref runs/exp-130-stock --out runs/exp-134-a2/gate.json
for p in v2 v3; do
  echo "== exp-ref-$p"; $PY bench/cooldown.py --min 120
  $B --exp exp-ref-$p --prompt $p
done
