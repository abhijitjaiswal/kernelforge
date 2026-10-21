#!/bin/sh
# After the exact v2/v3 references (phase4g): caching A2 on v2 and v3 with their quality gates.
cd "$HOME/kernelforge" || exit 1
PY=$HOME/kf-video-venv/bin/python
until grep -q "^EXIT=" runs/jobs/phase4g.log; do sleep 15; done
B="$PY -m adapters.wan.bench_wan --width 832 --height 480 --frames 33 --steps 50 --seed 42 --te-stream --dit stream --vae-stream --cache-limit-gb 0.5"
for p in v2 v3; do
  echo "== exp-a2-$p"; $PY bench/cooldown.py --min 120
  $B --exp exp-a2-$p --prompt $p --cache-schedule a2
  $HOME/kf-venv/bin/python bench/video_gate.py runs/exp-a2-$p runs/exp-ref-$p --latents-ref runs/exp-ref-$p --out runs/exp-a2-$p/gate.json
done
