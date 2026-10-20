#!/bin/sh
# Phase 2 (2): text-encoder memory anatomy + exact streaming. Each variant in its own process; 60 s cool-downs (short runs).
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-60}; }
[ -f models/klein-te-bf16/layer026.safetensors ] || $PY -m adapters.flux2_klein.te_probe --prep
du -sh models/klein-te-bf16
for i in 1 2; do
  echo "== exp-060-stock-$i"; cool
  $PY -m adapters.flux2_klein.te_probe --exp exp-060-stock-$i --variant stock
  for v in barrier truncate stream; do
    echo "== exp-060-$v-$i"; cool
    $PY -m adapters.flux2_klein.te_probe --exp exp-060-$v-$i --variant $v --ref runs/exp-060-stock-1
  done
done
