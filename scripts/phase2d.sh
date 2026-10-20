#!/bin/sh
# Diagnose p2/p3 divergence of the final pipeline vs the baseline image (all p3, chunked VAE decode, resident TE).
#  exp-063-rcomp-p3   resident, mflux whole-model compile      -> isolates the chunked decoder end to end
#  exp-063-rblockc-p3 resident, per-block compile              -> isolates per-block compilation; same-mode ref for exp-062-full-p3
#  exp-063-eager-p3   resident, eager                          -> numerics of an eager (streaming-compatible) pipeline
# then LPIPS + grid of the final pipeline p1-p3 vs the baseline.
PY=$HOME/kf-venv/bin/python
cool() { $PY bench/cooldown.py --min ${1:-150}; sysctl vm.swapusage; }
for m in rcomp:resident_compiled rblockc:resident_blockc eager:resident; do
  tag=${m%%:*}; mode=${m##*:}
  echo "== exp-063-$tag-p3"; cool
  $PY -m adapters.flux2_klein.stream_bench --exp exp-063-$tag-p3 --prompt p3 --mode $mode --runs 2 --ref runs/exp-062-full-p3
done
mkdir -p runs/exp-064-final-quality/cand
cp runs/exp-061-full-1/p1.png runs/exp-062-full-p2/p2.png runs/exp-062-full-p3/p3.png runs/exp-064-final-quality/cand/
$PY bench/quality.py --cand runs/exp-064-final-quality/cand --ref baselines/bf16_free_te --out runs/exp-064-final-quality --label "final 3.5 GB pipeline vs bf16 baseline"
