# RESULTS (newest first)

### exp-041: Exp A1 rerun with mean-centered token similarity
- total 0.00 s (denoise 0.00, VAE 0.00, text 0.00); peak footprint None GB, MLX peak None GB; swap grew: None
- tier A, verdict **keep**. Centered near-dup >0.95: 0.016% (0% per prompt). The raw 53.9% is anisotropy (a direction shared by all tokens), not redundancy; median centered max-cos 0.38-0.84.

### exp-040: Exp A1 probe: block-step output similarity + token near-duplicates, bf16 4 steps, p1-p3 (eager, hooked)
- total 0.00 s (denoise 0.00, VAE 0.00, text 0.00); peak footprint None GB, MLX peak None GB; swap grew: None
- tier A, verdict **keep**. Block-steps cos_out>0.98: 28.4% (p1 34.7, p2 37.3, p3 13.3); own-contribution cos>0.98 only 4.9%. Only double blocks 0-3 are static (~13% of compute) -> caching ceiling ~1.09x. Raw token near-dup 53.9%.

### exp-032: per-layer quantization sensitivity scan (engine/sensitivity.py), 109 Linears x {q8,q4}, one real denoise step
- total 0.00 s (denoise 0.00, VAE 0.00, text 0.00); peak footprint None GB, MLX peak None GB; swap grew: None
- tier A, verdict **keep**. Metric floor verified 0 (deterministic; fp32 recompute bit-identical). q8 per-layer rel 1.9e-3..2.3e-2 (median 1.0e-2); q4 5.6e-3..5.7e-2. Most sensitive: embedders, modulation, proj_out, first blocks; least: double-block attn q/k, ff, norm_out.

### exp-031-arch-g16g: microbench with MLX_METAL_GPU_ARCH=applegpu_g16g (pre-M5: no NAX kernels)
- total 0.00 s (denoise 0.00, VAE 0.00, text 0.00); peak footprint None GB, MLX peak None GB; swap grew: None
- tier E, verdict **keep**. Matmul 2.9-3.0 TF and SDPA 2.5 TF without NAX: the baseline already gets ~3.7x from the M5 neural accelerators.

### exp-031-arch-default: microbench: matmul peak, bandwidth, Klein linear + SDPA shapes, detected arch
- total 0.00 s (denoise 0.00, VAE 0.00, text 0.00); peak footprint None GB, MLX peak None GB; swap grew: None
- tier E, verdict **keep**. bf16 peak 11.5 TF, Klein matmuls 10-11.5 TF (compute-bound, ~95% of peak); SDPA 8.7 TF; 135 GB/s; q8/q4 matmuls slower than bf16.

### exp-030: denoise op profile, one step bf16 1024^2 (MLX eager per-module; xctrace unavailable: no Xcode)
- total 3.87 s (denoise 0.00, VAE 0.00, text 0.00); peak footprint None GB, MLX peak 8.521366279572248 GB; swap grew: None
- tier E, verdict **keep**. Compiled step 3.87 s (eager 4.41). Single-stream blocks 77%: to_qkv_mlp_proj 36%, to_out 15%, attention+RoPE ~23%. ~35 TFLOP/step -> ~9 TFLOPS achieved.

### exp-022-cap6: simulated 6 GB limit: mx.set_memory_limit(6 GB), bf16 + chunked VAE
- total 23.59 s (denoise 19.15, VAE 3.39, text 1.04); peak footprint 8.059 GB, MLX peak 7.704 GB; swap grew: False
- PSNR vs ref: p1 52.3 dB, max abs diff 42
- tier E, verdict **inconclusive**. SIMULATION. Soft limit exceeded (MLX 7.70 GB, OS 8.06 GB); denoise 15.9 -> 19.2 s as MLX frees buffers early.

### exp-022-cap8: simulated 8 GB machine: mx.set_memory_limit(8 GB), bf16 + chunked VAE
- total 20.37 s (denoise 15.96, VAE 3.36, text 1.03); peak footprint 8.537 GB, MLX peak 8.205 GB; swap grew: False
- PSNR vs ref: p1 52.3 dB, max abs diff 42
- tier E, verdict **inconclusive**. SIMULATION. MLX limit is soft: completes, MLX peak 8.21 GB, OS 8.54 GB (over 8), denoise unaffected.

### exp-021-chunk-3: A/B round 3: full pipeline bf16 staged, chunked VAE
- total 19.72 s (denoise 15.73, VAE 2.93, text 1.04); peak footprint 9.376 GB, MLX peak 8.574 GB; swap grew: False
- PSNR vs ref: p1 52.3 dB, max abs diff 42
- tier E, verdict **keep**. peak moves to denoise (8.57 GB MLX)

### exp-021-stock-3: A/B round 3: full pipeline bf16 staged, stock VAE
- total 19.07 s (denoise 15.91, VAE 2.09, text 1.03); peak footprint 10.277 GB, MLX peak 9.495 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. reference for chunked decode

### exp-021-chunk-2: A/B round 2: full pipeline bf16 staged, chunked VAE
- total 20.26 s (denoise 15.88, VAE 3.33, text 1.04); peak footprint 9.175 GB, MLX peak 8.574 GB; swap grew: False
- PSNR vs ref: p1 52.3 dB, max abs diff 42
- tier E, verdict **keep**. peak moves to denoise (8.57 GB MLX)

### exp-021-stock-2: A/B round 2: full pipeline bf16 staged, stock VAE
- total 19.35 s (denoise 16.19, VAE 2.09, text 1.04); peak footprint 9.741 GB, MLX peak 9.495 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. reference for chunked decode

### exp-021-chunk-1: A/B round 1: full pipeline bf16 staged, chunked VAE
- total 20.15 s (denoise 15.75, VAE 3.35, text 1.03); peak footprint 9.377 GB, MLX peak 8.574 GB; swap grew: True
- PSNR vs ref: p1 52.3 dB, max abs diff 42
- tier E, verdict **keep**. peak moves to denoise (8.57 GB MLX)

### exp-021-stock-1: A/B round 1: full pipeline bf16 staged, stock VAE
- total 19.10 s (denoise 15.90, VAE 2.10, text 1.07); peak footprint 10.19 GB, MLX peak 9.495 GB; swap grew: True
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. reference for chunked decode

### exp-020-chunk256-p1: chunked VAE decode, 256 rows, p1
- total 2.95 s (denoise 0.00, VAE 2.95, text 0.00); peak footprint 2.767 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p1 52.6 dB, max abs diff 24
- tier E, verdict **keep**. chunk size barely matters

### exp-020-chunk64-p1: chunked VAE decode, 64 rows, p1
- total 2.76 s (denoise 0.00, VAE 2.76, text 0.00); peak footprint 2.411 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p1 52.1 dB, max abs diff 21
- tier E, verdict **keep**. chunk size barely matters

### exp-019-chunk128-p3: exact row-chunked VAE decode (global GroupNorm stats, 128 rows), p3
- total 2.84 s (denoise 0.00, VAE 2.84, text 0.00); peak footprint 2.464 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p3 46.1 dB, max abs diff 36
- tier E, verdict **keep**. 1.81 GB MLX, Tier E; float-rounding differences only (scattered, not on chunk edges)

### exp-018-barriers-p3: VAE decode with mx.eval barriers between blocks, p3
- total 2.20 s (denoise 0.00, VAE 2.20, text 0.00); peak footprint 7.276 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p3 999.0 dB, max abs diff 0
- tier E, verdict **keep**. bit-identical, 9.55 -> 7.26 GB

### exp-017-stock-p3: VAE-only stock decode, p3 (reference)
- total 2.07 s (denoise 0.00, VAE 2.07, text 0.00); peak footprint 10.293 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p3 999.0 dB, max abs diff 0
- tier E, verdict **keep**. 9.55 GB MLX for the decoder alone

### exp-019-chunk128-p2: exact row-chunked VAE decode (global GroupNorm stats, 128 rows), p2
- total 2.82 s (denoise 0.00, VAE 2.82, text 0.00); peak footprint 2.397 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p2 45.2 dB, max abs diff 52
- tier E, verdict **keep**. 1.81 GB MLX, Tier E; float-rounding differences only (scattered, not on chunk edges)

### exp-018-barriers-p2: VAE decode with mx.eval barriers between blocks, p2
- total 2.20 s (denoise 0.00, VAE 2.20, text 0.00); peak footprint 7.276 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p2 999.0 dB, max abs diff 0
- tier E, verdict **keep**. bit-identical, 9.55 -> 7.26 GB

### exp-017-stock-p2: VAE-only stock decode, p2 (reference)
- total 2.20 s (denoise 0.00, VAE 2.20, text 0.00); peak footprint 10.293 GB, MLX peak None GB; swap grew: True
- PSNR vs ref: p2 999.0 dB, max abs diff 0
- tier E, verdict **keep**. 9.55 GB MLX for the decoder alone

### exp-019-chunk128-p1: exact row-chunked VAE decode (global GroupNorm stats, 128 rows), p1
- total 2.93 s (denoise 0.00, VAE 2.93, text 0.00); peak footprint 2.421 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p1 52.3 dB, max abs diff 42
- tier E, verdict **keep**. 1.81 GB MLX, Tier E; float-rounding differences only (scattered, not on chunk edges)

### exp-018-barriers-p1: VAE decode with mx.eval barriers between blocks, p1
- total 2.19 s (denoise 0.00, VAE 2.19, text 0.00); peak footprint 7.276 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. bit-identical, 9.55 -> 7.26 GB

### exp-017-stock-p1: VAE-only stock decode, p1 (reference)
- total 2.07 s (denoise 0.00, VAE 2.07, text 0.00); peak footprint 10.663 GB, MLX peak None GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. 9.55 GB MLX for the decoder alone

### exp-016-lat-p3: stage bench single run to save p3 final latents (bf16)
- total 19.64 s (denoise 17.17, VAE 2.13, text 0.33); peak footprint 10.319 GB, MLX peak 9.559 GB; swap grew: True
- PSNR vs ref: p3 999.0 dB, max abs diff 0
- tier E, verdict **keep**. latents for VAE-only tests; chained output bit-identical to reference

### exp-016-lat-p2: stage bench single run to save p2 final latents (bf16)
- total 19.35 s (denoise 16.88, VAE 2.10, text 0.37); peak footprint 10.516 GB, MLX peak 9.559 GB; swap grew: True
- PSNR vs ref: p2 999.0 dB, max abs diff 0
- tier E, verdict **keep**. latents for VAE-only tests; chained output bit-identical to reference

### exp-015-q8-3: stage bench round 3: q8 staged, cache cap 0.5 GB, load separated, 5 warm runs
- total 23.96 s (denoise 21.25, VAE 2.32, text 0.39); peak footprint 10.909 GB, MLX peak 9.556 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. compute-only A/B

### exp-015-q8-2: stage bench round 2: q8 staged, cache cap 0.5 GB, load separated, 5 warm runs
- total 22.97 s (denoise 20.41, VAE 2.18, text 0.37); peak footprint 10.886 GB, MLX peak 9.556 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. compute-only A/B

### exp-015-q8-1: stage bench round 1: q8 staged, cache cap 0.5 GB, load separated, 5 warm runs
- total 22.87 s (denoise 20.32, VAE 2.18, text 0.37); peak footprint 10.942 GB, MLX peak 9.556 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. compute-only A/B

### exp-015-bf16-3: stage bench round 3: bf16 staged, cache cap 0.5 GB, load separated, 5 warm runs
- total 20.82 s (denoise 18.24, VAE 2.25, text 0.34); peak footprint 10.874 GB, MLX peak 9.559 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. compute-only A/B

### exp-015-bf16-2: stage bench round 2: bf16 staged, cache cap 0.5 GB, load separated, 5 warm runs
- total 20.70 s (denoise 18.20, VAE 2.16, text 0.33); peak footprint 10.879 GB, MLX peak 9.559 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. compute-only A/B

### exp-015-bf16-1: stage bench round 1: bf16 staged, cache cap 0.5 GB, load separated, 5 warm runs
- total 20.42 s (denoise 17.93, VAE 2.16, text 0.33); peak footprint 10.869 GB, MLX peak 9.559 GB; swap grew: False
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. compute-only A/B

### exp-014: VAE decoder per-module memory profile (bf16, real p1 latents)
- total 2.05 s (denoise 0.00, VAE 2.05, text 0.00); peak footprint None GB, MLX peak 9.552989134564996 GB; swap grew: None
- PSNR vs ref: p1 999.0 dB, max abs diff None
- tier E, verdict **keep**. Fused decode peaks 9.55 GB; eager per-module max is 4.33 GB with bit-identical output, so most of the 9.3 GB is lazy-graph lifetime. Hot ops: 1024^2 upsample conv +3.05 GB, full-res GroupNorm +2.0 GB. Mid-block attention +0.09 GB: no need to chunk it.

### exp-013: q8 + tiled VAE images for p1/p2/p3 (quality grid)
- total 22.81 s (denoise 19.70, VAE 2.73, text 0.36); peak footprint 6.163 GB, MLX peak 5.297 GB; swap grew: False
- PSNR vs ref: p1 26.5, p2 26.3, p3 27.1 dB, max abs diff 122
- tier A, verdict **revert**. vs bf16: PSNR 24.7/23.8/19.8, LPIPS 0.058/0.090/0.110 (fail, >0.05). Tiling alone vs untiled q8: LPIPS 0.035/0.055/0.036 and a visible vertical seam in p1 (per-tile GroupNorm stats).

### exp-010-rep-3: repeat 3 of exp-010 (q8 staged + tiled VAE + cache cap 0.5 GB)
- total 20.48 s (denoise 17.28, VAE 2.70, text 0.49); peak footprint 6.049 GB, MLX peak 5.19 GB; swap grew: False
- PSNR vs ref: p1 26.5 dB, max abs diff 55
- tier A, verdict **revert**. Footprint 6.04-6.05 GB confirmed over 4 runs, but the config fails Tier A (LPIPS vs bf16 0.058-0.110, visible tile seam).

### exp-010-rep-2: repeat 2 of exp-010 (q8 staged + tiled VAE + cache cap 0.5 GB)
- total 20.42 s (denoise 17.23, VAE 2.68, text 0.49); peak footprint 6.043 GB, MLX peak 5.19 GB; swap grew: False
- PSNR vs ref: p1 26.5 dB, max abs diff 55
- tier A, verdict **revert**. Footprint 6.04-6.05 GB confirmed over 4 runs, but the config fails Tier A (LPIPS vs bf16 0.058-0.110, visible tile seam).

### exp-010-rep-1: repeat 1 of exp-010 (q8 staged + tiled VAE + cache cap 0.5 GB)
- total 20.31 s (denoise 17.12, VAE 2.69, text 0.49); peak footprint 6.044 GB, MLX peak 5.19 GB; swap grew: False
- PSNR vs ref: p1 26.5 dB, max abs diff 55
- tier A, verdict **revert**. Footprint 6.04-6.05 GB confirmed over 4 runs, but the config fails Tier A (LPIPS vs bf16 0.058-0.110, visible tile seam).

## Summary

<!-- summary:start -->
Speedup vs **exp-011-bf16** = 18.92 s (median of 3). *memory-only* = swap grew, so the timing is invalid. PSNR is against the run's own quality reference (`=` means bit-identical). Stage-bench and VAE-only runs have no end-to-end total.

| Exp | Config | Cache cap | Quality ref | Total s | Speedup | MLX peak GB | OS peak GB | PSNR dB per prompt | Swap |
|---|---|---|---|---|---|---|---|---|---|
| exp-000-smoke | q8 · free_te | none | – | 24.85 | memory-only | 9.49 | 12.40 | – | grew |
| exp-001 | q8 · free_te | none | – | 25.42 | memory-only | 9.51 | 12.65 | – | grew |
| exp-002 | q8 · stock | none | q8_free_te | 26.61 | memory-only | 15.99 | 14.12 | p1 = · p2 = · p3 = | grew |
| exp-003 | bf16 · free_te | none | q8_free_te | 26.67 | memory-only | 9.51 | 12.32 | p1 29.4 · p2 27.5 · p3 20.7 | grew |
| exp-004 | q8 · free_te · tiled VAE | none | q8_free_te | 39.70 | memory-only | 5.30 | 11.90 | p1 26.5 · p2 26.3 · p3 27.1 | grew |
| exp-005 | q8 · free_te | none | q8_free_te | 21.65 | memory-only | 9.49 | 11.92 | p1 = | grew |
| exp-006 | q8 · staged | none | q8_free_te | 20.88 | memory-only | 9.49 | 11.70 | p1 = | grew |
| exp-007 | bf16 · staged | none | bf16_free_te | 19.50 | 0.97× | 9.49 | 11.54 | p1 = | flat |
| exp-008 | q8 · staged · tiled VAE | none | q8_free_te | 21.10 | 0.90× | 5.19 | 8.71 | p1 26.5 | flat |
| exp-009 | bf16 · staged | 0.5 GB | bf16_free_te | 19.57 | 0.97× | 9.49 | 10.26 | p1 = | flat |
| exp-010 | q8 · staged · tiled VAE | 0.5 GB | q8_free_te | 20.66 | 0.92× | 5.19 | 6.04 | p1 26.5 | flat |
| exp-010-rep-1 | q8 · staged · tiled VAE | 0.5 GB | q8_free_te | 20.31 | 0.93× | 5.19 | 6.04 | p1 26.5 | flat |
| exp-010-rep-2 | q8 · staged · tiled VAE | 0.5 GB | q8_free_te | 20.42 | 0.93× | 5.19 | 6.04 | p1 26.5 | flat |
| exp-010-rep-3 | q8 · staged · tiled VAE | 0.5 GB | q8_free_te | 20.48 | 0.92× | 5.19 | 6.05 | p1 26.5 | flat |
| exp-011-bf16-1 | bf16 · staged | 0.5 GB | bf16_free_te | 19.20 | 0.99× | 9.49 | 10.08 | p1 = | flat |
| exp-011-bf16-2 | bf16 · staged | 0.5 GB | bf16_free_te | 18.86 | 1.00× | 9.49 | 10.24 | p1 = | flat |
| exp-011-bf16-3 | bf16 · staged | 0.5 GB | bf16_free_te | 18.92 | 1.00× | 9.49 | 9.74 | p1 = | flat |
| exp-011-q8-1 | q8 · staged | 0.5 GB | q8_free_te | 20.02 | 0.95× | 9.49 | 9.80 | p1 = | flat |
| exp-011-q8-2 | q8 · staged | 0.5 GB | q8_free_te | 20.00 | 0.95× | 9.49 | 9.80 | p1 = | flat |
| exp-011-q8-3 | q8 · staged | 0.5 GB | q8_free_te | 20.04 | 0.94× | 9.49 | 9.80 | p1 = | flat |
| exp-012-smoke | bf16 · staged | 0.5 GB | bf16_free_te | – | memory-only | – | 10.82 | p1 = | grew |
| exp-013 | q8 · free_te · tiled VAE | 0.5 GB | q8_free_te | 22.81 | 0.83× | 5.30 | 6.16 | p1 26.5 · p2 26.3 · p3 27.1 | flat |
| exp-014 | bf16 | none | – | – | – | – | – | p1 = | – |
| exp-015-bf16-1 | bf16 · staged | 0.5 GB | bf16_free_te | – | – | – | 10.87 | p1 = | flat |
| exp-015-bf16-2 | bf16 · staged | 0.5 GB | bf16_free_te | – | – | – | 10.88 | p1 = | flat |
| exp-015-bf16-3 | bf16 · staged | 0.5 GB | bf16_free_te | – | – | – | 10.87 | p1 = | flat |
| exp-015-q8-1 | q8 · staged | 0.5 GB | q8_free_te | – | – | – | 10.94 | p1 = | flat |
| exp-015-q8-2 | q8 · staged | 0.5 GB | q8_free_te | – | – | – | 10.89 | p1 = | flat |
| exp-015-q8-3 | q8 · staged | 0.5 GB | q8_free_te | – | – | – | 10.91 | p1 = | flat |
| exp-016-lat-p2 | bf16 · staged | 0.5 GB | bf16_free_te | – | memory-only | – | 10.52 | p2 = | grew |
| exp-016-lat-p3 | bf16 · staged | 0.5 GB | bf16_free_te | – | memory-only | – | 10.32 | p3 = | grew |
| exp-017-stock-p1 | bf16 | 0.5 GB | bf16_free_te | – | – | 9.55 | 10.66 | p1 = | flat |
| exp-017-stock-p2 | bf16 | 0.5 GB | bf16_free_te | – | memory-only | 9.55 | 10.29 | p2 = | grew |
| exp-017-stock-p3 | bf16 | 0.5 GB | bf16_free_te | – | – | 9.55 | 10.29 | p3 = | flat |
| exp-018-barriers-p1 | bf16 | 0.5 GB | bf16_free_te | – | – | 7.26 | 7.28 | p1 = | flat |
| exp-018-barriers-p2 | bf16 | 0.5 GB | bf16_free_te | – | – | 7.26 | 7.28 | p2 = | flat |
| exp-018-barriers-p3 | bf16 | 0.5 GB | bf16_free_te | – | – | 7.26 | 7.28 | p3 = | flat |
| exp-019-chunk128-p1 | bf16 | 0.5 GB | bf16_free_te | – | – | 1.81 | 2.42 | p1 52.3 | flat |
| exp-019-chunk128-p2 | bf16 | 0.5 GB | bf16_free_te | – | – | 1.81 | 2.40 | p2 45.2 | flat |
| exp-019-chunk128-p3 | bf16 | 0.5 GB | bf16_free_te | – | – | 1.81 | 2.46 | p3 46.1 | flat |
| exp-020-chunk64-p1 | bf16 | 0.5 GB | bf16_free_te | – | – | 1.81 | 2.41 | p1 52.1 | flat |
| exp-020-chunk256-p1 | bf16 | 0.5 GB | bf16_free_te | – | – | 1.89 | 2.77 | p1 52.6 | flat |
| exp-021-chunk-1 | bf16 · staged | 0.5 GB | bf16_free_te | 20.15 | memory-only | 8.57 | 9.38 | p1 52.3 | grew |
| exp-021-chunk-2 | bf16 · staged | 0.5 GB | bf16_free_te | 20.26 | 0.93× | 8.57 | 9.18 | p1 52.3 | flat |
| exp-021-chunk-3 | bf16 · staged | 0.5 GB | bf16_free_te | 19.72 | 0.96× | 8.57 | 9.38 | p1 52.3 | flat |
| exp-021-stock-1 | bf16 · staged | 0.5 GB | bf16_free_te | 19.10 | memory-only | 9.49 | 10.19 | p1 = | grew |
| exp-021-stock-2 | bf16 · staged | 0.5 GB | bf16_free_te | 19.35 | 0.98× | 9.49 | 9.74 | p1 = | flat |
| exp-021-stock-3 | bf16 · staged | 0.5 GB | bf16_free_te | 19.07 | 0.99× | 9.49 | 10.28 | p1 = | flat |
| exp-022-cap6 | bf16 · staged | 0.5 GB | bf16_free_te | 23.59 | 0.80× | 7.70 | 8.06 | p1 52.3 | flat |
| exp-022-cap8 | bf16 · staged | 0.5 GB | bf16_free_te | 20.37 | 0.93× | 8.21 | 8.54 | p1 52.3 | flat |
| exp-030 | bf16 | none | – | – | – | – | – | – | – |
| exp-031-arch-default | microbench | none | – | – | – | – | – | – | – |
| exp-031-arch-g16g | microbench | none | – | – | – | – | – | – | – |
| exp-032 | bf16 | none | – | – | – | – | – | – | – |
| exp-040 | bf16 | none | – | – | – | – | – | – | – |
| exp-041 | bf16 | none | – | – | – | – | – | – | – |
<!-- summary:end -->

## Log

### exp-011-q8-3: A/B round 3: q8 staged, cache cap 0.5 GB
- total 20.04 s (denoise 17.42, VAE 2.09, text 0.49); peak footprint 9.799 GB, MLX peak 9.492 GB; swap grew: False
- vs exp-011-bf16-3: 0.9445x speed, -0.0059 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier A, verdict **revert**. Alternating A/B with 150 s cool-downs.

### exp-011-bf16-3: A/B round 3: bf16 staged, cache cap 0.5 GB
- total 18.92 s (denoise 15.74, VAE 2.10, text 1.05); peak footprint 9.742 GB, MLX peak 9.495 GB; swap grew: False
- vs exp-011-q8-3: 1.0588x speed, 0.0058 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. Alternating A/B with 150 s cool-downs.

### exp-011-q8-2: A/B round 2: q8 staged, cache cap 0.5 GB
- total 20.00 s (denoise 17.39, VAE 2.09, text 0.49); peak footprint 9.8 GB, MLX peak 9.492 GB; swap grew: False
- vs exp-011-bf16-2: 0.9427x speed, 0.0434 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier A, verdict **revert**. Alternating A/B with 150 s cool-downs.

### exp-011-bf16-2: A/B round 2: bf16 staged, cache cap 0.5 GB
- total 18.86 s (denoise 15.69, VAE 2.10, text 1.04); peak footprint 10.245 GB, MLX peak 9.495 GB; swap grew: False
- vs exp-011-q8-2: 1.0608x speed, -0.0454 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. Alternating A/B with 150 s cool-downs.

### exp-011-q8-1: A/B round 1: q8 staged, cache cap 0.5 GB
- total 20.02 s (denoise 17.42, VAE 2.08, text 0.49); peak footprint 9.801 GB, MLX peak 9.492 GB; swap grew: False
- vs exp-011-bf16-1: 0.9589x speed, 0.0273 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier A, verdict **revert**. Alternating A/B with 150 s cool-downs.

### exp-011-bf16-1: A/B round 1: bf16 staged, cache cap 0.5 GB
- total 19.20 s (denoise 16.04, VAE 2.09, text 1.04); peak footprint 10.076 GB, MLX peak 9.495 GB; swap grew: False
- vs exp-011-q8-1: 1.0428x speed, -0.0281 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. Alternating A/B with 150 s cool-downs.

### exp-010: q8 staged + tiled VAE + cache cap 0.5 GB
- total 20.66 s (denoise 17.47, VAE 2.69, text 0.49); peak footprint 6.043 GB, MLX peak 5.19 GB; swap grew: False
- vs exp-005: 1.0481x speed, 0.493 memory reduction
- PSNR vs ref: p1 26.5 dB, max abs diff 55
- tier A, verdict **keep**. Lowest footprint: 6.04 GB OS / 5.19 GB MLX at 20.7 s, swap flat. Tier A (tiled decode PSNR 26.5 dB vs untiled); LPIPS and visual check pending.

### exp-009: MLX cache cap 0.5 GB (bf16 staged)
- total 19.57 s (denoise 16.34, VAE 2.16, text 1.04); peak footprint 10.258 GB, MLX peak 9.495 GB; swap grew: False
- vs exp-007: 0.9962x speed, 0.1111 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. OS footprint 11.54 -> 10.26 GB, pixel-identical, no speed cost (19.6 vs 19.5 s), swap flat.

### exp-008: staged lifecycle + tiled VAE decode (q8)
- total 21.10 s (denoise 18.05, VAE 2.55, text 0.49); peak footprint 8.706 GB, MLX peak 5.19 GB; swap grew: False
- vs exp-005: 1.026x speed, 0.2695 memory reduction
- PSNR vs ref: p1 26.5 dB, max abs diff 55
- tier A, verdict **keep**. Lowest so far: MLX 5.19 GB, OS 8.7 GB, no swap growth. VAE 2.6 s when cool (the 7.5 s in exp-004 was mostly heat). Output differs from untiled (PSNR 26.5 dB): Tier A, needs LPIPS.

### exp-007: lifecycle staged with bf16 weights
- total 19.50 s (denoise 16.40, VAE 2.04, text 1.04); peak footprint 11.54 GB, MLX peak 9.495 GB; swap grew: False
- vs exp-005: 1.1105x speed, 0.0317 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. Fastest so far (19.5 s, denoise 16.4 s) and first run with no swap growth. bf16 denoise is faster than q8 (18.3 s) when both are cool: the 8-bit dequant costs time.

### exp-006: lifecycle staged: also free the transformer before VAE decode (q8)
- total 20.88 s (denoise 18.33, VAE 2.04, text 0.49); peak footprint 11.698 GB, MLX peak 9.492 GB; swap grew: True
- vs exp-005: 1.0367x speed, 0.0185 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. MLX peak unchanged at 9.49 GB: the VAE decoder alone needs ~9.5 GB of activations at 1024^2. OS footprint -0.2 GB only. Pixel-identical.

### exp-005: cold single image: q8 free_te with memtrace (reference for staged lifecycle)
- total 21.65 s (denoise 18.47, VAE 2.66, text 0.49); peak footprint 11.918 GB, MLX peak 9.492 GB; swap grew: True
- vs exp-001: 1.1742x speed, 0.0577 memory reduction
- PSNR vs ref: p1 999.0 dB, max abs diff 0
- tier E, verdict **keep**. Peak is in VAE decode: footprint 11.9 GB = MLX active 5.0 + MLX cache 6.3 GB. Cache explains the OS/MLX gap.

### exp-004: VAE tiled decode (mflux TilingConfig defaults), q8 free_te
- total 39.70 s (denoise 31.95, VAE 7.52, text 0.39); peak footprint 11.901 GB, MLX peak 5.297 GB; swap grew: True
- vs exp-001: 0.6404x speed, 0.0591 memory reduction
- PSNR vs ref: p1 26.510566998549024, p2 26.275264999274107, p3 27.10092174227129 dB, max abs diff 122
- tier A, verdict **inconclusive**. MLX peak 9.5 -> 5.3 GB but VAE 2.8 -> 7.5 s and output differs from the untiled decode (PSNR 26-27 dB), so not Tier E. Denoise 32 s vs 22 s shows thermal drift; needs an interleaved re-test.

### exp-003: precision: bf16 weights (HF snapshot), free_te
- total 26.67 s (denoise 22.23, VAE 3.98, text 0.36); peak footprint 12.325 GB, MLX peak 9.511 GB; swap grew: True
- vs exp-001: 0.9532x speed, 0.0255 memory reduction
- PSNR vs ref: p1 29.377667336114783, p2 27.510796738178463, p3 20.69256020204857 dB, max abs diff 241
- tier E, verdict **keep**. bf16 denoise 22.2 s = q8 22.0 s: 8-bit weights give no speed, so denoise is not weight-bandwidth bound (or dequant cost cancels it). Saved as bf16 reference images.

### exp-002: lifecycle: stock (all components resident, mflux default)
- total 26.61 s (denoise 22.77, VAE 2.78, text 1.12); peak footprint 14.116 GB, MLX peak 15.99 GB; swap grew: True
- vs exp-001: 0.9554x speed, -0.1161 memory reduction
- PSNR vs ref: p1 999.0, p2 999.0, p3 999.0 dB, max abs diff 0
- tier E, verdict **revert**. Lifecycle scheduling (free_te) is Tier E (pixel-identical) and saves 6.5 GB of MLX peak (16.0 -> 9.5) and 1.5 GB OS footprint. Stock pushes 1.8 GB into swap.

### exp-001: baseline: mflux 0.23 q8 (saved 8-bit), text encoder freed after encoding (free_te)
- total 25.42 s (denoise 21.99, VAE 2.78, text 0.36); peak footprint 12.648 GB, MLX peak 9.508 GB; swap grew: True
- tier A, verdict **keep**. Baseline. Denoise is 87% of time; VAE decode sets the MLX peak (5.3 -> 9.5 GB). Swap grew 0.7 GB, so timing carries a caveat. Tier A: 8-bit vs bf16 PSNR 20.7-29.4 dB.

