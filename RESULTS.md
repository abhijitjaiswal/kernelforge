# RESULTS (newest first)

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

