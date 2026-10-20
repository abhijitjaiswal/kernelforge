# Klein 4B on Apple M5 (16 GB): current state

*Draft, 2026-10-20. Measured on m5-test (MacBook Air M5, 8-core GPU, 16 GB, macOS 26.6); MLX 0.32.4, mflux 0.23.0.
1024², 4 steps, seed 42, prompts p1–p3. Every number traces to `runs/<exp>/result.json`.*

## Headline

**Exact (bit-identical) memory reduction works: OS peak 10.1 GB → 6.65 GB, with no change to the image.** It costs about
10 % of denoise time today. Recovering that cost (per-block compile) is in progress. **There is no low-hanging speed left in
Klein T2I on this machine**: the matmuls already run at ~95 % of the M5 neural-accelerator peak.

## Where we started vs where we are

| | Baseline (Gate 0) | Now: exact pipeline |
|---|---|---|
| Config | bf16, text encoder freed, MLX cache cap 0.5 GB, compiled | + exact chunked VAE decode + transformer weight streaming (2 blocks resident, prefetch) |
| **End-to-end per image** | **18.92 s** (exp-011, median of 3, cold process) | **pending exp-053** (*estimate* ≈ 23–24 s eager; the compiled-streaming target is ≤ 5 % over resident) |
| Denoise (4 steps) | 15.74 s compiled (cold) | 20.0 s eager streaming, cold SSD (exp-052). The eager-resident reference is 18.2 s |
| **OS peak footprint** | **10.1 GB** (exp-011) | **6.65 GB** (exp-051/052) |
| Image vs reference | — | **bit-identical** to the same-mode resident run (10/10 runs; protocol rule R1) |

*Note:* the "~19.8 s" figure is denoise-only and eager, so it is not comparable to the 18.92 s end-to-end baseline.
End-to-end streaming numbers come from exp-053.

## What moved memory (all Tier E)

| Stage | Before | After | How | Exp |
|---|---|---|---|---|
| VAE decode | 9.55 GB | **1.81 GB** | exact row-chunked conv/upsample with GroupNorm stats computed over the full latent; 52 / 45 / 46 dB vs reference | exp-014, 019 |
| Denoise (transformer) | 8.52 GB | **2.22 GB** | block-wise weight streaming; prefetch hides **99.9 %** of cold-SSD load (0.274 GB/block, 44 ms load vs ~180 ms compute) | exp-051, 052 |
| Text encoder | 16.0 GB total stack | freed after encoding | lifecycle scheduling | exp-001/002 |
| OS vs MLX gap | +6.3 GB cache | ≤ 0.8 GB | MLX buffer-cache cap | exp-009 |
| **Process peak now set by** | | **text encoder (bf16 Qwen3, ~6.3 GB)** | next target: per-layer barriers / streaming | — |

## Speed: closed, with reasons (`reports/speed_track_closure.md`)

| Lever | Ceiling | Evidence |
|---|---|---|
| Roofline | already ~95 % of NAX bf16 peak (11.5 TFLOPS); attention 76 % | exp-030/031: NAX gives 3.7× over the non-NAX path |
| Exact fusion | ≤ 1.07× | non-matmul remainder is 12 % of a step |
| Block caching | ≤ 1.09× | only double blocks 0–3 are static (~13 % of compute) |
| Token merging | ~1.0× | 0.016 % near-duplicates after mean-centering (raw 53.9 % is anisotropy) |
| 8 / 4-bit weights | < 1.0× | 8–25 % slower; compute-bound, so dequant adds work |

## Quality boundary

Approximate (Tier A) shortcuts tried so far fail the LPIPS < 0.05 bar vs bf16:
- 8-bit weights: 0.027 / 0.044 / **0.081**
- 8-bit + mflux tiled VAE: 0.058 / 0.090 / 0.110, plus a visible tile seam

The exact path above is why the footprint result needs no quality caveat.

## Open items

1. Per-block compiled streaming (≤ 5 % over resident = done; timebox 1 day). Running now (exp-053).
2. Text encoder: measure its lazy-graph share, then per-layer barriers or streaming. It sets the process peak and the demo number.
3. Per-component 8-bit quality test (text encoder 8-bit on hold until then).
4. **A real 8 GB Mac test.** No 8 GB claim is made yet. `mx.set_memory_limit` is a soft limit, so that simulation is invalid (R2).

## Reusable outputs (model-agnostic)

Exact chunked VAE decode · block weight streaming with prefetch · allocator cache cap · eval barriers · per-layer quantization
sensitivity scan · LPIPS/grid quality gate · stage, layer and shape profilers · NAX A/B microbench. Protocol rules R1–R3. Hardware
profile with measured peak, bandwidth, SSD speed and a unified-memory streaming contention floor (+2–3.5 %).
