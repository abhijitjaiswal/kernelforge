# Klein 4B on Apple M5 (16 GB): current state

*Draft, 2026-10-20. Measured on m5-test (MacBook Air M5, 8-core GPU, 16 GB, macOS 26.6); MLX 0.32.4, mflux 0.23.0.
1024², 4 steps, seed 42, prompts p1–p3. Every number traces to `runs/<exp>/result.json`.*

## Headline

> **Same model, same pixels, proven bit-identical: peak RAM 10.1 → 6.65 GB (next target ~3 GB), for ~7 % extra time.**
> Separately established: Klein's compute is already at ~95 % of this chip's ceiling, so we stopped chasing speed with
> evidence, not fatigue.

*Basis:* 6.65 GB OS peak in every streaming run (exp-051/052/053). Bit-identical to the same-mode resident reference in all
14 streaming runs (R1). The extra time is +7.5–7.8 % per denoise step vs the resident per-block-compiled pipeline (exp-053,
cold SSD, 2 rounds); lagged barriers (exp-054) are being measured to reduce it. The ~95 % is matmul throughput vs the measured
NAX bf16 peak (exp-031). The next target is set by the text encoder, now the largest stage (~6.3 GB bf16).

**Exact (bit-identical) memory reduction works: OS peak 10.1 GB → 6.65 GB, with no change to the image.** It costs about
10 % of denoise time today. Recovering that cost (per-block compile) is in progress. **There is no low-hanging speed left in
Klein T2I on this machine**: the matmuls already run at ~95 % of the M5 neural-accelerator peak.

## Where we started vs where we are

| | Baseline (Gate 0) | Now: exact pipeline |
|---|---|---|
| Config | bf16, text encoder freed, MLX cache cap 0.5 GB, compiled | + exact chunked VAE decode + transformer weight streaming (2 blocks resident, prefetch) |
| **End-to-end per image** | **18.92 s** (exp-011: cold single image, stock decode) | **22.71 / 22.84 s** (exp-053, steady state, cold SSD, exact chunked decode). The same pipeline resident: 21.41 / 21.77 s |
| Denoise (4 steps) | 15.74 s compiled (cold single image) | 18.35 / 18.47 s per-block-compiled streaming vs 17.15 / 17.26 s resident (**+7.5–7.8 %**) |
| **OS peak footprint** | **10.1 GB** (exp-011) | **6.65 GB** (exp-051/052) |
| Image vs reference | — | **bit-identical** to the same-mode resident run (10/10 runs; protocol rule R1) |

*Like-for-like:* of the 18.92 → 22.8 s gap, about 1.3 s is streaming (+7.5–7.8 % of denoise). The rest is measurement context:
steady-state throttled steps vs a cold single image, plus the exact chunked decode (3.5 s vs 2.1 s stock). The earlier "~19.8 s"
was eager denoise-only and is superseded.

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

1. Per-block compiled streaming: +7.5–7.8 % measured (exp-053). Lagged barriers (exp-054) are running; ≤ 5 % = done (1-day timebox).
2. Text encoder: measure its lazy-graph share, then per-layer barriers or streaming. It sets the process peak and the demo number.
3. Per-component 8-bit quality test (text encoder 8-bit on hold until then).
4. **A real 8 GB Mac test.** No 8 GB claim is made yet. `mx.set_memory_limit` is a soft limit, so that simulation is invalid (R2).

## Reusable outputs (model-agnostic)

Exact chunked VAE decode · block weight streaming with prefetch · allocator cache cap · eval barriers · per-layer quantization
sensitivity scan · LPIPS/grid quality gate · stage, layer and shape profilers · NAX A/B microbench. Protocol rules R1–R3. Hardware
profile with measured peak, bandwidth, SSD speed and a unified-memory streaming contention floor (+2–3.5 %).
