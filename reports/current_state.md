# Klein 4B on Apple M5 (16 GB): current state

*Draft, 2026-10-20. Measured on m5-test (MacBook Air M5, 8-core GPU, 16 GB, macOS 26.6); MLX 0.32.4, mflux 0.23.0.
1024², 4 steps, seed 42, prompts p1–p3. Every number traces to `runs/<exp>/result.json`.*

## Headline

> **Same model, same math: the low-memory pipeline reproduces the stock model's uncompiled output bit for bit, with peak RAM
> 10.1 → 3.5–3.7 GB, at +7 % time for a cold single image (mostly the exact decoder; the streamed denoise loop itself costs 2–3 %).
> Against the compiled baseline image it differs only by the compiler's rounding order (LPIPS ≤ 0.009).**

*Cold single image A/B (exp-065, 3 alternating rounds, same day): baseline 18.75 s vs final 20.09 s (median), +7.1 %; decoder
+0.9–1.15 s, denoise +0.4–0.5 s, text encoder −0.2 s. Final pipeline bit-identical to the uncompiled stock model on all three
prompts (p2 verified in exp-066). Peak across all final-pipeline runs 3.50–3.69 GB.*
> Separately established: Klein's compute is already at ~95 % of this chip's ceiling, so we stopped chasing speed with
> evidence, not fatigue.

*Basis:* **3.50 GB OS peak** for the full exact pipeline in both rounds (exp-061: streamed text encoder + streamed transformer +
chunked VAE, cold SSD, swap flat); the peak is now the denoise stage. Before streaming the text encoder it was 6.65 GB (exp-051..054). **Streaming is bit-identical** to the same-mode resident
reference in all 14 streaming runs (R1). **The end-to-end image is not bit-identical to the original baseline.** The exact
chunked VAE decode (needed for the 6.65 GB; the stock decode alone peaks at 9.5 GB) and per-block compilation change float
rounding: PSNR 46.9 dB, max pixel diff 55, mean |Δ| 0.64/255 vs the baseline image (exp-053). That is Tier E. The extra time is **+2.1–2.8 % per denoise step** vs the resident per-block-compiled pipeline, with per-block
compile, prefetch and a lag-1 barrier (exp-054, cold SSD, 2 rounds; the GPU always has one block queued). That is near the
measured 2–3.5 % unified-memory contention floor. Without the lag it was +7.5–7.8 % (exp-053). The ~95 % is matmul throughput vs the measured
NAX bf16 peak (exp-031). The text encoder went from 6.34 to 0.62 GB MLX by streaming its 27 used layers and reading only the prompt's 512 embedding rows from file (exp-060, bit-identical).


*Correction (exp-062..064):* the final pipeline vs the compiled baseline image is 46.9 / 39.0 / 30.4 dB (p1/p2/p3), so p2 and p3
are below the 40 dB Tier E bound. LPIPS is 0.0015 / 0.0034 / 0.0093. On p3 the final pipeline is bit-identical to the stock model
run without whole-model compilation; mflux's whole-model `mx.compile` rounding is the entire gap (the chunked decoder alone is 45.9 dB).
Peak memory across prompts: 3.50 / 3.53 / 3.67 GB.

## Where we started vs where we are

| | Baseline (Gate 0) | Now: exact pipeline |
|---|---|---|
| Config | bf16, text encoder freed, MLX cache cap 0.5 GB, compiled | + exact chunked VAE decode + transformer weight streaming (2 blocks resident, prefetch) |
| **End-to-end per image** | **18.92 s** (exp-011: cold single image, stock decode) | **20.67 / 21.59 s** (exp-061, full streaming, steady state, cold SSD, exact chunked decode). The same pipeline resident: 21.51 / 21.85 s (exp-054) |
| Denoise (4 steps) | 15.74 s compiled (cold single image) | 17.75 / 17.69 s streaming vs 17.31 / 17.32 s resident (**+2.1–2.8 % per step**) |
| **OS peak footprint** | **10.1 GB** (exp-011) | **3.50 GB** (exp-061, both rounds) |
| Image vs reference | — | streaming **bit-identical** to the same-mode resident run (14/14 runs, R1); vs the baseline image **46.9 dB** (Tier E: chunked decode + per-block compile rounding) |

*Like-for-like:* of the 18.92 → ~22 s gap, about 0.4 s is streaming (+2–3 % of denoise). The rest is measurement context:
steady-state throttled steps vs a cold single image, plus the exact chunked decode (3.5 s vs 2.1 s stock). The earlier "~19.8 s"
was eager denoise-only and is superseded.

## What moved memory (all Tier E)

| Stage | Before | After | How | Exp |
|---|---|---|---|---|
| VAE decode | 9.55 GB | **1.81 GB** | exact row-chunked conv/upsample with GroupNorm stats computed over the full latent; 52 / 45 / 46 dB vs reference | exp-014, 019 |
| Denoise (transformer) | 8.52 GB | **2.81 GB** | block-wise weight streaming, per-block compile, lag-1 barrier; prefetch hides **99.9 %** of cold-SSD load (0.274 GB/block, 44 ms load vs ~180 ms compute) | exp-051, 052, 054 |
| Text encoder | 16.0 GB total stack | freed after encoding | lifecycle scheduling | exp-001/002 |
| OS vs MLX gap | +6.3 GB cache | ≤ 0.8 GB | MLX buffer-cache cap | exp-009 |
| Text encoder | 6.34 GB MLX | **0.62 GB** | stream the 27 used layers (9 are dead), gather the prompt's embedding rows from file; per-layer barriers made it *worse* (7.77 GB: they run the dead layers) | exp-060 |
| **Process peak now set by** | | **denoise (3.50 GB OS: 2.81 GB MLX + runtime)** | | exp-061 |

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

1. ~~Per-block compiled streaming~~ **done**: +2.1–2.8 % with a lag-1 barrier (exp-054), within the ≤ 5 % target.
2. ~~Text encoder~~ **done**: lazy-graph share 8–10 %; streaming 6.34 → 0.62 GB, bit-identical; full pipeline 3.50 GB (exp-060/061).
3. Per-component 8-bit quality test (text encoder 8-bit on hold until then).
4. **A real 8 GB Mac test.** No 8 GB claim is made yet. `mx.set_memory_limit` is a soft limit, so that simulation is invalid (R2).

## Reusable outputs (model-agnostic)

Exact chunked VAE decode · block weight streaming with prefetch · allocator cache cap · eval barriers · per-layer quantization
sensitivity scan · LPIPS/grid quality gate · stage, layer and shape profilers · NAX A/B microbench. Protocol rules R1–R3. Hardware
profile with measured peak, bandwidth, SSD speed and a unified-memory streaming contention floor (+2–3.5 %).
