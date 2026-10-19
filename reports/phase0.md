# Phase 0 report: Baseline (FLUX.2 Klein 4B on m5-test, Apple M5 16 GB)

**Headline.** The new baseline to beat is **bf16, staged lifecycle, MLX cache cap 0.5 GB: 18.92 s per 1024² image**
(median of 3 alternating cold runs, range 18.86–19.20 s). Its peak footprint is **10.1 GB**, swap does not grow, and the
output is pixel-identical to the bf16 reference.
- Speed goal (≥ 1.3× over this baseline, i.e. ≤ 14.55 s): **not started**; that is Phase 2 work.
- Footprint goal (≤ 8 GB, Tier A): **met on measured peak** by exp-010 at **6.04 GB** / 20.7 s, pending LPIPS
  and your visual check. Exact (Tier E) footprint ≤ 8 GB: **not met**. The untiled VAE decoder alone needs about 9.3 GB.

All numbers come from `runs/<exp>/result.json` (raw per-image rows in `per_run.jsonl`, memory timelines in `memtrace.json`).
"Measured" means read from those files; anything else is labelled as a hypothesis.

## Stack

| | |
|---|---|
| Machine | m5-test: MacBook Air Mac17,3, Apple M5 (4P+6E CPU, 8-core GPU, Metal 4), 16 GB, macOS 26.6, on AC, Low Power Mode off |
| Software | Python 3.12.15, **mflux 0.23.0** (latest), **MLX 0.32.4**, venv `~/kf-venv` |
| Model | black-forest-labs/FLUX.2-klein-4B @ e7b7dc2, **Apache-2.0**; bf16 snapshot and the user's saved 8-bit copy, both read-only from `~/imagegen` |
| Path taken | mflux supports Klein 4B natively; no fallback was needed |
| Settings | 1024×1024, 4 steps, guidance 1.0 (so no CFG: one transformer pass per step), seed 42, prompts p1/p2/p3 (`bench/prompts.json`) |

## Baseline numbers per stage

Alternating A/B, cold single image p1 per process, staged lifecycle, cache cap 0.5 GB, 150 s cool-down between runs
(exp-011, 3 rounds each; every run swap-flat, so the timings are valid):

| | Text encode | Denoise (4 steps) | VAE decode | **Total** | Peak footprint | Quality vs own-precision ref |
|---|---|---|---|---|---|---|
| **bf16** (new baseline) | 1.04 s (5.5 %) | 15.74 s (83.2 %) | 2.10 s (11.1 %) | **18.92 s** | 9.74–10.25 GB | identical |
| q8 (8-bit weights) | 0.49 s | 17.42 s | 2.09 s | 20.02 s | 9.80 GB | identical |

**bf16 is 1.06× faster than 8-bit**: 8-bit weights save memory but cost about 1.7 s of denoise. The hypothesis is that
dequantization inside the quantized matmul costs more than the halved weight traffic saves. This is a fused
dequant-matmul target (technique 7) and suggests denoise is not bound by weight bandwidth (to confirm in Phase 1).

**Quality across precision.** 8-bit output vs bf16 reference: PSNR 29.4 / 27.5 / 20.7 dB (p1/p2/p3), max pixel
difference 241 (exp-003). The stock 8-bit configuration is therefore **Tier A** relative to bf16, and LPIPS is still to
be measured. p3 (text sign) is the most sensitive prompt.

## Memory breakdown by stage (bf16 baseline, exp-009, from the memory timeline)

| Stage | MLX active peak | Process footprint peak |
|---|---|---|
| Weights resident | text encoder 7.49 GB, transformer 7.22 GB, VAE 0.16 GB (bytes on disk) | |
| Text encoding | 6.15 GB | 6.65 GB |
| Denoise | 8.57 GB (weights 7.22 + about 1.35 activations) | 9.30 GB |
| **VAE decode** | **9.50 GB** (transformer already freed, so about 9.3 GB is decoder activations) | **10.22 GB** |

Measured facts that matter for Phases 1–2:
1. **The VAE decoder is the memory ceiling.** It needs about 9.3 GB of activations at 1024², more than the 7.2 GB bf16
   transformer. Freeing the transformer before decode (exp-006) did not move the peak (9.49 → 9.49 GB).
2. **The OS-vs-MLX gap is MLX's freed-buffer cache.** At the exp-005 peak, footprint 11.87 GB = active 4.95 + cache
   6.26 GB. Capping the cache at 0.5 GB (exp-009) cuts the footprint 11.54 → 10.26 GB, pixel-identical, at no speed cost.
3. **Lazy weight loading already skips unused text-encoder layers.** The bf16 text encoder peaks at 6.34 GB against 7.49 GB
   of weights. Hypothesis: the layers after 27 (hidden states are taken from 9/18/27) are never loaded. To verify in Phase 1.
4. Denoise activations are small: about 1.35 GB in both precisions.

## Lifecycle scheduling result (technique 1)

| Exp | Lifecycle | MLX peak | Footprint | Swap growth | Total | Pixels |
|---|---|---|---|---|---|---|
| exp-002 | stock (everything resident) | 15.99 GB | 14.12 GB | +1.8 GB | 26.6 s* | (ref) |
| exp-001 | free text encoder after encode | 9.51 GB | 12.65 GB | +0.7 GB | 25.4 s* | identical to exp-002 |
| exp-006 | + free transformer before decode | 9.49 GB | 11.70 GB | +67 MB | 20.9 s | identical |
| exp-009 / 011 | + cache cap 0.5 GB (bf16) | 9.50 GB | 9.74–10.26 GB | none | 18.92 s | identical |

\* exp-001/002 ran warm in a long batch with swap growth, so their timings are invalid under the protocol. They are kept
for the memory comparison only.

Verdict: lifecycle scheduling plus a cache cap is **Tier E** (bit-identical) and takes the stock pipeline from
16.0 GB MLX / 14.1 GB footprint with swapping to 10.1 GB without swapping. That change is also what made valid timing possible.

## Lowest footprint so far (Tier A)

exp-010: 8-bit + staged + tiled VAE decode + cache cap → **6.04 GB footprint, 5.19 GB MLX, 20.7 s, swap flat**. The peak
has moved to denoise. Tiling is not exact: PSNR 26.5 dB, max diff 55 vs the untiled decode (exp-004/008). LPIPS and the
side-by-side grid are still owed before it can be called Tier A. The 8 GB goal therefore looks reachable. The open question is quality.

## Thermal behaviour (no powermetrics: sudo needs a password over SSH)

- In a warm 5-round batch the median total drifts **23.5 → 27.5 s** (exp-001) and **25.5 → 31.9 s** (exp-002), so run 5
  is 17–25 % slower than run 1.
- After about 40 minutes of continuous load, denoise rose from 22 s to 32 s (exp-004), with the same code.
- With 150 s cool-downs and one image per process, runs are stable: bf16 18.86–19.20 s, q8 20.00–20.04 s.
- Consequence: every comparison from here on is **alternating A/B with cool-downs**, and the 5-run warm batches are kept
  only for memory and determinism. A 5 % effect would be invisible under the warm-batch drift.

## What didn't work / surprises

- **8-bit weights for speed**: 6 % slower than bf16 and Tier A. They remain useful only as a memory lever.
- **Freeing the transformer before decode**: no peak reduction on its own, because the decoder dominates. It only helps together with tiling.
- **Warm multi-run batches** on this fanless machine: thermal drift and swap growth invalidated their timings.
- **Port 8765** was taken locally by another program, so the dashboard uses 8790.
- `powermetrics` is unavailable without a sudo password, and tmux is not installed (`nohup` is used instead).

## Generality

Model-independent knowledge from this phase (recorded as patterns in `library/patterns/`):
- `lifecycle-free-after-last-use`: free each pipeline stage's weights after its last use. Tier E. Applies to any
  multi-model pipeline (diffusion text encoder → DiT → VAE, LLM draft and target models, video pipelines).
- `allocator-cache-cap`: bound the framework's freed-buffer cache. Tier E. Applies to any MLX workload, and the
  same idea applies to PyTorch's caching allocator.
- `vae-tiled-decode`: Tier A for this decoder. Applies to every latent-diffusion image or video model.
- Measurement method: lazy-load plus stage-attributed footprint tracing (`bench/memtrace.py`) is model-agnostic.
- Hardware lesson for `library/hardware/apple_m5_8gpu.yaml`: 8-bit quantized matmul is slower than bf16 matmul at
  these shapes on M5. This needs a microbenchmark in Phase 1 before it becomes a hardware fact.

Still Klein-specific: the adapter hooks (`_encode_prompt_pair`, `vae.decode_packed_latents`) and the 9.3 GB decoder activation figure.

## Proposed Phase 1 focus (for your decision)

1. Profile denoise (83 % of time) with `xctrace` / MLX tooling. Rank kernels, check whether the bf16 matmuls use the
   M5 neural accelerators, and microbenchmark q8 vs bf16 matmul at the real shapes.
2. Profile the VAE decoder's 9.3 GB. If it comes from the mid-block attention or full-resolution fp32 intermediates,
   a memory-efficient attention or chunked decode could make an **exact** (Tier E) decode fit under 8 GB.
3. Measure LPIPS for the 8-bit and tiled-decode results.
