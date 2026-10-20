# Phase 1 report: Profile (FLUX.2 Klein 4B, m5-test M5 16 GB)

**Headline.** The denoise step is **compute-bound on the M5 neural accelerators, and already efficient.** The large matmuls run at
10–11.5 TFLOPS, about 95 % of the measured bf16 peak, and they plus attention are **88 % of a step**. Everything that exact
fusion could remove totals at most **~0.45 s per step (12 %)**. Even eliminating all of it would give ≤ 1.10× end-to-end.
**Verdict on the speed goal (≥ 1.3×, Tier E): not reachable by exact kernel fusion on this machine.** It needs fewer FLOPs
(Tier A methods) or a different target; see the recommendation.
Memory: **an exact VAE decode now fits in 1.8 GB** (was 9.55). The whole-pipeline peak is the bf16 transformer during denoise
(8.57 GB MLX, 9.2–9.4 GB OS), about 1.4 GB above the 8 GB goal.

Everything below is measured unless marked *estimate*. Raw data: `runs/exp-0NN/result.json`.

## Corrections from Gate 0, applied

| Correction | Done | Result |
|---|---|---|
| Separate load from compute; warm per-stage; per-step | `bench/stages.py` (exp-015, 3 alternating rounds × 5 warm runs) | bf16: load 2.68 s; compute: text 0.334, denoise **18.20** (4.55 s/step), VAE 2.16 s |
| Re-derive bf16 vs q8 on compute only | exp-015 | q8 denoise **20.42 s** (5.10 s/step): **q8 is 12 % slower** on compute. Its only gain is load (1.28 vs 2.68 s) |
| Footprint result is "candidate"; LPIPS + grids | exp-010 ×4, exp-013, `bench/quality.py` | 6.04–6.05 GB reproduced 4×, but **fails Tier A**: LPIPS vs bf16 0.058 / 0.090 / 0.110, plus a visible tile seam. 8-bit alone also fails on p3 (0.081) |
| Tiled VAE with global GroupNorm stats, Tier E | `engine/chunked.py`, exp-019/020 | **1.81 GB MLX, Tier E on p1/p2/p3** (52.3 / 45.2 / 46.1 dB) |
| VAE decoder memory first | exp-014 | done; see the memory section |
| tmux + powermetrics | in use | tmux runs every job. powermetrics records GPU MHz, power and thermal level (macOS 26 has no `smc` sampler, so no temperature) |

**Steady state vs cold.** Warm back-to-back denoise throttles the GPU to ~1150 MHz (vs ~1360 MHz cold) while macOS still reports
thermal pressure "Nominal". The steady-state step is 4.55 s vs 3.87 s cold. The pressure level is not a sufficient cool-down
gate, so cool-downs now also need an idle period.

## Ranked op table: one denoise step (bf16, 4096 image + 512 text tokens)

`xctrace` needs full Xcode, and m5-test has only Command Line Tools (installing Xcode is a system-level install, so not done). Per
the brief's fallback, ops are timed with MLX: matmul and SDPA times come from microbenchmarks at the exact shapes (exp-031),
and the remainder is the compiled step minus those. Compiled step = **3.87 s** (cold, exp-030).

| # | Op (source) | Calls/step | ms each | s/step | % of step | Achieved | Bound |
|---|---|---|---|---|---|---|---|
| 1 | single-block fused QKV+MLP-in matmul `to_qkv_mlp_proj` (4608×3072→27648) | 20 | 72.5 | 1.45 | 37.5 % | 10.8 TF | compute |
| 2 | joint attention SDPA (24 heads, L 4608, d 128) | 25 | 30.0 | 0.75 | 19.4 % | 8.7 TF | compute |
| 3 | single-block out matmul `to_out` (4608×12288→3072) | 20 | 34.8 | 0.70 | 18.0 % | 10.0 TF | compute |
| 4 | **non-matmul remainder** (RoPE, q/k RMSNorm, LayerNorm+adaLN modulate, SwiGLU, concat, transposes, residual gates) | — | — | **0.45** | **11.6 %** | — | memory |
| 5 | double-block FF in (4096×3072→18432) | 5 | 41.8 | 0.21 | 5.4 % | 11.1 TF | compute |
| 6 | double-block img q/k/v/out (4096×3072→3072) | 20 | 6.7 | 0.13 | 3.5 % | 11.5 TF | compute |
| 7 | double-block FF out (4096×9216→3072) | 5 | 21.7 | 0.11 | 2.8 % | 10.7 TF | compute |
| 8 | double-block text FF in/out | 10 | 5.5 / 3.1 | 0.04 | 1.1 % | 9.5–10.5 TF | compute |
| 9 | double-block text q/k/v/out | 20 | 1.1 | 0.02 | 0.6 % | 9.2 TF | compute |
| 10 | context / x embedders, modulation, proj_out | 8 | 0.2–2.2 | 0.01 | 0.3 % | — | launch / memory |

Within the remainder, eager per-module timing (exp-030, without compile) attributes most of it to RoPE + head reshapes
(~0.54 s eager across 25 blocks), SwiGLU (0.10 s), q/k RMSNorm (0.07 s) and LayerNorm (0.02 s). Compile already fuses part of this.

**Neural accelerators (Metal 4 / NAX).** MLX 0.32.4 ships NAX kernels (`steel_gemm_*_nax`, `affine_qmm_*_nax`). Determined by
A/B: forcing a pre-M5 architecture (`MLX_METAL_GPU_ARCH=applegpu_g16g`, exp-031) drops bf16 matmul from 11.5 to 3.0 TFLOPS and
SDPA from 8.7 to 2.5. The Klein matmuls take 9.96 s/step instead of 2.67. **The baseline already uses the neural
accelerators (~3.7×)** for matmuls and attention.

**Hardware facts measured** (to `library/hardware/apple_m5_8gpu.yaml`): bf16 matmul peak 11.5 TFLOPS (NAX), fp16 11.5, fp32
9.1 (TF32 on NAX); without NAX ~3.0. Elementwise bandwidth 135 GB/s (read+write). Launch overhead ~0.2 ms per eval. Ridge
point ≈ 85 FLOP/B.

## Fusion candidates (estimates of the upper bound, all Tier E)

| Candidate | What it removes | Time saved (*estimate*) | Memory saved |
|---|---|---|---|
| 1. **QK-RMSNorm + RoPE + head transpose** fused into one kernel after the QKV matmul (25 blocks) | fp32 q/k copies, 2 transposes, RoPE temporaries | ≤ 0.2 s/step (≤ 0.8 s/image, ≤ 4 %) | 2 × 53 MB fp32 transients per block |
| 2. **SwiGLU + concat-free `to_out`**: `to_out([attn, mlp]) = W_a·attn + W_m·swiglu(mlp)` as two accumulating matmuls (20 blocks) | the 4608×12288 concat copy and the SwiGLU output | ≤ 0.05 s/step | 113 MB + 79 MB per block |
| 3. **LayerNorm + adaLN modulate (+ gated residual)** (45 norm sites) | 2–3 elementwise passes over 4608×3072 | ≤ 0.05 s/step | ~26 MB per site |

Even all three at their upper bounds give ~0.3 s/step = 1.2 s/image → **~1.07×**. The kill criterion (< 15 %) will almost
certainly trigger for exact fusion.

## Memory profile

**Allocation timeline (exact pipeline, exp-021-chunk, MLX active):** text encoder loads lazily (6.0–6.3 GB peak), is freed →
~0 GB. Transformer weights 7.2 GB plus activations ≤ 1.35 GB → **8.57 GB peak during denoise**. The transformer is freed, then
chunked VAE decode at 1.8 GB. With the cache cap, the OS footprint is 9.2–9.4 GB.

**VAE decoder (exp-014, per module):** fused stock decode 9.55 GB, but no single module needs more than 4.33 GB with bit-identical
output, so the excess was lazy-graph lifetime. Hot ops were the 1024² upsample conv (+3.05 GB) and full-resolution GroupNorm
(+2.0 GB, a transposed fp32 copy inside `nn.GroupNorm`). Mid-block attention is only +0.09 GB, so it was not chunked (measured
rather than assumed). Fixes: eval barriers (bit-identical, 7.26 GB) → exact row-chunking (1.81 GB, Tier E).

**Top 10 allocations during denoise** (exp-030):
1. transformer weights: 20 single blocks × 245 MB
2. transformer weights: 5 double blocks × ~340 MB
3. `to_qkv_mlp_proj` output 237 MB
4. double FF-in output 141 MB
5. concat input to `to_out` ~113 MB *(avoidable: candidate 2)*
6. SwiGLU output 79 MB *(avoidable)*
7. double SwiGLU 70 MB
8. q/k RMSNorm fp32 outputs 2 × 53 MB *(avoidable: candidate 1)*
9. double-block q/k fp32 2 × 47 MB *(avoidable)*
10. LayerNorm outputs 26 MB

Activations are small. **Weights are the footprint.**

## Per-layer quantization sensitivity (exp-032, `engine/sensitivity.py`, model-agnostic)

Metric: rel-L2 of the transformer output (step 1, p1) with one Linear quantized (group 64), vs bf16. The floor is verified
zero: repeated forwards and an fp32-recomputed layer are bit-identical.

| | min | median | max |
|---|---|---|---|
| 8-bit, one layer | 1.9e-3 | 1.0e-2 | 2.3e-2 |
| 4-bit, one layer | 5.6e-3 | 2.4e-2 | 5.7e-2 |

- **Most sensitive** (keep bf16): `time_guidance_embed`, `context_embedder`, `x_embedder`, `proj_out`, the three modulation
  linears (113 M params together), block 0's v-projection, and single block 0's QKV-MLP projection.
- **Least sensitive**: double-block `attn.to_q/to_k`, double-block `ff.linear_in/out` (≈1.5e-2 at 4-bit), `norm_out.linear`,
  late single-block `to_out` (single block 19: 7.2e-3).
- The single-block projections (the bulk of the 3.88 B linear params) sit in the middle (median 2.6–3.0e-2 at 4-bit).
- One-layer errors do not add linearly. A recipe must be validated end-to-end with LPIPS (8-bit everywhere already fails p3).

## What didn't work / surprises

- **8-bit for speed**: slower on compute (compute-bound, dequant adds work), and Tier A only.
- **Tiled VAE (mflux)**: per-tile GroupNorm, visible seam, fails Tier A. Replaced by the global-stats chunked decoder.
- **`mx.set_memory_limit` as an 8 GB simulation**: it is a soft limit. Runs exceed it (8 GB cap → 8.54 GB OS; 6 GB cap →
  8.06 GB OS, denoise +20 %). It cannot show behaviour on a real 8 GB Mac.
- **xctrace**: unavailable without Xcode, so there are no per-kernel GPU timestamps.
- The thermal pressure level stays "Nominal" while the GPU clock drops ~15 %.
- Tooling bugs fixed along the way: `quality.py` import path, tmux quoting (job scripts are now files), record.py for non-image runs.

## Generality

Model-independent tools and knowledge added: `bench/stages.py` (load/compute split), `bench/layer_profile.py` (per-module
memory/time + shape capture), `bench/microbench.py` (shape-driven matmul/SDPA/bandwidth, NAX A/B), `engine/chunked.py`
(exact chunked conv / upsample-conv / global-stat GroupNorm, property-tested), `engine/sensitivity.py`, `bench/quality.py`
(LPIPS + grids), `bench/power.py`.

Patterns: `vae-exact-row-chunked-decode` (Tier E; applies to every conv VAE: SD, SDXL, Flux, video VAEs),
`eval-barriers-between-blocks` (Tier E, any MLX lazy graph), `allocator-cache-cap`, and the 8-bit-is-slower-when-compute-bound
lesson (applies to any NAX-era matmul-bound model; it does **not** carry to batch-1 LLM decode, which is bandwidth-bound).

## Recommendation (single candidates)

**Speed.** Exact fusion is capped at ~1.07–1.10×, so the 1.3× Tier E goal is not achievable by the Phase 2 plan as written.
I recommend still doing **fusion candidate 1 (QK-RMSNorm + RoPE + transpose)** as the one hand-written kernel. It is the largest
non-matmul cost, it is a pattern in every modern DiT and LLM (so it feeds the library), and it gives an honest measurement
for the kill criterion. Then decide between:
- (a) accepting the speed result and putting the 1.3× budget into Tier A methods (step/feature caching), or
- (b) moving speed work to another target.

**Footprint.** **Block-wise transformer weight streaming** (technique 6) is the only exact path under 8 GB: keep ~2 blocks
resident instead of all 25, which saves up to ~6.5 GB. Its cost depends on SSD/page-cache read speed: 7.2 GB per step must be
re-read, *estimate* 1–1.5 s/step at 5–7 GB/s, unless it overlaps with compute (matmuls are compute-bound, so overlap is
plausible). The fallback is a Tier A **mixed-precision recipe** from the sensitivity ranking: 8-bit for the least-sensitive
~60 % of params saves ~2.3 GB, validated with LPIPS on p1–p3.

**Stopping at Gate 1 for your decision.**
