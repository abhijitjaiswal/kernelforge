# Phase 4b: video (Wan 2.1 1.3B, then Wan 2.2 5B), frozen engine

*Status: in progress. Started 2026-10-21.*

## Frozen state
- Engine and pattern library frozen at git tag `frozen-phase4b` (`engine/`, `library/`, `bench/`). Phase 4b adds only a new adapter
  (`adapters/wan/`). Any engine change after the frozen test is logged separately, as tuning.

## Predictions (logged before any measurement)
1. **Activations dominate the memory** at the fixed config, unlike Klein, where weights dominated. To be measured first.
2. **Streaming overhead < Klein's +2–3 %**: compute per block ≫ load per block (video sequences are long, weights per block small).
3. **Encoder row-gather is the largest single win.** The text encoder (umT5-XXL) is much larger than the 1.3B DiT, and its vocabulary is large.
4. **Stock VAE decode OOMs or swaps** at the fixed config.
5. (A1) At 30+ steps, step-to-step caching may be viable; same thresholds as before (< 20 % → drop).

## Quality gates
Same-mode bit-identity (R1); per-frame PSNR and LPIPS vs the baseline; a temporal-flicker check (frame-to-frame difference of
candidate vs baseline).

## First result: the frozen pattern library scored a hit before the baseline even ran

**Finding (upstreaming candidate for Blaizzy/mlx-video @ 87db56a):** stock mlx-video cannot run Wan 2.1 1.3B on a 16 GB Mac.
`load_t5_encoder` upcasts the whole umT5-XXL encoder to fp32 at load (21.16 GB of weights from an 11.36 GB bf16 file)
and evaluates every parameter at once. Measured (exp-101): MLX peak 23.12 GB, process footprint
23.90 GB, swap +5.6 GB net
(about +11.6 GB at peak), encode 141 s (swap-bound; memory-only under R3).
The first component touched is unrunnable stock, and an existing pattern fixes it. That is the generality thesis
showing up as a measurement.

**Fix = frozen patterns, adapter only** (exp-102): `block-weight-streaming-prefetch` over the 24 encoder blocks, plus
`row-gathered embeddings` for the 256,384-row table, using a new pattern variant, **store bf16, upcast per layer at load**:
fp32 numerics at roughly bf16-file memory. It is implemented as a `WeightStreamer` subclass in `adapters/wan/te_tools.py`,
so the engine stays frozen.

| umT5-XXL encoder | stock (exp-101) | streamed + upcast-on-load (exp-102) |
|---|---|---|
| MLX peak | 23.12 GB | **2.07 GB** |
| process peak footprint | 23.90 GB | **2.59 GB** |
| swap | +5.6 GB | none |
| encode time | 141.0 s (swapping) | 6.17 s (weight stall 0.08 s) |
| output vs frozen reference | (is the reference) | **bit-identical**: context True, context_null True |

The frozen embedding reference `baselines_video/wan21/v1_text_context.safetensors` (read-only) settles the encoder claim for every
downstream frame comparison by construction. Future encoder iterations verify against it in seconds.

Prediction 3 (encoder row-gather is the largest single win): **supported so far.** The encoder dominates the weights (21.2 GB fp32
vs a 2.84 GB bf16 DiT). The DiT and VAE numbers are still to come.

## Phase-0 baseline (exp-103): stock mlx-video downstream of the frozen stock embeddings

Config: Wan2.1 T2V 1.3B bf16, 832×480, 33 frames requested (36 decoded; mlx-video rounds to the 4-frame latent grid),
50 steps, CFG 5.0 (B=2 per step), seed 42, VAE tiling `auto` (stock default), compile on. Text context fed from
`baselines_video/wan21/v1_text_context.safetensors`, so the run is stock math throughout.

| Stage | Time | MLX peak | Weights | Activations (peak − weights) |
|---|---|---|---|---|
| DiT load | 0.6 s | 2.83 GB | 2.64 GB | – |
| Denoise (50 steps) | 1158 s | 4.69 GB | 2.64 GB | 1.86 GB |
| VAE decode | 82.7 s | **18.63 GB** | 0.27 GB | **18.36 GB** |

- Process peak footprint 18.07 GB on a 16 GB machine. Swap grew 2.5 GB (net),
  during the VAE decode only, so the decode timing is memory-only (R3).
- Per-step time 19.7 → 27.3 s (median 23.6 s) as the fanless chip heated: GPU busy median
  993 MHz (min 344), thermal Nominal → Moderate → Heavy.

**Predictions scored so far**
1. Activations dominate: **yes for the VAE decode** (18.4 GB vs 0.27 GB of weights); **no for
   denoise** (1.86 GB vs 2.64 GB).
2. Streaming overhead < Klein: not yet tested.
3. Encoder is the largest single win: holding (−19.1 GB). The VAE decode (18.6 GB) is the rival.
4. Stock VAE decode OOMs or swaps: **confirmed**, even with mlx-video's own `auto` tiling.

## Protocol additions (2026-10-21)
- **Frozen latents, one stage downstream of the frozen embeddings.** exp-103 did not save its latents, so exp-105 reruns the
  identical compiled config with decoder-input capture. Its frames must be bit-identical to exp-103 (determinism check). Its
  latents then become the frozen decoder reference: decoder variants iterate from them in about 80 s and bit-check against
  the stock frames.
- **Memory experiments at 4–8 steps.** Memory does not depend on the step count; 50-step runs are reserved for quality checks
  and finals.
- **A1 probes three redundancies:** block outputs across steps, token near-duplicates (raw and mean-centred), and the two CFG
  passes (cond vs uncond per block, plus the noise-prediction gap per step). At 100 transformer passes per video, speed has
  room to matter here.

## Exp A1 (exp-104): three redundancies at 50 steps, uncompiled, v1

| Redundancy | Wan 2.1 1.3B | Klein (4 steps) | ≥ 20 % threshold |
|---|---|---|---|
| Block output vs previous step, cos > 0.98 | **95.6 %** of 1470 block-steps | 28.4 % | pass |
| Block's own contribution vs previous step | **77.3 %**; 88.1 % after step 10 | 4.9 % | **pass** |
| Token near-duplicates, raw / mean-centred | 2.4 % / 0.49 % | 53.9 % / 0.016 % | **fail: drop token merging** |
| CFG cond vs uncond block outputs | **80.1 %** | (no CFG) | pass |
| CFG cond vs uncond own contributions | 39.9 % | – | pass |
| CFG noise predictions cos > 0.98 / median relative gap | 100 % of steps / 3.6 % | – | – |

- **Three phases over the steps:** steps 1–13 change a lot (7–50 % of blocks stable); **steps 14–47 are 90–100 % stable**;
  steps 48–50 change again (17 % at step 50).
- Every block is stable in the middle phase. The edge blocks (0–4, 26–29) are near 1.000; blocks 21–24 are the least stable (0.976–0.979).
- The CFG gap between cond and uncond predictions shrinks from 17 % (step 1) to 2.5–3 % (steps 40–45): reusing or skipping the
  uncond pass is most plausible late in the run.
- **Upper bound (not a measurement):** reusing blocks on about 2 of every 3 steps in the stable window skips about 23 of 50 steps,
  so denoise ≤ about 1.8×. Quality gates (LPIPS, flicker) decide. This is the first lever in the project with room above 1.3×.
- Prediction 5 (caching viable at 30+ steps): **supported by the probe**; implementation and the quality gate are pending (Tier A).
- The uncompiled run's final latents are saved (runs/exp-104-a1/latents.safetensors, shape [1, 16, 9, 60, 104]). R1: they are
  eager-mode latents, so they are not the frozen decoder reference (that is exp-105).

## Notes for the speed track (queued after the decoder)
- **Cross-model rule** (library/notes/step-caching-viability.md): step caching is viable at 50 steps (Wan, 77 %) and dead at
  4 distilled steps (Klein, 4.9 %). That depends on step count and distillation, not architecture. Token merging fails on both
  (library/notes/token-merging-anisotropy.md).
- **Do not stack the two caching wins.** Block-step caching and CFG skipping reuse the same redundancy: a step whose blocks barely
  changed is also one whose CFG gap is small. When implemented, they are measured as **one combined mechanism with one quality
  gate**, not two multipliers. Any clip-time projection stays out of reports until it has been run.
