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
