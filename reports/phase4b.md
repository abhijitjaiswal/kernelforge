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
