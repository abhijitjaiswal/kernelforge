# Finding: Klein-4B T2I on M5 has no low-hanging speed left

**Status: speed track closed (2026-10-20). Future speed work targets reference edits or bandwidth-bound workloads.**
A2/A3 were skipped by decision; nothing below is extrapolated from them.

## Closure chain (each link measured; raw data in runs/)

| # | Lever | Evidence | Ceiling |
|---|---|---|---|
| 1 | Roofline | Denoise matmuls run at 10–11.5 TFLOPS, ~95 % of the measured NAX bf16 peak (11.5); attention at 8.7 TFLOPS (76 %). Matmul + attention = 88 % of a step. The baseline already uses the M5 neural accelerators: forcing a pre-M5 arch gives 3.7× slower matmuls (exp-030, exp-031) | compute-bound, near peak |
| 2 | Exact fusion (Tier E) | The whole non-matmul remainder is 0.45 s of a 3.87 s step (12 %). The top 3 fusion candidates together save ≤ 0.3 s/step (phase1 report) | **≤ 1.07×** |
| 3 | Block-output caching (Tier A) | Only double blocks 0–3 are stable step to step (cos 0.97–0.998), about 13 % of compute. The single-stream blocks (≈ 80 % of compute) sit at 0.83–0.93; own-contribution cos > 0.98 in only 4.9 % of block-steps (exp-040) | **≤ 1.09×** |
| 4 | Token merging (Tier A) | Raw near-duplicate rate 53.9 % (cos > 0.95), but **0.016 % after subtracting the mean token**. The raw figure is anisotropy (a direction shared by all tokens), not redundancy (exp-041) | **~1.0×** |
| — | Lower-precision weights | q8 / q4 matmuls are 8–25 % *slower* than bf16 at these shapes (compute-bound; dequant adds work) (exp-015, exp-031) | < 1.0× |

Ship criterion was ≥ 1.3× end-to-end with LPIPS < 0.05 on all prompts. No lever, alone or combined (1.07 × 1.09 ≈ 1.17 even
if they composed perfectly), reaches it.

## Where speed work should go instead
- **Reference edits** (in-context reference tokens double or triple the sequence, so attention share and KV reuse change the picture).
- **Bandwidth-bound workloads** (batch-1 LLM decode, small-M matmuls). There q8/q4 and fusion candidate 1 (QK-norm + RoPE)
  are expected to pay; the deferred pattern is in `library/patterns/fused-qknorm-rope-transpose.yaml`.
