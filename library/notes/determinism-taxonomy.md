# Determinism taxonomy: when "the same computation" gives different bits

A growing list of measured reasons why two runs of mathematically identical work differ. Each changes what a bit-identity
claim may be checked against.

| # | Cause | Measured | Consequence |
|---|---|---|---|
| a | **Execution mode** (eager vs compiled): the compiler fuses ops and changes rounding order | Klein p3: 30.4 dB between eager and compiled runs of the unmodified model (exp-040, exp-063) | Rule R1: compare only against a same-mode reference |
| b | **Swap** changes timing, never values | exp-063-rcomp: swap grew, image identical to its same-mode reference | Rule R3: swap invalidates timing only; quality from a swapping run stands |
| c | **Input-size-dependent algorithm selection**: the framework picks a kernel (e.g. Winograd vs implicit GEMM for 3×3 convolutions) by input size or working set, so the same op on a strip and on the full tensor can round differently | Wan 2.1 VAE: row-chunking the upsampler Conv2d gave 60.0 dB min / max diff 5 vs the full conv (exp-112). Row-chunking the 3D conv's per-frame conv2d stayed bit-identical | Chunking patterns are bit-exact only if every chunk takes the **same code path** as the full op. Choose chunk sizes on the same side of the selection threshold, or report Tier E. Applies to every framework: cuDNN/PyTorch pick algorithms by heuristic too |

Status of (c): hypothesis being confirmed in isolation (one conv at the real upsampler shape, full vs strip, toggling the
`MLX_CONV_WINOGRAD*` settings). The fix may not change the reference configuration: the reference is stock as it ships.
