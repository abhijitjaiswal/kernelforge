# Determinism taxonomy: when "the same computation" gives different bits

**General rule:** bits vary across compiled vs eager execution AND across input-size-dependent algorithm selection. Both recur on
every framework (PyTorch/cuDNN choose algorithms by heuristic too). A bit-identity claim is valid only against a reference in
the same execution mode, computed through the same kernel paths.

A growing list of measured reasons why two runs of mathematically identical work differ. Each changes what a bit-identity
claim may be checked against.

| # | Cause | Measured | Consequence |
|---|---|---|---|
| a | **Execution mode** (eager vs compiled): the compiler fuses ops and changes rounding order | Klein p3: 30.4 dB between eager and compiled runs of the unmodified model (exp-040, exp-063) | Rule R1: compare only against a same-mode reference |
| b | **Swap** changes timing, never values | exp-063-rcomp: swap grew, image identical to its same-mode reference | Rule R3: swap invalidates timing only; quality from a swapping run stands |
| c | **Input-size-dependent algorithm selection**: the framework picks a kernel (e.g. Winograd vs implicit GEMM for 3×3 convolutions) by input size or working set, so the same op on a strip and on the full tensor can round differently | Wan 2.1 VAE: row-chunking the upsampler Conv2d gave 60.0 dB min / max diff 5 vs the full conv (exp-112). Row-chunking the 3D conv's per-frame conv2d stayed bit-identical | Chunking patterns are bit-exact only if every chunk takes the **same code path** as the full op. Choose chunk sizes on the same side of the selection threshold, or report Tier E. Applies to every framework: cuDNN/PyTorch pick algorithms by heuristic too |

**(c) confirmed in isolation (exp-114, real Wan upsampler shapes, random inputs):**
- Stock MLX (Winograd on by default, same as `MLX_CONV_WINOGRAD=1`): every re-padded strip height differs from the full conv
  (max abs 0.07–0.2). With `MLX_CONV_WINOGRAD=0`: every strip height is bit-identical.
- Mechanism: Winograd rounds per output tile. Re-padding each strip (slice with a halo, then let the conv zero-pad again) shifts
  the strip's tile grid by the halo, relative to the full conv's grid.
- **Fix without changing the reference configuration:** pad once as the full conv does, start every strip on the full conv's
  tile grid, run strips with zero row padding, and use strip heights that are **multiples of 24** rows. That is bit-identical at
  24/48/96/120/240 on all three convs; 12 works on two of three; 16 never does.
- General rule for chunking patterns: **align chunks to the kernel algorithm's tile grid and pad once.** Re-padding each chunk is
  exact in real arithmetic but not in floating point once the framework picks a tiled algorithm.
