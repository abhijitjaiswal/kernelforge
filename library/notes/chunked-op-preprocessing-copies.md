# Once an op is chunked, the stock pre-processing copies around it dominate

Profile the concat, pad, transpose and cast that surround a chunked op, not just the op itself. Observed twice:

| Model | Op chunked | What dominated afterwards | Measured |
|---|---|---|---|
| FLUX.2 Klein VAE | 2D conv (row strips) | `nn.GroupNorm` makes a transposed fp32 copy of the full tensor | +2.0 GB at 1024² (exp-014) |
| Wan 2.1 VAE | per-frame conv2d inside CausalConv3d (row strips) | stock concatenates state + chunk (6 frames), zero-pads, transposes to channels-last: about 3 full copies | +3.3 GB per full-resolution 3D conv (exp-116) |

Fix pattern: build each kernel input (e.g. one output frame's window) directly from its sources. The inputs stay byte-identical,
so the result stays bit-identical.
