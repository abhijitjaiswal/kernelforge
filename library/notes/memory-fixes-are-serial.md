# Memory fixes are serial: project against the sorted component list, never the fix's local saving

**Rule:** a memory fix's value is capped by the gap to the next-largest component. Before quoting a projected peak, sort the
peak's components and subtract the fix from its own component only. The new peak is the maximum of what remains.

| Observation | Fix | Local saving | Projected | Measured | Why |
|---|---|---|---|---|---|
| Klein Phase 0 (exp-005 → exp-006) | free the transformer before VAE decode | 3.8 GB of weights | lower decode peak | **0 GB** (9.49 → 9.49) | the VAE decoder's own 9.3 GB sat right behind |
| Wan decoder option 1 (exp-115 → exp-119/122/125) | build each 3D conv window directly | conv transient 3.34 → 2.05 GB | **~6 GB** | **8.19 GB footprint / 8.42 MLX** | spatial upsampler (+3.9 GB) next in line, then a 3.8 GB causal-state floor + 2.0 GB conv + 1.2 GB activations |

Applied to the Wan case, the sorted list would have predicted the result: causal state 3.8 + next conv 2.0 + activations 1.2 ≈ 7
GB plus weights/output, not ~6 GB.
