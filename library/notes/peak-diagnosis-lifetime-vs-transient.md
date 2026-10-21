# Diagnose the memory peak before choosing a fix: graph lifetime vs single-op transient

**Rule:** run the per-module profiler (`bench/layer_profile.py`) once. If the end-to-end peak is far above the largest
single-module peak, the excess is **graph lifetime** (intermediates kept alive across the lazy graph), and eval barriers fix it.
If one module's own transient (its delta) accounts for the peak, it is a **single-op transient**, and only chunking that op's
input fixes it. Barriers do nothing there.

| Model / decoder | End-to-end peak | Largest per-module peak | Diagnosis | Barriers |
|---|---|---|---|---|
| FLUX.2 Klein VAE (2D) | 9.55 GB | 4.33 GB | graph lifetime | 9.55 → 7.26 GB, bit-identical (exp-014, exp-018) |
| Wan 2.1 VAE (3D causal), 9 frames | 16.77 GB | 16.75 GB (one `CausalConv3d`: +13.0 GB on a 1.71 GB input, about 7.6×) | single-op transient | 16.77 → 16.77 GB, bit-identical (exp-107) |

The measurement costs one profiler run and answers the question on any future model before a line of fix code is written.
