# Step-caching viability is a property of the step count and distillation, not the architecture

**Rule (two cross-model datapoints):** probe it (A1, about a day) before building anything. Expect step or block caching to pay
above roughly 20 sampling steps, and to be dead on distilled few-step models.

| Model | Steps | Distilled | Block-steps with own-contribution cos > 0.98 | Source |
|---|---|---|---|---|
| FLUX.2 Klein 4B (image) | 4 | yes (few-step) | **4.9 %** | exp-040 |
| Wan 2.1 T2V 1.3B (video) | 50 | no | **77.3 %** (88.1 % after step 10; 90–100 % in steps 14–47) | exp-104 |

- Both are DiTs with residual streams; what differs is how far the sampler moves the latent per step.
- Measure the block's **own contribution** (output − input), not its output. The output cosine is inflated by the residual
  stream: Klein's output-cosine figure looked like a pass (28 %) while the contribution showed nothing to reuse.
- Logged as an experience-dataset predictor: features (steps, distilled, guidance) → measured cacheable fraction.
