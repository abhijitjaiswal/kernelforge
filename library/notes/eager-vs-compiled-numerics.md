# Eager vs mx.compile numerics are not bit-identical

**Observed (exp-040):** the Klein transformer run eagerly vs through mflux's `mx.compile`'d predict closure gives final latents
with cosine 0.9996 / 0.9994 / 0.9967 (p1/p2/p3) and max abs difference up to 4.6. Compilation fuses elementwise ops and
changes rounding order, and 4 denoise steps amplify the difference.

**Consequences (model-independent):**
- An optimisation that runs eagerly (hooks, streaming, per-module profiling) cannot be judged against a compiled reference.
  The difference from compilation alone would be attributed to the optimisation.
- Bit-identity claims ("Tier E, identical") are only valid within one execution mode.
- The same applies to any graph compiler (torch.compile, XLA): compare like with like.

**Protocol rule:** see README "Protocol rules" R1.
