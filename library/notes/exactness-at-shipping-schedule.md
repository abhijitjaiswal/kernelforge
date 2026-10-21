# Exactness is verified at the configuration that ships, including the RNG path

1. **Bit-identity at one schedule is not bit-identity at another.** An exactness claim must be verified at the step count it ships
   with. Short schedules (e.g. 8 steps) remain valid for development A/B; they are not what gets claimed. (Wan DiT, 2026-10-21:
   per-block compile = stock at 8 steps was marked provisional until 50-step runs confirmed it.)
2. **A reference must be produced by the same RNG-consuming path as the system under test.** Model constructors draw random
   initial weights from the global generator, so skipping or adding a model construction between `seed()` and the noise draw
   changes the noise. Observed: a frozen baseline made with a stub text-encoder loader (no constructor) vs a pipeline that
   constructs the encoder: identical embeddings, identical DiT, cosine 0.06 final latents (completely different video).
   Check: replay the RNG order (seed -> every constructor -> noise) and compare the saved initial noise before comparing outputs.
3. **Read the divergence magnitude first.** Cosine ~0.999x points to a numeric-path difference snowballing; cosine near 0 points
   to a structural input difference (noise, conditioning, schedule). Different hunts.
