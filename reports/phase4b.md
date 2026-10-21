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

## First result: the frozen pattern library scored a hit before the baseline even ran

**Finding (upstreaming candidate for Blaizzy/mlx-video @ 87db56a):** stock mlx-video cannot run Wan 2.1 1.3B on a 16 GB Mac.
`load_t5_encoder` upcasts the whole umT5-XXL encoder to fp32 at load (21.16 GB of weights from an 11.36 GB bf16 file)
and evaluates every parameter at once. Measured (exp-101): MLX peak 23.12 GB, process footprint
23.90 GB, swap +5.6 GB net
(about +11.6 GB at peak), encode 141 s (swap-bound; memory-only under R3).
The first component touched is unrunnable stock, and an existing pattern fixes it. That is the generality thesis
showing up as a measurement.

**Fix = frozen patterns, adapter only** (exp-102): `block-weight-streaming-prefetch` over the 24 encoder blocks, plus
`row-gathered embeddings` for the 256,384-row table, using a new pattern variant, **store bf16, upcast per layer at load**:
fp32 numerics at roughly bf16-file memory. It is implemented as a `WeightStreamer` subclass in `adapters/wan/te_tools.py`,
so the engine stays frozen.

| umT5-XXL encoder | stock (exp-101) | streamed + upcast-on-load (exp-102) |
|---|---|---|
| MLX peak | 23.12 GB | **2.07 GB** |
| process peak footprint | 23.90 GB | **2.59 GB** |
| swap | +5.6 GB | none |
| encode time | 141.0 s (swapping) | 6.17 s (weight stall 0.08 s) |
| output vs frozen reference | (is the reference) | **bit-identical**: context True, context_null True |

The frozen embedding reference `baselines_video/wan21/v1_text_context.safetensors` (read-only) settles the encoder claim for every
downstream frame comparison by construction. Future encoder iterations verify against it in seconds.

Prediction 3 (encoder row-gather is the largest single win): **supported so far.** The encoder dominates the weights (21.2 GB fp32
vs a 2.84 GB bf16 DiT). The DiT and VAE numbers are still to come.

## Phase-0 baseline (exp-103): stock mlx-video downstream of the frozen stock embeddings

Config: Wan2.1 T2V 1.3B bf16, 832×480, 33 frames requested (36 decoded; mlx-video rounds to the 4-frame latent grid),
50 steps, CFG 5.0 (B=2 per step), seed 42, VAE tiling `auto` (stock default), compile on. Text context fed from
`baselines_video/wan21/v1_text_context.safetensors`, so the run is stock math throughout.

| Stage | Time | MLX peak | Weights | Activations (peak − weights) |
|---|---|---|---|---|
| DiT load | 0.6 s | 2.83 GB | 2.64 GB | – |
| Denoise (50 steps) | 1158 s | 4.69 GB | 2.64 GB | 1.86 GB |
| VAE decode | 82.7 s | **18.63 GB** | 0.27 GB | **18.36 GB** |

- Process peak footprint 18.07 GB on a 16 GB machine. Swap grew 2.5 GB (net),
  during the VAE decode only, so the decode timing is memory-only (R3).
- Per-step time 19.7 → 27.3 s (median 23.6 s) as the fanless chip heated: GPU busy median
  993 MHz (min 344), thermal Nominal → Moderate → Heavy.

**Predictions scored so far**
1. Activations dominate: **yes for the VAE decode** (18.4 GB vs 0.27 GB of weights); **no for
   denoise** (1.86 GB vs 2.64 GB).
2. Streaming overhead < Klein: not yet tested.
3. Encoder is the largest single win: holding (−19.1 GB). The VAE decode (18.6 GB) is the rival.
4. Stock VAE decode OOMs or swaps: **confirmed**, even with mlx-video's own `auto` tiling.

## Protocol additions (2026-10-21)
- **Frozen latents, one stage downstream of the frozen embeddings.** exp-103 did not save its latents, so exp-105 reruns the
  identical compiled config with decoder-input capture. Its frames must be bit-identical to exp-103 (determinism check). Its
  latents then become the frozen decoder reference: decoder variants iterate from them in about 80 s and bit-check against
  the stock frames.
- **Memory experiments at 4–8 steps.** Memory does not depend on the step count; 50-step runs are reserved for quality checks
  and finals.
- **A1 probes three redundancies:** block outputs across steps, token near-duplicates (raw and mean-centred), and the two CFG
  passes (cond vs uncond per block, plus the noise-prediction gap per step). At 100 transformer passes per video, speed has
  room to matter here.

## Exp A1 (exp-104): three redundancies at 50 steps, uncompiled, v1

| Redundancy | Wan 2.1 1.3B | Klein (4 steps) | ≥ 20 % threshold |
|---|---|---|---|
| Block output vs previous step, cos > 0.98 | **95.6 %** of 1470 block-steps | 28.4 % | pass |
| Block's own contribution vs previous step | **77.3 %**; 88.1 % after step 10 | 4.9 % | **pass** |
| Token near-duplicates, raw / mean-centred | 2.4 % / 0.49 % | 53.9 % / 0.016 % | **fail: drop token merging** |
| CFG cond vs uncond block outputs | **80.1 %** | (no CFG) | pass |
| CFG cond vs uncond own contributions | 39.9 % | – | pass |
| CFG noise predictions cos > 0.98 / median relative gap | 100 % of steps / 3.6 % | – | – |

- **Three phases over the steps:** steps 1–13 change a lot (7–50 % of blocks stable); **steps 14–47 are 90–100 % stable**;
  steps 48–50 change again (17 % at step 50).
- Every block is stable in the middle phase. The edge blocks (0–4, 26–29) are near 1.000; blocks 21–24 are the least stable (0.976–0.979).
- The CFG gap between cond and uncond predictions shrinks from 17 % (step 1) to 2.5–3 % (steps 40–45): reusing or skipping the
  uncond pass is most plausible late in the run.
- **Upper bound (not a measurement):** reusing blocks on about 2 of every 3 steps in the stable window skips about 23 of 50 steps,
  so denoise ≤ about 1.8×. Quality gates (LPIPS, flicker) decide. This is the first lever in the project with room above 1.3×.
- Prediction 5 (caching viable at 30+ steps): **supported by the probe**; implementation and the quality gate are pending (Tier A).
- The uncompiled run's final latents are saved (runs/exp-104-a1/latents.safetensors, shape [1, 16, 9, 60, 104]). R1: they are
  eager-mode latents, so they are not the frozen decoder reference (that is exp-105).

## Notes for the speed track (queued after the decoder)
- **Cross-model rule** (library/notes/step-caching-viability.md): step caching is viable at 50 steps (Wan, 77 %) and dead at
  4 distilled steps (Klein, 4.9 %). That depends on step count and distillation, not architecture. Token merging fails on both
  (library/notes/token-merging-anisotropy.md).
- **Do not stack the two caching wins.** Block-step caching and CFG skipping reuse the same redundancy: a step whose blocks barely
  changed is also one whose CFG gap is small. When implemented, they are measured as **one combined mechanism with one quality
  gate**, not two multipliers. Any clip-time projection stays out of reports until it has been run.

## Decoder: frozen-pattern tests from the frozen latents (exp-106..108)

| Run | Frames | Peak MLX | vs reference | Note |
|---|---|---|---|---|
| exp-106 stock `tiled_auto` | 36 | 17.90 GB | **bit-identical** to the frozen stock frames | validates the decode-only tool |
| exp-107 stock untiled | 9 (3 latent) | 16.77 GB | (reference) | a 9-frame prefix already needs 16.8 GB |
| exp-107 **eval barriers** (frozen pattern) | 9 | 16.77 GB | **bit-identical** to untiled | **no saving** (Klein: 9.55 → 7.26 GB) |
| exp-108 eval barriers, untiled | 36 | – | – | **crash:** `metal::malloc` asked for one 11.04 GB buffer, over the 9.53 GB per-buffer limit |

**Cause (exp-107 per-module profile):** a single `CausalConv3d` (96→96, 3×3×3) on a 1.71 GB fp32 input (12 frames × 480×832 × 96
ch) needs **+13.0 GB of working memory**, about 7.6× its input. Half-resolution layers: 0.86 GB input, +11.7 GB. The peak is the
**workspace inside one 3D convolution**, not intermediates kept alive across the graph. That is why the barrier pattern (a
lazy-graph-lifetime fix) does not fire here.

- Hit matrix: `eval-barriers-between-blocks` **applies but has no effect** on Wan's decoder (a different failure mode). The frozen
  2D `vae-exact-row-chunked-decode` does not apply as-is (3D causal convolutions, temporal cache).
- **Prediction 4 is strengthened:** the untiled stock decode of 36 frames is **impossible** on this machine (exceeds Metal's
  per-buffer limit), not merely swapping. Only mlx-video's approximate tiling runs, at 17.9–18.6 GB.
- Exactness consequence: no untiled stock reference exists for the full video. Exact decoders are bit-checked against the untiled
  stock decode on prefixes where it runs, and compared to the stock tiled frames with PSNR, LPIPS and flicker for the full video.

## Exact streaming decoder (v3): temporal streaming + Winograd-aligned height chunking, fp32

| Run | Frames | Peak MLX | Process footprint | Swap | vs stock |
|---|---|---|---|---|---|
| exp-115 v3, 12-frame prefix | 12 | 8.41 GB | – | – | **bit-identical** to stock untiled |
| exp-115 v3, 16-frame prefix | 16 | 8.76 GB | – | – | **bit-identical** to stock untiled |
| exp-115 v3, full | 36 | 8.78 GB (flat per chunk: 7.99 → 8.78) | 12.47 GB (cache uncapped) | grew | frames 1–16 **bit-identical** to the untiled 16-frame prefix; frames vs stock tiled: 46.8 dB min / 56.0 mean, max diff 26, flicker excess 0.016 |
| exp-117 v3, full, cache cap 0.5 GB | 36 | 8.85 GB | **9.23 GB** | **none** | frames 1–16 **bit-identical** |

Stock: tiled `auto` 17.9–18.6 GB (swaps), 134 s; untiled 36 frames cannot run (Metal per-buffer limit). v3 decodes in 47–51 s.
The **layered claim**: bit-identical on the first 16 frames (the longest prefix stock can decode untiled), and Tier E vs stock
tiled on the rest, where stock tiled is itself the approximation. Memory is flat in video length, so lengths stock cannot
decode untiled are decodable.

Remaining peak (exp-116, per-module profile inside the streaming decoder): about 3.8 GB of fp32 causal state carried between
chunks (2 frames per 3D conv), plus about 3.3 GB transient per full-resolution 3D conv (stock concatenates, pads and transposes the
6-frame state + chunk block).

Engineering notes: (1) a slice kept every conv's whole chunk input alive until the state was copied; (2) re-padded upsampler
strips broke bit-identity under Winograd (60 dB) until strips were grid-aligned (library/notes/determinism-taxonomy.md).

### Decoder v4–v6 (option 1 and follow-ups; all cache-capped 0.5 GB, swap-flat, bit-identical on the 12- and 16-frame prefixes and on frames 1–16 of the full decode)

| Version | Change | Full video: peak MLX / footprint | Decode time |
|---|---|---|---|
| v4 (exp-119) | each frame's 3-frame window built directly from state + new frames (option 1) | 8.85 / 9.12 GB | 56.3 s |
| v5 (exp-122) | upsampler strips built directly from input rows (no full 4× nearest tensor) | 8.66 / 8.30 GB | 55.0 s |
| **v6 (exp-125)** | **+ eval barriers between decoder layers inside each chunk** | **8.42 / 8.19 GB** | **56.1 s** |

- Option 1 did its local job (full-resolution 3D conv transient +3.34 → +2.05 GB; residual-block peak 8.29 → 7.00 GB), but the peak
  moved to the spatial upsampler (+3.94 GB, the full 4× tensor). v5 removed that.
- After v5: largest single-module peak 7.00 GB vs a whole-chunk peak of 8.54 GB, a graph-lifetime gap by the diagnosis rule.
  Barriers within the chunk recovered only 0.24 GB.
- **Floor: about 3.8 GB of fp32 causal state** (2 frames per 3D conv) carried between chunks. Lower needs whole-decoder spatial
  tiling (option 2, deferred; NOTES.md).
- **Quotable decoder (exp-125, R3-valid timing):** 8.19 GB footprint / 8.42 GB MLX, 56.1 s, swap-flat. Bit-identical on frames 1–16
  (the longest prefix stock decodes untiled); vs stock tiled 46.8 dB min / 56.0 mean, max diff 26, flicker excess 0.016.
  Stock: tiled 17.9–18.6 GB (swaps), untiled cannot run (Metal per-buffer limit).

## DiT block streaming (exp-126; frozen engine via adapter subclasses; 8 steps, stop after denoise, cache cap 0.5 GB, alternating A/B)

| Mode | Median step (r1 / r2) | Denoise peak MLX | Footprint | Final latents |
|---|---|---|---|---|
| stock (mlx-video whole-model compile) | 22.11 / 22.64 s | 4.62 GB | 5.62 / 5.70 GB | reference |
| per-block compile, resident (R1 reference) | 22.46 / 22.69 s | 4.60 GB | 5.61 GB | **bit-identical to stock** |
| **block streaming (prefetch, lag 1)** | **22.63 / 22.72 s** | **2.41 GB** | **3.44 GB** | **bit-identical to both** |

- **Prediction 2 confirmed:** streaming overhead vs the per-block-compiled reference is **+0.74 % / +0.12 %**, below Klein's +2.1–2.8 %.
  Blocks are small (92.9 MB) and the compute per block is large (about 14k tokens), so prefetch hides the load fully. Vs stock: +2.36 % (round 1,
  thermally unbalanced) / +0.33 % (round 2).
- Unlike Klein, streaming with per-block compilation does not change Wan's numerics: bit-identical to stock at 8 steps (exp-126) and,
  verified at the shipping schedule, at 50 steps on both RNG paths (exp-130 = exp-127; exp-131 = frozen exp-105).
- Streaming starts at the first denoising step, after mlx-video's cross-attention K/V precompute (which reads every block once).
- Engine frozen: positional-argument folding is an adapter subclass (adapters/wan/dit_stream.py).

**Decision (2026-10-21):** decoder closed at v6 (8.19 GB). Micro-steps (~6.5–7 GB projected) skipped: they serve no machine that
8.2 GB doesn't already serve, and option 2 (spatial halves) would subsume them. Option 2 stays behind its trigger (8 GB-Mac
decision). Projection miss recorded as library/notes/memory-fixes-are-serial.md.

## Composed pipeline exp-127 (50 steps, streamed TE + streamed DiT + streaming decoder v6, cache cap 0.5 GB)

Memory and time (swap-flat, valid under R3): text encode 2.07 GB MLX, denoise 2.42 GB, decode 8.58 GB; process footprint peak
**8.57 GB**; total 1266 s (21.1 min; fan added at ~step 20, see NOTES.md).

**Exactness: NOT claimed yet.** Final latents vs the frozen v1 latents: cosine 0.06, 0 % identical elements (uncorrelated, a
different but valid video). Diagnosis:
- text embeddings produced inside the pipeline: bit-identical to the frozen capture (exp-128)
- streamed encoder construction consumes the global RNG exactly as the stock constructor (bench/rng_probe.py)
- initial noise: bit-identical to a replay of stock mlx-video's RNG order (seed -> T5Encoder ctor -> load_wan_model -> normal) (exp-129)
- **cause: the frozen baseline (exp-103) and frozen latents (exp-105) were run with `--context-ref`, whose stub loader never
  constructs the T5Encoder.** Model constructors draw random init weights, so the baseline's initial noise came from a shifted
  point of the seed-42 stream. The composed run reproduces real stock mlx-video's RNG; the frozen latent reference does not.
  The frozen reference is a valid stock DiT+VAE trajectory, just not the one stock mlx-video draws for seed 42.
- verification queued: exp-130-stock (stock DiT + exact TE, should equal exp-127 latents) and exp-131 (streamed DiT with
  --context-ref, should equal the frozen latents), both 50 steps.

## Wan 2.1 1.3B before/after: stock vs composed exact pipeline (832×480, 33→36 frames, 50 steps, CFG, seed 42, M5 16 GB)

| Stage | Stock mlx-video | Composed exact pipeline | Exactness (composed vs stock) |
|---|---|---|---|
| Text encoder (umT5-XXL, fp32 upcast) | **23.1 GB MLX / 23.9 GB footprint**, swap +5.6 GB, encode 141 s\* (exp-101): **cannot run in 16 GB without heavy swap** | **2.07 GB MLX / 2.59 GB**, encode 5.3–6.2 s, swap flat (exp-102, exp-127) | **bit-identical** embeddings (exp-102; re-checked inside the pipeline, exp-128) |
| DiT denoise (30 blocks, 2.64 GB bf16) | 4.6 GB MLX / 5.6 GB footprint, 22.1–22.6 s/step (exp-126) | **2.42 GB MLX / 3.44 GB**, +0.1–0.7 % vs per-block compile (exp-126), 23.1 s/step in the full run | **bit-identical at 50 steps**: exp-130 (stock DiT) = exp-127 (streamed) |
| VAE decode (fp32) | tiled `auto`: **17.9–18.6 GB MLX**, swap, 83–113 s\* (exp-103, exp-132); untiled: **cannot run** (11.0 GB single buffer > 9.53 GB Metal limit) | **8.42 GB MLX / 8.19 GB**, 56 s, swap flat (exp-125); 8.58 GB / 56.7 s inside the pipeline (exp-127) | frames 1–16 **bit-identical** to stock untiled (longest prefix stock can decode); all 36 vs stock tiled: Tier E, 46.8 dB min / 56.0 mean, flicker excess 0.016 (exp-125) |
| **End to end** | **2 of 3 stages cannot run within 16 GB** (encoder swaps, decoder swaps tiled / impossible untiled); the frozen-embeddings baseline peaked at 18.07 GB footprint with swap (exp-103) | **8.57 GB footprint peak, 1266 s (21.1 min), swap flat** (exp-127) | final latents **bit-identical** to stock (exp-130); end-to-end frames vs stock mlx-video's own (tiled) decode of those latents: **Tier E, 49.5 dB min / 58.7 mean, max pixel diff 17, flicker excess 0.002** (exp-132) |

\* Swap-inflated (R3): memory-valid, not speed-valid. These are what stock takes on this 16 GB machine, not stock's speed. The composed
pipeline's numbers are the only swap-flat end-to-end timings in the table; its streaming decoder (56 s) is faster than stock's
decode on this machine (83–113 s) because it does not swap.

Conditions and caveats:
- Peak footprint = macOS process `peak_footprint` (`/usr/bin/time -l`); MLX = `mx.get_peak_memory` per stage. Cache cap 0.5 GB on all composed runs.
- Timing is quotable only where swap stayed flat (R3). Stock stages that swapped are reported for memory, not speed.
- m5-test throttles (fanless; Heavy thermal for most of every 50-step run). A fan was added under it at ~12:30, mid exp-127 (NOTES.md).
- Stock text-encoder and decoder numbers come from stage-isolated runs; stock mlx-video was never run end to end because its encoder alone exceeds 16 GB.
- Reference correction: the frozen v1 latents (exp-105) came from the `--context-ref` path, which skips the T5 constructor and so draws
  the initial noise from a shifted point of the seed-42 stream. Real stock latents for seed 42 are exp-130 (stock DiT, exact streamed encoder,
  stock RNG path). **Confirmed (exp-131):** the streamed DiT on the --context-ref path reproduces the frozen latents bit for bit, so the shift alone
  explains the exp-127 mismatch, and streaming = stock at 50 steps holds on both RNG paths (exp-130 and exp-131).

exp-132 (stock tiled `auto` decode of the true-stock latents, after a 5-min cool-down to Nominal): 17.90 GB MLX, 16.92 GB footprint,
swap grew (+1.2 GB), 113 s. Its frames differ from the composed pipeline's by Tier E only (49.5 dB min), as expected: stock's
spatial tiling blends tile overlaps, so it is not the exact decode of the latents; the composed decoder reproduces stock's untiled
decode bit for bit wherever that can run (16-frame prefix, exp-125). The end-to-end claim is therefore: **latents bit-identical to
stock; video bit-identical to stock's untiled decoder where it can run, and Tier E (≥ 49.5 dB) to stock's default tiled decoder.**

**Reference retired, cause of death:** the frozen v1 latent reference (exp-105, and the baseline exp-103 it reproduces) was produced on
the `--context-ref` path, whose stub text-encoder loader skips the T5Encoder constructor; model constructors draw random init weights,
so its initial noise came from a shifted point of the seed-42 RNG stream. Confirmed by exp-131 (the streamed DiT on that same path
reproduces it bit for bit). Replaced by exp-130 (true stock RNG path) as the v1 reference.

## Generality

Klein's memory chapter took two days; Wan's took one working day — this morning's encoder fix to this table — on a model family the
playbook had never seen, including one genuinely new component (the 3D decoder) and two live debugging detours.

## Step-caching track: A2 (combined step + CFG reuse, one mechanism, one gate; Tier A by design)

Mechanism (adapters/wan/step_cache.py), schedule from A1: steps 1–11 full; 12–34 every second step skipped (block-stack residual
reused, embeddings and head recomputed); 35–46 cycle full / skip / cond-only (B=1, uncond rebuilt from the last full CFG gap);
47–50 full. 30 full, 16 skip, 4 cond: ideal denoise speed-up 1.56×. Runs on the composed low-memory pipeline (streamed TE, streamed
DiT, streaming decoder). Exact timings reused from recorded runs (exp-131 denoise, exp-127 end to end), not re-run.

exp-134 (v1, fox):

| | Exact | A2 | |
|---|---|---|---|
| Denoise | 1148 s (exp-131) | **706 s** | 1.63× measured; ~1.55× schedule-attributable (see below) |
| End to end | 1266 s (exp-127) | **771 s (12.9 min)** | 1.57–1.64× |
| Denoise peak MLX | 2.41 GB | 2.58 GB | +0.17 GB = the cached residual (2 × 14040 × 1536 fp32), as predicted |
| Footprint peak | 8.57 GB | 8.61 GB | decoder-bound, unchanged |
| Swap | flat | flat | |

Quality vs the exact video (exp-127 frames, exp-130 latents): LPIPS max **0.027** (frame 1), mean 0.011: **Tier A passes**.
PSNR min 31.6 / mean 37.2 dB. Latent cosine 0.998, rel L2 6.2 %.
- **Temporal:** no flicker spike (largest per-transition excess +0.16 levels on ~5-level frame changes). Systematic effect is mild
  **motion damping: cached frame-to-frame change is 0.97× exact**, uniform across the clip. The fast-motion prompt (v2) is the stress test.
- **Speed decomposition:** A2's full steps ran at 22.2 s median vs 23.2 s in exp-131: skipped steps let the passively cooled chip
  rest, so the remaining work runs faster. Thermally normalised, the schedule itself gives ~1.55×; the extra ~5 % is machine-specific.
- **Post-skip stall:** not recorded directly in exp-134 (bench did not save streamer stats; fixed for later runs). Upper bound from
  step times: full steps after a skip 22.21 s median vs 21.95 s after a full step, ≤ 0.26 s/step (≤ 1.2 %), confounded by thermal drift
  (post-skip steps fall later in the run). Direct block-0 stall measured in exp-a2-v2/v3.
- **First-frame LPIPS (0.027, worst):** the drift is already in the latents, not the decoder: latent rel L2 7.8 % in latent frames 0–1
  vs 5.4 % mid-clip. The causal-start-padding explanation is not supported; why the trajectory drifts most at the clip start is open
  (candidate: latent frame 0 encodes 1 video frame, the others 4 each; untested).

### A2 on all three prompts

| Prompt | Speed-up end to end | LPIPS max / mean | PSNR min / mean | Motion vs exact | Tier A |
|---|---|---|---|---|---|
| v1 fox (exp-134) | 1.57–1.64× | 0.027 / 0.011 | 31.6 / 37.2 dB | 0.97× | pass |
| v2 ocean waves (exp-a2-v2 vs exp-ref-v2) | 1.63× (752 vs 1226 s) | **0.075 / 0.049** (16 of 36 frames > 0.05) | 25.7 / 29.4 dB | 0.98× | **fail** |
| v3 neon sign (exp-a2-v3 vs exp-ref-v3) | 1.65× (747 vs 1231 s) | **0.092 / 0.062** (all 36 frames > 0.05) | 25.9 / 27.5 dB | 1.00× | **fail** |

Failing the gate does not mean the waves video looks bad: by eye it is the same scene (sun, rocks, horizon, where the waves break)
with differently shaped foam and spray, and the motion is not slowed (0.98×). It means the claim "indistinguishable from exact"
cannot be made at 1.56× on chaotic content. The strictness is the product; a looser gate would just be marketing.
Error shape: largest at the clip start (LPIPS 0.074 on frame 1) easing to 0.036 at the end, the same shape as the fox at ~3× the size.
Exact v2 reference: 1226 s, 8.52 GB footprint, swap flat (exp-ref-v2).

Post-skip streaming stall (exp-a2-v2, recorded directly): block-0 stall 0.0 ms after a skip (n=16) vs 0.0 ms after a normal step;
total stall 0.7 ms/step either way. Recorded in library/patterns/block-weight-streaming-prefetch.yaml as a measured-absent interaction.

Sweep plan (A3): A3-a (skip window starts at step 16 instead of 12; ~1.47× ideal) on waves first; A3-b (skip every third step) if a is
borderline; A3-c (no cond steps) dropped: it removes reuse where the error is already smallest. A3-d: dynamic rule, skip only when the
last full step's measured block-stack change rate is below a threshold, so the schedule adapts per prompt; thresholds calibrated from
the change rates logged in A3-a.

Neon by eye: sign, stall, main umbrella figure, reflections and colours match; small background content differs (pedestrians and
umbrellas on the right appear, vanish or move). Error is uniform over the clip (0.053–0.092), not start-weighted.

**Block-stack change rate per denoising step (logged on every full step, exp-a2-v3):** high in warm-up (0.16 at step 2 → 0.024 by
step 9), lowest mid-run (0.019–0.026 per step, steps 13–35), then **rising through the late steps** (0.030 at 38, 0.051 at 44, 0.081 at
47, 0.28 at 50). Correction to the A3-c reasoning: the "error largest at the start" shape is along video time (output frame index),
not denoising steps. Along the denoising axis, late steps change the most after warm-up, so late skips/cond reuse are not the
cheapest reuse. A3-d handles this by construction: rising rates stop skipping.

A3-d threshold 0.022: inside the measured mid-run band (0.019–0.026), so on neon it skips about half the mid-run steps A2 skipped and
none late. Runs exp-a3d-v3 (neon, hardest), then exp-a3d-v2 (waves).

### Pre-registered before any A3-d result (2026-10-21 15:39; exp-a3d-v3 still denoising, no gate exists)

**Ship criterion for the dynamic mode (A3-d), revised before its data:** the "≥ 1.3× on all three prompts" bar was written for a static
schedule (speed fixed, quality floats). A dynamic threshold inverts that: quality is held by construction (skip only when the
measured change is low) and speed floats with content. So:
- **Hard gate:** LPIPS < 0.05 on every frame of all three prompts (v1 fox, v2 waves, v3 neon).
- **Speed:** reported per prompt. The mode ships if the quality gate holds everywhere and the speed-up is meaningful on the easy/medium
  prompts, even if neon lands at ~1.1–1.2×. Product sentence if it ships: "fast mode: up to ~1.6×, content-dependent, quality-bounded".
- **If the quality gate fails under dynamic control, the track closes as stated before:** curve recorded, no weaker claim shipped.
- Thresholds may be swept (more than one point on the curve); the gate applies to whichever threshold is proposed for shipping, on
  all three prompts, with the threshold fixed before that prompt's run.
