# NOTES

## m5-test (measurement machine), recorded 2026-10-19
- MacBook Air (Mac17,3), Apple M5: 10 CPU cores (4 Super + 6 Efficiency), 8-core GPU, Metal 4, 16 GB unified memory.
  MLX `device_info`: architecture `applegpu_g17g`, max_recommended_working_set_size 11.84 GiB, max_buffer_length 8.88 GiB.
- macOS 26.6 (25G72). Disk: 271 GB free. On AC power, battery 100 %, Low Power Mode off.
- Fanless: expect thermal throttling (see per-run drift in results).
- Swap already in use before any experiment: about 2.0 GB of 3 GB, left over from earlier sessions. Runs use the swap *delta*.

## Safety rules for m5-test (user request)
- `~/imagegen` (user's working setup) is **read-only**: weights are loaded from it and nothing is written there.
- All writes go to `~/kernelforge` (code, runs, baselines) and `~/kf-venv` (Python env). Undo: `rm -rf ~/kernelforge ~/kf-venv`.
- No sudo, no system installs. tmux is not installed, so jobs run under `nohup` (scripts/remote.sh).
- `sudo powermetrics` needs a password over SSH (`sudo -n` fails), so it is **not captured**. Thermal behaviour is inferred from run-to-run timing drift.

## Stack (Phase 0)
- Python 3.12.15 venv at `~/kf-venv` (uv 0.13.1): **mflux 0.23.0** (latest on PyPI at setup), **MLX 0.32.4**, numpy 2.5.4, pillow 12.3.0, torch 2.14.1 (pulled in by mflux).
- mflux supports FLUX.2 Klein 4B natively (`Flux2Klein`, `ModelConfig.flux2_klein_4b`), so no fallback was needed.

## Models and licenses
- black-forest-labs/FLUX.2-klein-4B @ e7b7dc27f91deacad38e78976d1f2b499d76a294, **Apache-2.0** (from the HF card, also recorded in
  `~/imagegen/README.md` and `SOURCE.json`). No restriction on this use. Already on m5-test, so nothing was downloaded.
  - bf16: `~/imagegen/hf/hub/.../snapshots/e7b7dc2…` (15 GB: Qwen3 text encoder, transformer, VAE)
  - 8-bit: `~/imagegen/models/flux2-klein-4b-q8` (8 GB, made with mflux-save 0.23.0 by the user)
- Pipeline facts (from the mflux source): 4 steps, guidance 1.0. guidance ≤ 1 means **no CFG** (one transformer pass per step, no
  negative prompt encoded). The text encoder is Qwen3, with hidden states from layers 9/18/27, max 512 tokens. The `predict` closure
  is wrapped in `mx.compile` on M3 and later.

## Harness decisions
- Stage timing wraps mflux's own `generate_image()` (encode and VAE decode are wrapped with `mx.eval` plus timestamps), so the
  baseline is the stock code path, not a re-implementation.
- `free_te` lifecycle: prompts are encoded up front (after one warm-up encode), then the text encoder is dropped. Its measured
  encode time is added to each image's total, so totals stay end-to-end.
- Non-stock lifecycles do **not** `mx.eval` the parameters at load. mflux/MLX weights are lazy and mmap-backed, so a component
  only becomes resident when first used (exp-000: active memory after freeing the text encoder = 0.007 GB, with the transformer not yet loaded).
- Two memory numbers: MLX `get_peak_memory` (reset per image) and the OS "peak memory footprint" from `/usr/bin/time -l`
  (whole process, including load). They differ a lot (exp-000: 9.49 vs 13.31 GB); this needs investigating.
- Timing interleaving: inside one process, prompts are interleaved across rounds. Configs run sequentially with 120 s cool-downs
  (the 16 GB machine cannot hold two pipelines). Close A/B comparisons will alternate processes.

## Where an LLM proposer would plug in (Phase 3 note)
- (to fill in Phase 3)

## Decisions log
- 2026-10-20 Gate 1: 1.3x Tier E speed goal dropped (roofline: compute-bound at ~95% of NAX peak). Fusion candidate 1 deferred
  until a bandwidth-bound target exists. mx.set_memory_limit is a soft limit: smaller-machine simulation is invalid; 8 GB claims
  need a real 8 GB Mac. New Tier A speed track (A1 probe -> A2 -> A3), timeboxed 4 days, after the footprint batch.
- xctrace unavailable (m5-test has Command Line Tools only; Xcode not installed). Profiling uses MLX timing + microbenchmarks.
- 2026-10-20: speed track CLOSED, A2/A3 skipped (user decision). Finding: reports/speed_track_closure.md. Eager-vs-compiled
  numerics promoted to protocol rule R1 (README) + library/notes/eager-vs-compiled-numerics.md.
