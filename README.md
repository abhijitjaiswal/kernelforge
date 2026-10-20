# kernelforge

Verified-fast inference research. First workload: FLUX.2 Klein 4B on an Apple M5 (m5-test). Brief: `PROJECT_BRIEF.md`.

- Code lives here (local Mac); every measurement runs on **m5-test** over SSH. m5-test writes only to `~/kernelforge` and `~/kf-venv`; `~/imagegen` is read-only.
- `scripts/sync.sh` pushes code, `scripts/remote.sh <job> '<cmd>'` runs it detached on m5-test, `scripts/pull.sh` fetches `runs/` + `baselines/`, `scripts/record.py` logs an experiment to `results.jsonl`, `RESULTS.md` and `dataset/candidates.jsonl`.
- `python3 dashboard/server.py` serves the tracker at http://localhost:8790.

## Reproduce Phase 0
    scripts/sync.sh
    scripts/remote.sh phase0  'sh scripts/phase0.sh'    # exp-001..004  warm 5-run batches (baselines saved)
    scripts/remote.sh phase0b 'sh scripts/phase0b.sh'   # exp-005..008  lifecycle, cold single image, memtrace
    scripts/remote.sh phase0c 'sh scripts/phase0c.sh'   # exp-009..011  cache cap + alternating bf16/q8 A/B
    scripts/pull.sh
Report: `reports/phase0.md`.

## Protocol rules
- **R1 (execution mode):** every correctness comparison uses a reference produced in the *same* execution mode (eager vs
  `mx.compile`). Eager candidates (hooks, streaming, profiling) are verified against an eager reference; compiled candidates
  against a compiled reference. Cross-mode differences are reported separately and are not attributed to the optimisation.
  Background: `library/notes/eager-vs-compiled-numerics.md` (exp-040: cos 0.9967-0.9996 between modes).
- **R2 (simulation):** `mx.set_memory_limit` is a soft limit; no smaller-machine claim is made from it. 8 GB claims need a real 8 GB Mac.
- **R3 (timing validity):** a run whose swap grew is memory-only; timing comparisons alternate A/B processes with cool-downs.
