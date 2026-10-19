"""Model-agnostic stage-wise benchmark: separates weight load from warm compute. Runs on m5-test.

    python bench/stages.py --exp exp-013 --adapter flux2_klein --opts '{"weights":"bf16"}' --prompt p1 --runs 5

For each stage the adapter declares (in pipeline order) the harness:
  1. load:   materializes the stage's weights (mx.eval of its parameters), timed  -> load_s
  2. warm-up: one untimed run (mx.compile tracing, kernel JIT)
  3. N timed runs, each ending in mx.eval of the stage outputs                    -> compute_s
     stages with steps (denoise) are also timed per step                          -> step_s
  4. free:   drops the weights and clears the MLX cache before the next stage
The chained output (last run of every stage) is saved as <prompt>.png and verified against --ref.

Adapter contract: Adapter(opts).stage_plan(prompt, seed, steps, res) -> list of objects with
  name, load() -> None, run(ctx) -> None (reads/writes ctx dict, returns arrays to eval),
  steps: int (0 = single shot; else run_step(ctx, i)), reset(ctx) before each timed run, free(ctx).
  Optional adapter.stage_ctx_init seeds ctx; ctx["image"] is the final output; ctx["_save"] is saved.
"""

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def st(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)} if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--prompt", default="p1")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--ref", help="dir with <prompt>.png to verify the chained output against")
    a = ap.parse_args()

    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("KF_CHILD") != "1":  # re-exec under /usr/bin/time -l for the OS peak footprint
        env = dict(os.environ, KF_CHILD="1")
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, *sys.argv], env=env, stdout=lf, stderr=tf).returncode
        from bench.run import parse_time_l
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "notes": "crashed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "stages_summary", "returncode")}, default=str))
        sys.exit(rc)

    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from bench.memtrace import MemTrace
    from bench.power import Power, thermal_now
    from bench.run import swap_used_mb
    from bench.verify import compare

    prompt = json.loads((ROOT / "bench" / "prompts.json").read_text())[a.prompt]
    opts = json.loads(a.opts)
    swap0, thermal0 = swap_used_mb(), thermal_now()
    power = Power(out / "power.txt").start()
    trace = MemTrace().start()
    Adapter = importlib.import_module(f"adapters.{a.adapter}.pipeline").Adapter
    ad = Adapter(opts)
    plan = ad.stage_plan(prompt, seed=a.seed, steps=a.steps, res=a.res)
    ctx, stages = dict(getattr(ad, "stage_ctx_init", {})), {}
    g = 1024**3
    for s in plan:
        rec = {}
        trace.mark(f"{s.name}:load")
        mx.reset_peak_memory()
        t = time.perf_counter()
        s.load()
        rec["load_s"] = time.perf_counter() - t
        rec["weights_gb"] = round(mx.get_active_memory() / g, 3)
        trace.mark(f"{s.name}:warmup")
        t = time.perf_counter()
        s.reset(ctx)
        mx.eval(s.run(ctx))
        rec["warmup_s"] = time.perf_counter() - t
        trace.mark(f"{s.name}:timed")
        mx.reset_peak_memory()
        comp, steps = [], []
        for _ in range(a.runs):
            s.reset(ctx)
            t = time.perf_counter()
            if s.steps:
                for i in range(s.steps):
                    t1 = time.perf_counter()
                    mx.eval(s.run_step(ctx, i))
                    steps.append(time.perf_counter() - t1)
            else:
                mx.eval(s.run(ctx))
            comp.append(time.perf_counter() - t)
        rec["compute_s"] = st(comp)
        rec["compute_runs_s"] = comp
        if steps:
            rec["step_s"] = st(steps)
            rec["step_s_by_index"] = {i: st(steps[i::s.steps]) for i in range(s.steps)}
        rec["peak_mlx_gb"] = round(mx.get_peak_memory() / g, 3)
        trace.mark(f"{s.name}:free")
        s.free(ctx)
        stages[s.name] = rec
        print(f"[stages] {s.name}: load {rec['load_s']:.2f}s warm-up {rec['warmup_s']:.2f}s "
              f"compute {rec['compute_s']['median']:.3f}s peak {rec['peak_mlx_gb']} GB", flush=True)
    trace.mark("done")
    trace.stop()
    trace.dump(out / "memtrace.json")
    power.stop()
    psum = power.summary()
    (out / "power.json").write_text(json.dumps(psum))
    psum.pop("series", None)
    psum["thermal_at_start"] = thermal0
    swap1 = swap_used_mb()

    img = ctx["image"]
    Image.fromarray(img).save(out / f"{a.prompt}.png")
    verify = {}
    if a.ref:
        refp = ROOT / a.ref / f"{a.prompt}.png"
        verify[a.prompt] = compare(img, np.asarray(Image.open(refp).convert("RGB")), str(out / f"{a.prompt}_diff.png"))
    for k, v in ctx.get("_save", {}).items():  # e.g. final latents for VAE-only experiments
        mx.save_safetensors(str(out / f"{k}.safetensors"), {k: v})

    res = {
        "exp_id": a.exp, "kind": "stages",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": ad.model_id,
        "config": {"resolution": a.res, "steps": a.steps, "seed": a.seed, "prompt_ids": [a.prompt], **opts},
        "runs": a.runs,
        "stages": stages,
        "stages_summary": {k: {"load_s": round(v["load_s"], 2), "compute_s": round(v["compute_s"]["median"], 3),
                               **({"step_s": round(v["step_s"]["median"], 3)} if "step_s" in v else {})} for k, v in stages.items()},
        "compute_total_s": sum(v["compute_s"]["median"] for v in stages.values()),
        "load_total_s": sum(v["load_s"] for v in stages.values()),
        "memtrace": trace.summary(),
        "power": psum,
        "swap_used_mb_before_after": [swap0, swap1],
        "swap_grew": (swap1 or 0) - (swap0 or 0) > 32,
        "ref": a.ref,
        "verify": verify,
        "psnr_db_vs_baseline": verify.get(a.prompt, {}).get("psnr_db"),
        "max_abs_pixel_diff": verify.get(a.prompt, {}).get("max_abs_pixel_diff"),
        "mlx_version": mx.__version__,
    }
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print("[stages] done", flush=True)


if __name__ == "__main__":
    main()
