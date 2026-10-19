"""Model-agnostic benchmark + verification harness. Runs on m5-test.

    python bench/run.py --exp exp-001 --adapter flux2_klein --opts '{"weights":"q8"}' \
        --prompts p1,p2,p3 --runs 5 --ref baselines/q8_free_te [--save-baseline DIR]

Writes runs/<exp>/: result.json, per_run.jsonl, <prompt>.png, <prompt>_diff.png, time.txt, log.
The process re-executes itself under /usr/bin/time -l to capture the OS "peak memory footprint".
"""

import argparse
import importlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def swap_used_mb():
    m = re.search(r"used = ([\d.]+)M", sh("sysctl vm.swapusage"))
    return float(m.group(1)) if m else None


def stats(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def parse_time_l(text):
    out = {}
    for key, name in [("peak memory footprint", "peak_footprint_gb"), ("maximum resident set size", "max_rss_gb")]:
        m = re.search(r"(\d+)\s+" + key, text)
        if m:
            out[name] = round(int(m.group(1)) / 1024**3, 3)
    m = re.search(r"([\d.]+) real", text)
    if m:
        out["process_wall_s"] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--prompts", default="p1,p2,p3")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--ref", help="baseline dir with <prompt>.png to verify against")
    ap.add_argument("--save-baseline", help="copy output pngs into this dir (refuses to overwrite)")
    ap.add_argument("--no-warmup", action="store_true", help="cold single-image measurement (needed for lifecycle=staged)")
    ap.add_argument("--meta", default="{}", help="extra fields for result.json (change, hypothesis, track...)")
    a = ap.parse_args()

    out = ROOT / "runs" / a.exp
    if os.environ.get("KF_CHILD") != "1":
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, KF_CHILD="1")
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, *sys.argv], env=env, stdout=lf, stderr=tf).returncode
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "verdict": "failed", "notes": "run crashed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "total_s", "peak_mem_gb", "psnr_db_vs_baseline", "returncode")}, default=str))
        sys.exit(rc)

    import numpy as np
    from PIL import Image

    from bench.memtrace import MemTrace
    from bench.verify import compare

    prompts_all = json.loads((ROOT / "bench" / "prompts.json").read_text())
    pids = a.prompts.split(",")
    opts = json.loads(a.opts)
    idle = sh("top -l 1 -n 5 -o cpu | head -12")
    swap0 = swap_used_mb()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[bench] {a.exp} adapter={a.adapter} opts={opts}", flush=True)

    trace = MemTrace().start()
    trace.mark("load")
    Adapter = importlib.import_module(f"adapters.{a.adapter}.pipeline").Adapter
    ad = Adapter(opts)
    ad.mark = trace.mark
    trace.mark("prepare")
    prep = ad.prepare([prompts_all[p] for p in pids])
    print(f"[bench] loaded in {ad.load_s:.1f}s, prep {prep}", flush=True)

    gen = lambda pid: ad.generate(prompts_all[pid], seed=a.seed, steps=a.steps, width=a.res, height=a.res)
    warm_s = None
    if not a.no_warmup:
        trace.mark("warmup")
        t = time.perf_counter()
        gen(pids[0])  # untimed warm-up (mx.compile, kernel JIT)
        warm_s = time.perf_counter() - t
    print(f"[bench] warm-up {warm_s}s", flush=True)

    per_run, per_prompt = [], {}
    # Interleave prompts across rounds so thermal drift spreads over every prompt.
    for r in range(a.runs):
        for pid in pids:
            o = gen(pid)
            rec = {"round": r, "prompt_id": pid, **o["times"], **{f"mem_{k}": v for k, v in o["mem"].items()}}
            pp = per_prompt.setdefault(pid, {"runs": [], "first": None, "deterministic": True})
            if pp["first"] is None:
                pp["first"] = o["image"]
                Image.fromarray(o["image"]).save(out / f"{pid}.png")
            elif not np.array_equal(pp["first"], o["image"]):
                pp["deterministic"] = False
            pp["runs"].append(rec)
            per_run.append(rec)
            print(f"[bench] r{r} {pid} total {rec['total_s']:.2f}s denoise {rec['denoise_s']:.2f}s vae {rec['vae_s']:.2f}s peak {rec['mem_peak_mlx_gb']}GB", flush=True)
    swap1 = swap_used_mb()
    trace.mark("done")
    trace.stop()
    trace.dump(out / "memtrace.json")

    with open(out / "per_run.jsonl", "w") as f:
        for rec in per_run:
            f.write(json.dumps(rec) + "\n")

    verify = {}
    if a.ref:
        for pid in pids:
            refp = ROOT / a.ref / f"{pid}.png"
            verify[pid] = compare(per_prompt[pid]["first"], np.asarray(Image.open(refp).convert("RGB")), str(out / f"{pid}_diff.png")) if refp.exists() else {"error": "no reference"}
    if a.save_baseline:
        bd = ROOT / a.save_baseline
        bd.mkdir(parents=True, exist_ok=True)
        for pid in pids:
            dst = bd / f"{pid}.png"
            if dst.exists():
                sys.exit(f"refusing to overwrite baseline {dst}")
            Image.fromarray(per_prompt[pid]["first"]).save(dst)

    col = lambda k: [r[k] for r in per_run]
    psnrs = [v["psnr_db"] for v in verify.values() if v.get("psnr_db") is not None]
    res = {
        "exp_id": a.exp,
        "timestamp": started,
        "model": ad.model_id,
        "config": {"resolution": a.res, "steps": a.steps, "seed": a.seed, "prompt_ids": pids, **opts},
        "runs": a.runs,
        "text_encode_s": stats(col("text_encode_s")),
        "text_encode_cached": per_run[0]["text_encode_cached"],
        "denoise_s": stats(col("denoise_s")),
        "vae_s": stats(col("vae_s")),
        "total_s": stats(col("total_s")),
        "total_s_by_prompt": {p: stats([r["total_s"] for r in v["runs"]]) for p, v in per_prompt.items()},
        "first_vs_last_round_total_s": [statistics.median([r["total_s"] for r in per_run if r["round"] == k]) for k in (0, a.runs - 1)],
        "peak_mem_mlx_gb": max(col("mem_peak_mlx_gb")),
        "mem_by_stage_gb": {
            "weights": ad.weights_gb,
            "text_encoder_peak": prep.get("text_encode_peak_gb"),
            "denoise_peak": max(col("mem_denoise_peak_gb")),
            "vae_peak": max(col("mem_vae_peak_gb")),
        },
        "prepare": {k: v for k, v in prep.items() if k != "text_encode_s"} | {"text_encode_s": list(prep.get("text_encode_s", {}).values())},
        "adapter": ad.describe(),
        "warmup_s": warm_s,
        "swap_used_mb_before_after": [swap0, swap1],
        "swap_grew": (swap1 or 0) - (swap0 or 0) > 32,
        "deterministic_across_runs": all(v["deterministic"] for v in per_prompt.values()),
        "memtrace": trace.summary(),
        "ref": a.ref,
        "verify": verify,
        "psnr_db_vs_baseline": min(psnrs) if psnrs else None,
        "max_abs_pixel_diff": max((v.get("max_abs_pixel_diff", 0) for v in verify.values()), default=None) if verify else None,
        "idle_check": idle,
        "mlx_version": __import__("mlx.core").core.__version__,
        **json.loads(a.meta),
    }
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print("[bench] done", flush=True)


if __name__ == "__main__":
    main()
