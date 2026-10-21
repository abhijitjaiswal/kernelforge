"""Wan (mlx-video) adapter: Phase-0-style instrumented run of mlx-video's own generate_video().

    python -m adapters.wan.bench_wan --exp exp-100 --model-dir models/wan21_mlx --prompt v1 \
        --width 832 --height 480 --frames 33 --steps 50 --seed 42 [--tiling auto|none] [--no-compile] [--ref runs/exp-100]

Nothing in mlx-video is re-implemented. Its module-level functions are wrapped to timestamp and measure each stage:
  text encoder load -> encode -> DiT load -> denoise (per step, via the loop's tqdm iterator) -> VAE load -> VAE decode
Per stage: wall time, MLX peak (reset at each stage start), resident weight bytes, and so the activation share =
stage peak - resident weights. Process OS peak via /usr/bin/time -l, memory timeline (bench/memtrace.py),
powermetrics (bench/power.py), swap delta. The decoded frames are saved as frames.npy (uint8 [T,H,W,3]) for quality checks.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))  # tokenizer cache stays inside the project
G = 1024**3


class _StopDecode(Exception):
    pass


def gc_collect():
    import gc
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()


def nbytes(module):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model-dir", default="models/wan21_mlx")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiling", default="auto")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--cache-limit-gb", type=float, default=None)
    ap.add_argument("--ref", help="run dir with frames.npy to compare against")
    ap.add_argument("--context-ref", help="frozen text-context safetensors (context, context_null): skip the text encoder, feed stock embeddings")
    ap.add_argument("--te-stream", action="store_true", help="exact streamed text encoder (adapters/wan/te_stream.py)")
    ap.add_argument("--dit", default="stock", choices=["stock", "blockc", "stream"], help="stock: mlx-video whole-model compile; blockc: per-block compile, resident; stream: per-block compile + streaming (prefetch, lag 1)")
    ap.add_argument("--vae-stream", action="store_true", help="exact streaming decoder v6 (rows 96, direct, barriers)")
    ap.add_argument("--cache-schedule", default=None, help="combined step+CFG reuse schedule (adapters/wan/step_cache.py), e.g. a2")
    ap.add_argument("--save-noise", action="store_true", help="save the initial latent noise (RNG-parity check)")
    ap.add_argument("--stop-after-denoise", action="store_true", help="save latents and skip the VAE decode")
    ap.add_argument("--swap-abort-gb", type=float, default=8.0, help="watchdog: abort (recording the stage) if swap grows by more than this")
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)

    if os.environ.get("KF_CHILD") != "1":  # re-exec under /usr/bin/time -l for the OS peak footprint
        from bench.run import parse_time_l
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "adapters.wan.bench_wan", *sys.argv[1:]],
                                env=dict(os.environ, KF_CHILD="1"), stdout=lf, stderr=tf, cwd=ROOT).returncode
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "kind": "video", "notes": "crashed or killed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "total_s", "stages", "process", "swap_grew", "returncode")}, default=str))
        sys.exit(rc)

    import mlx.core as mx
    import numpy as np

    from bench.memtrace import MemTrace
    from bench.power import Power, thermal_now
    from bench.run import swap_used_mb
    import mlx_video.models.wan_2.generate as gen

    if a.cache_limit_gb is not None:
        mx.set_cache_limit(int(a.cache_limit_gb * G))
    prompt = json.loads((ROOT / "bench" / "video_prompts.json").read_text())[a.prompt]
    swap0, thermal0 = swap_used_mb(), thermal_now()
    power = Power(out / "power.txt").start()
    trace = MemTrace().start()
    T = {"t0": time.perf_counter()}
    stages, order = {}, []
    weights = {}
    cur = {"name": None, "t": None}

    def begin(name):
        end()
        trace.mark(name)
        mx.reset_peak_memory()
        cur.update(name=name, t=time.perf_counter(), active0=mx.get_active_memory())

    def end():
        if cur["name"] is None:
            return
        stages[cur["name"]] = {"s": time.perf_counter() - cur["t"], "peak_mlx_gb": mx.get_peak_memory() / G,
                               "active_start_gb": cur["active0"] / G, "active_end_gb": mx.get_active_memory() / G}
        order.append(cur["name"])
        cur["name"] = None

    def wrap(fn_name, stage, after=None):
        orig = getattr(gen, fn_name)

        def w(*args, **kw):
            begin(stage)
            r = orig(*args, **kw)
            if after:
                after(r)
            return r
        setattr(gen, fn_name, w)

    def after_t5(m):
        mx.eval(m.parameters())
        weights["text_encoder"] = nbytes(m) / G

    dit_state = {}

    def after_dit(m):
        weights.setdefault("dit", 0)
        weights["dit"] += nbytes(m) / G
        dit_state["model"] = m

    def after_vae(v):
        weights["vae"] = nbytes(v) / G
        if a.vae_stream:
            from adapters.wan.vae_stream import StreamingDecoder
            sd = StreamingDecoder(v, rows=96, chunk_latents=1, direct=True, barriers=True)
            v.decode_tiled = lambda z, tiling_config=None, _sd=sd: _sd.decode(z)
            v.decode = lambda z, _sd=sd: _sd.decode(z)
        for meth in ("decode", "decode_tiled"):
            if hasattr(v, meth):
                orig = getattr(v, meth)

                def d(*args, _o=orig, **kw):
                    if args and isinstance(args[0], mx.array):  # decoder input = final latents (frozen-reference capture)
                        mx.save_safetensors(str(out / "latents.safetensors"), {"z": args[0]})
                    if a.stop_after_denoise:
                        end()
                        raise _StopDecode()
                    begin("vae_decode")
                    r = _o(*args, **kw)
                    mx.eval(r)
                    return r
                setattr(v, meth, d)

    if a.save_noise:  # record the initial latent noise (the first 4-D mx.random.normal draw) for RNG-parity checks
        _normal = mx.random.normal

        def normal_w(shape, *args, **kw):
            r = _normal(shape, *args, **kw)
            if len(shape) == 4 and not (out / "noise.safetensors").exists():
                mx.save_safetensors(str(out / "noise.safetensors"), {"noise": r})
            return r
        mx.random.normal = normal_w
    if a.te_stream:
        from adapters.wan.te_stream import load_streamed_t5
        gen.load_t5_encoder = load_streamed_t5
    if a.context_ref:  # stock embeddings from the frozen capture; everything downstream is unmodified mlx-video
        ref_ctx = mx.load(str(ROOT / a.context_ref))
        queue = [ref_ctx["context"], ref_ctx["context_null"]]
        gen.load_t5_encoder = lambda *args, **kw: object()
        gen.encode_text = lambda *args, **kw: queue.pop(0)
    else:
        wrap("load_t5_encoder", "text_encoder_load", after_t5)
    orig_encode = gen.encode_text

    def encode_w(*args, **kw):
        if cur["name"] != "text_encode":
            begin("text_encode")
        r = orig_encode(*args, **kw)
        mx.eval(r)
        enc_outs.append(r)
        if len(enc_outs) == 2:  # prompt + negative prompt, as consumed by the DiT (checked against the frozen capture)
            mx.save_safetensors(str(out / "text_context.safetensors"), {"context": enc_outs[0], "context_null": enc_outs[1]})
        return r
    enc_outs = []
    gen.encode_text = encode_w
    wrap("load_wan_model", "dit_load", after_dit)
    wrap("load_vae_decoder", "vae_load", after_vae)

    step_times = []
    orig_tqdm = gen.tqdm

    def tqdm_w(it, *args, **kw):  # the denoising loop iterates tqdm(range(steps)); each next() = one step boundary
        if kw.get("desc") != "Diffusion":
            return orig_tqdm(it, *args, **kw)

        def gen_():
            m = dit_state.get("model")
            if a.dit != "stock" and m is not None:   # after mlx-video's cross-attention K/V precompute (reads every block once)
                from adapters.wan.dit_stream import PosBlockCompiler, PosWeightStreamer, block_files
                blocks = list(m.blocks)
                if a.dit == "blockc":
                    dit_state["h"] = PosBlockCompiler(blocks)
                else:
                    dit_state["h"] = PosWeightStreamer(blocks, block_files(len(blocks)), prefetch=True, compile=True, lag=1)
                    dit_state["h"].release_all()
                    gc_collect()
                dit_state["h"].attach()
            if a.cache_schedule and m is not None:
                from adapters.wan.step_cache import SCHEDULES, StepCache
                modes = SCHEDULES[a.cache_schedule](a.steps)
                print("[cache]", a.cache_schedule, {k: modes.count(k) for k in ("full", "skip", "cond")}, flush=True)
                dit_state["cache"] = StepCache(m, modes)
                dit_state["cache"].attach()
            begin("denoise")
            last = time.perf_counter()
            for k, x in enumerate(it):
                yield x
                h = dit_state.get("h")
                if h is not None and hasattr(h, "flush"):
                    h.flush()
                now = time.perf_counter()
                step_times.append(now - last)
                print(f"[wan] step {k + 1}/{a.steps} {now - last:.1f}s", flush=True)
                last = now
        return gen_()
    gen.tqdm = tqdm_w

    frames = {}
    orig_save = gen.save_video

    def save_w(video, path, fps=16):
        end()
        frames["v"] = video
        return orig_save(video, path, fps=fps)
    gen.save_video = save_w

    import threading

    def watchdog():  # protects m5-test: a run that drives swap up by > swap_abort_gb is stopped and recorded, not allowed to thrash
        while True:
            time.sleep(2)
            s = swap_used_mb() or 0
            if s - (swap0 or 0) > a.swap_abort_gb * 1024:
                partial = {"exp_id": a.exp, "kind": "video", "error": f"ABORTED by swap watchdog: swap +{(s - swap0) / 1024:.1f} GB during stage {cur['name']}",
                           "aborted_stage": cur["name"], "stages_completed": stages, "weights_gb": weights, "step_s": step_times,
                           "swap_used_mb_before_after": [swap0, s], "swap_grew": True,
                           "config": {"prompt_ids": [a.prompt], "width": a.width, "height": a.height, "frames": a.frames, "steps": a.steps,
                                      "seed": a.seed, "tiling": a.tiling, "compile": not a.no_compile}}
                (out / "result.json").write_text(json.dumps(partial, indent=2, default=str))
                power.stop()
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    t_start = time.perf_counter()
    err = None
    try:
        gen.generate_video(model_dir=str(ROOT / a.model_dir), prompt=prompt, width=a.width, height=a.height, num_frames=a.frames,
                           steps=a.steps, seed=a.seed, output_path=str(out / "video.mp4"), tiling=a.tiling,
                           no_compile=a.no_compile or a.dit != "stock")
    except _StopDecode:
        pass
    except Exception as e:  # noqa: BLE001  record OOM / failures instead of losing the measurements
        err = f"{type(e).__name__}: {e}"
        end()
    total = time.perf_counter() - t_start
    trace.mark("done")
    trace.stop()
    trace.dump(out / "memtrace.json")
    power.stop()
    psum = power.summary()
    (out / "power.json").write_text(json.dumps(psum))
    psum.pop("series", None)
    psum["thermal_at_start"] = thermal0
    swap1 = swap_used_mb()

    res_w = {"text_encode": "text_encoder", "text_encoder_load": "text_encoder", "denoise": "dit", "dit_load": "dit", "vae_decode": "vae", "vae_load": "vae"}
    for k, v in stages.items():
        w = weights.get(res_w.get(k, ""), 0.0)
        v["resident_weights_gb"] = w
        v["activation_gb"] = max(v["peak_mlx_gb"] - max(w, v["active_start_gb"] if k == "denoise" else w), 0.0)
    verify = {}
    if "v" in frames:
        np.save(out / "frames.npy", frames["v"])
        if a.ref and (ROOT / a.ref / "frames.npy").exists():
            from bench.video_quality import compare_videos
            verify[a.ref] = compare_videos(frames["v"], np.load(ROOT / a.ref / "frames.npy"))
    res = {"exp_id": a.exp, "kind": "video", "model": Path(a.model_dir).name,
           "config": {"prompt_ids": [a.prompt], "width": a.width, "height": a.height, "frames": a.frames, "steps": a.steps,
                      "seed": a.seed, "tiling": a.tiling, "compile": not a.no_compile, "cache_limit_gb": a.cache_limit_gb,
                      "te_stream": a.te_stream, "dit": a.dit, "cache_schedule": a.cache_schedule, "vae_stream": a.vae_stream, "stop_after_denoise": a.stop_after_denoise},
           "error": err, "total_s": {"median": total} if not err else None, "stage_order": order, "stages": stages,
           "weights_gb": weights, "step_s": step_times,
           "step_s_median": sorted(step_times)[len(step_times) // 2] if step_times else None,
           "frames_shape": list(frames["v"].shape) if "v" in frames else None,
           "memtrace": trace.summary(), "power": psum, "swap_used_mb_before_after": [swap0, swap1],
           "swap_grew": (swap1 or 0) - (swap0 or 0) > 32, "ref": a.ref, "verify": verify, "mlx_version": mx.__version__}
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("error", "total_s", "weights_gb", "step_s_median")}, default=str))


if __name__ == "__main__":
    main()
