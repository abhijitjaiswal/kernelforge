"""Weight-streaming prototype for the Klein transformer (engine/streaming.py), 2-block residency.

    python -m adapters.flux2_klein.stream_bench --prep                       # once: write per-block bf16 files
    python -m adapters.flux2_klein.stream_bench --exp exp-051 --mode resident|sync|prefetch [--runs 2]
    python -m adapters.flux2_klein.stream_bench --exp exp-050 --disk         # block-file read speed

All modes run the transformer eagerly (streaming hooks cannot live inside mx.compile), so `resident`
(all weights loaded, eager) is the like-for-like reference for speed and must match streaming bit for bit.
Per denoise step we record the step time and, for streaming, each block's stall (time spent waiting for
its weights) and compute time. Decode uses the exact chunked VAE (Tier E).
"""

import argparse
import fcntl
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BLOCK_DIR = ROOT / "models" / "klein-blocks-bf16"
G = 1024**3


def block_list(tr):
    return list(tr.transformer_blocks) + list(tr.single_transformer_blocks)


def block_files(n):
    return [BLOCK_DIR / f"{i:03d}.safetensors" for i in range(n)]


def purge():
    subprocess.run(["sudo", "-n", "/usr/sbin/purge"], check=True)


def disk_speed(files, cold=False):
    """Read every block file: F_NOCACHE then normal. cold=True purges the page cache before each pass."""
    res = {}
    total = sum(f.stat().st_size for f in files)
    for label, nocache in (("nocache", True), ("cached", False)):
        if cold:
            purge()
            label = "purged_" + label
        t = time.perf_counter()
        for f in files:
            fd = os.open(f, os.O_RDONLY)
            if nocache:
                fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
            while os.read(fd, 64 << 20):
                pass
            os.close(fd)
        dt = time.perf_counter() - t
        res[label] = {"gb": total / G, "s": dt, "gbps": total / G / dt}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--mode", choices=["resident", "barrier", "sync", "prefetch", "resident_compiled", "resident_blockc", "prefetch_blockc"])
    ap.add_argument("--lag", type=int, default=0, help="prefetch_blockc: blocks kept in flight before the host waits")
    ap.add_argument("--te-stream", action="store_true", help="stream the text encoder too (te_probe.setup_te_streaming)")
    ap.add_argument("--purge-steps", action="store_true", help="purge the page cache before every denoise step (untimed): cold-SSD reads")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--disk", action="store_true")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--prompt", default="p1")
    ap.add_argument("--ref", help="eager-resident reference image dir (bit-identity check)")
    a = ap.parse_args()

    from adapters.flux2_klein.pipeline import DEFAULTS, Adapter

    if a.prep:
        BLOCK_DIR.mkdir(parents=True, exist_ok=True)
        ad = Adapter({"weights": "bf16", "lifecycle": "staged"})
        ad.model.text_encoder = None
        blocks = block_list(ad.model.transformer)
        from engine.streaming import _placeholder, save_block_files
        for i, (b, f) in enumerate(zip(blocks, block_files(len(blocks)))):
            save_block_files([b], [f])
            b.update(_placeholder(b.parameters()))  # release this block before reading the next
            print(f"[prep] {f.name} {f.stat().st_size / G:.3f} GB", flush=True)
        return

    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    if a.disk:
        files = sorted(BLOCK_DIR.glob("*.safetensors"))
        r = {"exp_id": a.exp, "kind": "disk", "disk": {**disk_speed(files), **disk_speed(files, cold=True)}, "n_files": len(files),
             "block_gb_mean": sum(f.stat().st_size for f in files) / len(files) / G}
        (out / "result.json").write_text(json.dumps(r, indent=2))
        print(json.dumps(r, indent=1))
        return

    if os.environ.get("KF_CHILD") != "1":  # re-exec under /usr/bin/time -l for the OS peak footprint
        from bench.run import parse_time_l
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "adapters.flux2_klein.stream_bench", *sys.argv[1:]],
                                env=dict(os.environ, KF_CHILD="1"), stdout=lf, stderr=tf, cwd=ROOT).returncode
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "notes": "crashed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "step_s_median", "stall_per_step_s", "peak_mlx_gb", "process", "verify", "returncode")}, default=str))
        sys.exit(rc)

    from bench.memtrace import MemTrace
    from bench.power import Power
    from bench.run import swap_used_mb
    from engine.streaming import BlockCompiler, WeightStreamer
    from mflux.models.common.config.config import Config

    mx.set_cache_limit(int(0.5 * G))
    swap0 = swap_used_mb()
    power = Power(out / "power.txt").start()
    trace = MemTrace().start()
    prompt = json.loads((ROOT / "bench" / "prompts.json").read_text())[a.prompt]
    ad = Adapter({"weights": "bf16", "lifecycle": "staged", "vae_impl": "chunked", "vae_rows": 64})
    m = ad.model
    te_streamer = None
    if a.te_stream:
        from adapters.flux2_klein.te_probe import setup_te_streaming
        te_streamer = setup_te_streaming(m.text_encoder)
    trace.mark("encode")
    t_enc = time.perf_counter()
    emb, tid, _, _ = ad._orig_encode(prompt=prompt, negative_prompt=" ", guidance=DEFAULTS["guidance"])
    mx.eval(emb, tid)
    encode_s = time.perf_counter() - t_enc  # includes lazy text-encoder load (cold process, as a user would run it)
    if te_streamer:
        te_streamer.detach()
        te_streamer.pending.clear()
    m.text_encoder = None
    gc.collect()
    mx.clear_cache()

    tr = m.transformer
    blocks = block_list(tr)
    streamer = None
    trace.mark("load_transformer")
    compiled_predict = None
    if a.mode == "resident_compiled":  # mflux's whole-transformer mx.compile (speed reference)
        mx.eval(tr.parameters())
        compiled_predict = m._predict(tr)
    elif a.mode == "resident_blockc":  # resident, one compiled function per block, no barriers (numerics reference)
        mx.eval(tr.parameters())
        BlockCompiler(blocks).attach()
    elif a.mode == "prefetch_blockc":
        streamer = WeightStreamer(blocks, block_files(len(blocks)), prefetch=True, compile=True, lag=a.lag)
        streamer.release_all()
        mx.eval(tr.parameters())
        streamer.attach()
    elif a.mode in ("resident", "barrier"):
        mx.eval(tr.parameters())
        if a.mode == "barrier":  # resident weights, but mx.eval after every block: isolates the barrier cost
            for b in blocks:
                cls = type(b)

                def call(self_, *args, _orig=cls.__call__, **kw):
                    o = _orig(self_, *args, **kw)
                    mx.eval(o)
                    return o

                b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
    else:
        streamer = WeightStreamer(blocks, block_files(len(blocks)), prefetch=(a.mode == "prefetch"))
        streamer.release_all()
        mx.eval(tr.parameters())  # only the non-block parameters remain
        streamer.attach()
    resident_gb = mx.get_active_memory() / G

    cfg_of = lambda: Config(model_config=m.model_config, num_inference_steps=4, height=1024, width=1024, guidance=DEFAULTS["guidance"],
                            image_path=None, image_strength=None, scheduler="flow_match_euler_discrete")
    runs = []
    lat = lh = lw = None
    mx.reset_peak_memory()
    for r in range(a.runs + 1):  # run 0 = warm-up
        trace.mark(f"denoise_run{r}")
        cfg = cfg_of()
        lat, lid, lh, lw = m._prepare_generation_latents(seed=42, config=cfg)
        mx.eval(lat)
        steps = []
        n0 = len(streamer.stats) if streamer else 0
        for s in range(4):
            if a.purge_steps:
                purge()
                time.sleep(0.5)
            t = time.perf_counter()
            if compiled_predict is not None:
                noise = compiled_predict(latents=lat, latent_ids=lid, prompt_embeds=emb, text_ids=tid, negative_prompt_embeds=None,
                                         negative_text_ids=None, guidance=DEFAULTS["guidance"], timestep=cfg.scheduler.timesteps[s])
            else:
                noise = tr(hidden_states=lat, encoder_hidden_states=emb, timestep=cfg.scheduler.timesteps[s], img_ids=lid, txt_ids=tid, guidance=None)
            lat = cfg.scheduler.step(noise=noise, timestep=s, latents=lat, sigmas=cfg.scheduler.sigmas)
            mx.eval(lat)
            if streamer:
                streamer.flush()
            steps.append(time.perf_counter() - t)
        st = streamer.stats[n0:] if streamer else []
        runs.append({"run": r, "step_s": steps, "total_s": sum(steps),
                     "stall_s": sum(x["stall_s"] for x in st), "block_s": sum(x["block_s"] for x in st)})
        print(f"[stream] {a.mode} run{r}: steps {[round(x, 3) for x in steps]} stall {runs[-1]['stall_s']:.2f}s", flush=True)
    denoise_peak = mx.get_peak_memory() / G
    if streamer:
        streamer.detach()
    trace.mark("decode")
    compiled_predict = None
    t_dec = time.perf_counter()
    m.transformer = None
    gc.collect()
    mx.clear_cache()
    from mflux.utils.image_util import ImageUtil
    packed = lat.reshape(lat.shape[0], lh, lw, lat.shape[-1]).transpose(0, 3, 1, 2)
    decoded = ad._orig_decode(packed)
    mx.eval(decoded)
    img = np.array(ImageUtil.to_pil(decoded).convert("RGB"))
    decode_s = time.perf_counter() - t_dec
    trace.mark("done")
    trace.stop()
    trace.dump(out / "memtrace.json")
    power.stop()
    psum = power.summary()
    psum.pop("series", None)
    from PIL import Image
    Image.fromarray(img).save(out / f"{a.prompt}.png")
    verify = {}
    from bench.verify import compare
    for ref in [x for x in (a.ref, "baselines/bf16_free_te") if x]:
        rp = ROOT / ref / f"{a.prompt}.png"
        if rp.exists():
            verify[ref] = compare(img, np.asarray(Image.open(rp).convert("RGB")))
    timed = runs[1:]
    swap1 = swap_used_mb()
    res = {"exp_id": a.exp, "kind": "streaming", "mode": a.mode, "purge_steps": a.purge_steps,
           "config": {"weights": "bf16", "prompt_ids": [a.prompt], "stream": a.mode + ("-cold" if a.purge_steps else "") + (f"-lag{a.lag}" if a.lag else ""), "te_stream": a.te_stream, "vae_impl": "chunked", "cache_limit_gb": 0.5},
           "runs": runs, "step_s_median": statistics.median([s for r in timed for s in r["step_s"]]),
           "denoise_s_median": statistics.median([r["total_s"] for r in timed]),
           "encode_s": encode_s, "decode_s": decode_s,
           "e2e_image_s": encode_s + statistics.median([r["total_s"] for r in timed]) + decode_s,
           "stall_per_step_s": statistics.median([r["stall_s"] for r in timed]) / 4 if streamer else 0.0,
           "resident_after_load_gb": resident_gb, "peak_mlx_gb": denoise_peak, "memtrace": trace.summary(), "power": psum,
           "verify": verify, "swap_used_mb_before_after": [swap0, swap1], "swap_grew": (swap1 or 0) - (swap0 or 0) > 32,
           "mlx_version": mx.__version__}
    if streamer:
        bs = [x for x in streamer.stats]
        res["per_block_stall_s_median"] = statistics.median(x["stall_s"] for x in bs)
        res["per_block_s_median"] = statistics.median(x["block_s"] for x in bs)
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
