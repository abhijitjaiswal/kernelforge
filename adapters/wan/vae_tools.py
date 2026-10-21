"""Wan 2.1 VAE decoder experiments from frozen latents (decode only, ~1-2 min per run).

    python -m adapters.wan.vae_tools --exp exp-106 --latents runs/exp-105-base-latents/latents.safetensors \
        --impl tiled_auto|untiled|barriers|profile [--latent-frames N] [--ref runs/exp-105-base-latents]

impl
  tiled_auto  stock path used by generate_video (decode_tiled with TilingConfig.auto) -> must reproduce the reference frames
  untiled     stock vae.decode() on the whole latent (what mlx-video does with tiling='none')
  barriers    FROZEN PATTERN eval-barriers: stock modules unchanged, mx.eval after conv1, every middle/upsample layer and the head
  profile     FROZEN TOOL bench/layer_profile.py on the decoder: per-module peak / delta / time (eager per module)
--latent-frames N decodes only the first N latent frames (the decoder is causal in time, so this is an exact prefix) for cheap
structural tests. A swap watchdog (default 8 GB) aborts and records instead of thrashing.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
G = 1024**3


def load_config(model_dir):
    from mlx_video.models.wan_2.config import WanModelConfig
    d = json.loads((model_dir / "config.json").read_text())
    d.pop("quantization", None)
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in WanModelConfig.__dataclass_fields__}
    return WanModelConfig(**d)


def frames_from(video):
    import numpy as np
    v = np.array(video[0])                     # [3, T, H, W] in [-1, 1]
    v = np.clip((v + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return v.transpose(1, 2, 3, 0)             # [T, H, W, 3] (same conversion as generate_video)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--impl", required=True, choices=["tiled_auto", "untiled", "barriers", "profile", "stream", "stream_profile"])
    ap.add_argument("--cache-limit-gb", type=float, default=None, help="mx.set_cache_limit (MLX freed-buffer cache counts in the OS footprint)")
    ap.add_argument("--rows", type=int, default=None, help="stream: height-chunk strip size in output rows (none = temporal streaming only)")
    ap.add_argument("--chunk-latents", type=int, default=1, help="stream: latent frames per streamed chunk")
    ap.add_argument("--direct", action="store_true", help="stream: build each frame window directly from state + new frames (option 1)")
    ap.add_argument("--prefix-ref", help="run dir with untiled prefix frames.npy: check the first N output frames bit for bit")
    ap.add_argument("--model-dir", default="models/wan21_mlx")
    ap.add_argument("--latent-frames", type=int, default=None)
    ap.add_argument("--ref", help="run dir with frames.npy (same latents) to compare against")
    ap.add_argument("--swap-abort-gb", type=float, default=8.0)
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("KF_CHILD") != "1":
        from bench.run import parse_time_l
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "adapters.wan.vae_tools", *sys.argv[1:]],
                                env=dict(os.environ, KF_CHILD="1"), stdout=lf, stderr=tf, cwd=ROOT).returncode
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "notes": "crashed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "impl", "decode_s", "peak_mlx_gb", "verify", "process", "swap_grew", "error")}, default=str)[:1500])
        sys.exit(rc)

    import mlx.core as mx
    import numpy as np

    from bench.memtrace import MemTrace
    from bench.run import swap_used_mb
    from mlx_video.models.wan_2.utils import load_vae_decoder

    swap0 = swap_used_mb()
    res = {"exp_id": a.exp, "kind": "video_vae", "impl": a.impl, "config": {"latents": a.latents, "impl": a.impl, "latent_frames": a.latent_frames}}

    def watchdog():
        while True:
            time.sleep(2)
            s = swap_used_mb() or 0
            if s - (swap0 or 0) > a.swap_abort_gb * 1024:
                res.update(error=f"ABORTED by swap watchdog (+{(s - swap0) / 1024:.1f} GB)", swap_used_mb_before_after=[swap0, s], swap_grew=True)
                (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    if a.cache_limit_gb is not None:
        mx.set_cache_limit(int(a.cache_limit_gb * G))
    trace = MemTrace().start()
    model_dir = ROOT / a.model_dir
    cfg = load_config(model_dir)
    vae = load_vae_decoder(model_dir / "vae.safetensors", cfg)
    z = mx.load(str(ROOT / a.latents))["z"]
    if a.latent_frames:
        z = z[:, :, : a.latent_frames]
    mx.eval(z)
    res["latents_shape"] = list(z.shape)
    res["vae_weights_gb"] = sum(v.nbytes for _, v in __import__("mlx.utils").utils.tree_flatten(vae.parameters())) / G

    def barriers_decode(z):
        mean = vae.mean.reshape(1, -1, 1, 1, 1)
        inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
        x = vae.conv2(z / inv_std + mean)
        mx.eval(x)
        d = vae.decoder
        x = d.conv1(x)
        mx.eval(x)
        for layer in list(d.middle) + list(d.upsamples):
            x = layer(x)
            mx.eval(x)
        x = nn_silu(d.head[0](x))
        mx.eval(x)
        x = d.head[2](x)
        return mx.clip(x, -1, 1)

    from mlx.nn import silu as nn_silu
    trace.mark("decode")
    mx.reset_peak_memory()
    t = time.perf_counter()
    if a.impl == "tiled_auto":
        from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
        T = (z.shape[2] - 1) * 4 + 1
        video = vae.decode_tiled(z, TilingConfig.auto(z.shape[3] * 8, z.shape[4] * 8, T))
    elif a.impl == "untiled":
        video = vae.decode(z)
    elif a.impl == "barriers":
        video = barriers_decode(z)
    elif a.impl == "stream_profile":  # frozen per-module profiler inside the streaming decoder, on chunk 2 (state warm)
        from adapters.wan.vae_stream import StreamingDecoder
        from bench.layer_profile import aggregate, profile
        sd = StreamingDecoder(vae, rows=a.rows, chunk_latents=1, direct=a.direct)
        sd._install()
        try:
            mean = vae.mean.reshape(1, -1, 1, 1, 1)
            inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
            o0 = vae.decode(z[:, :, 0:1])
            mx.eval(o0)
            x = vae.conv2(z[:, :, 1:2] / inv_std + mean)
            mx.eval(x)
            dec, records, e2e = profile(vae.decoder, [x], max_depth=3)
            video = mx.concatenate([o0, mx.clip(dec, -1, 1)], axis=2)
        finally:
            sd._remove()
        res["profile_e2e"] = e2e
        res["profile_top"] = sorted(records, key=lambda r: -r["peak_gb"])[:14]
        res["config"].update(rows=a.rows)
    elif a.impl == "stream":
        from adapters.wan.vae_stream import StreamingDecoder
        chunk_peaks = []

        def on_chunk(t0, o):
            chunk_peaks.append(mx.get_peak_memory() / G)
        video = StreamingDecoder(vae, rows=a.rows, chunk_latents=a.chunk_latents, direct=a.direct).decode(z, on_chunk=on_chunk)
        res["chunk_peaks_gb"] = chunk_peaks
        res["config"].update(rows=a.rows, chunk_latents=a.chunk_latents, direct=a.direct)
    else:
        from bench.layer_profile import aggregate, profile
        mean = vae.mean.reshape(1, -1, 1, 1, 1)
        inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
        x = vae.conv2(z / inv_std + mean)
        mx.eval(x)
        dec, records, e2e = profile(vae.decoder, [x], max_depth=2)
        video = mx.clip(dec, -1, 1)
        res["profile_e2e"] = e2e
        res["profile_top"] = sorted(records, key=lambda r: -r["peak_gb"])[:12]
        res["profile_aggregate"] = aggregate(records)[:15]
    mx.eval(video)
    res["decode_s"] = time.perf_counter() - t
    res["peak_mlx_gb"] = mx.get_peak_memory() / G
    trace.mark("done")
    trace.stop()
    frames = frames_from(video)
    np.save(out / "frames.npy", frames)
    res["frames_shape"] = list(frames.shape)
    if a.ref and (ROOT / a.ref / "frames.npy").exists():
        from bench.video_quality import compare_videos
        ref = np.load(ROOT / a.ref / "frames.npy")
        if a.latent_frames:
            ref = ref[: frames.shape[0]]
        res["verify"] = {a.ref: {k: v for k, v in compare_videos(frames, ref).items() if k != "psnr_db_per_frame"}}
    if a.prefix_ref and (ROOT / a.prefix_ref / "frames.npy").exists():
        pref = np.load(ROOT / a.prefix_ref / "frames.npy")
        n = pref.shape[0]
        same = [bool(np.array_equal(frames[i], pref[i])) for i in range(n)]
        res["prefix_check"] = {"ref": a.prefix_ref, "n_frames": n, "all_identical": all(same), "identical_per_frame": same,
                               "max_abs": int(np.abs(frames[:n].astype(int) - pref.astype(int)).max())}
    swap1 = swap_used_mb()
    res.update(memtrace=trace.summary(), swap_used_mb_before_after=[swap0, swap1], swap_grew=(swap1 or 0) - (swap0 or 0) > 32,
               mlx_version=mx.__version__)
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
