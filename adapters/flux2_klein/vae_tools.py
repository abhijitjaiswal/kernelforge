"""Klein VAE-only tools: load just the VAE, decode saved final latents, profile / compare decoders.

    python -m adapters.flux2_klein.vae_tools profile --latents runs/exp-012-smoke/latents.safetensors --exp exp-014
    python -m adapters.flux2_klein.vae_tools decode  --latents ... --exp ... --impl chunked --ref baselines/bf16_free_te
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.flux2_klein.pipeline import Adapter  # noqa: E402

G = 1024**3


def load_vae(weights="bf16"):
    ad = Adapter({"weights": weights, "lifecycle": "staged"})  # lazy: nothing resident yet
    m = ad.model
    m.text_encoder = None
    m.transformer = None
    gc.collect()
    mx.eval(m.vae.parameters())
    return m.vae


def vae_input(vae, latents, res=1024):
    """Replicates mflux: packed latents -> unpack (bn de-norm, unpatchify) -> post_quant_conv input."""
    lh = lw = res // 16
    packed = latents.reshape(latents.shape[0], lh, lw, latents.shape[-1]).transpose(0, 3, 1, 2)
    lat = vae.unpack_packed_latents(packed)
    lat = (lat / vae.scaling_factor) + vae.shift_factor
    lat = mx.transpose(lat, (0, 2, 3, 1))
    lat = vae.post_quant_conv(lat)
    lat = mx.transpose(lat, (0, 3, 1, 2))
    mx.eval(lat)
    return lat


def to_image(decoded):
    from mflux.utils.image_util import ImageUtil
    return np.array(ImageUtil.to_pil(decoded).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "decode"])
    ap.add_argument("--latents", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--weights", default="bf16")
    ap.add_argument("--impl", default="stock", help="decode: stock | chunked")
    ap.add_argument("--impl-opts", default="{}")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ref")
    ap.add_argument("--prompt", default="p1")
    ap.add_argument("--cache-limit-gb", type=float, default=0.5)
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(int(a.cache_limit_gb * G))

    vae = load_vae(a.weights)
    latents = mx.load(str(ROOT / a.latents))["latents"]
    x = vae_input(vae, latents)
    res = {"exp_id": a.exp, "kind": f"vae_{a.cmd}", "latents": a.latents, "weights": a.weights, "input_shape": list(x.shape),
           "mlx_version": mx.__version__}

    if a.cmd == "profile":
        from bench.layer_profile import profile
        dec, records, e2e = profile(vae.decoder, [x], max_depth=4)
        res["e2e"] = e2e
        res["records"] = records
        top = sorted(records, key=lambda r: -r["delta_gb"])[:15]
        res["top_by_delta"] = top
        img = to_image(dec)
    else:
        from adapters.flux2_klein.vae_chunked import decode_chunked
        opts = json.loads(a.impl_opts)
        fn = (lambda z: vae.decoder(z)) if a.impl == "stock" else (lambda z: decode_chunked(vae.decoder, z, mode=a.impl, **opts))
        mx.eval(fn(x))  # warm-up
        times, peaks = [], []
        for _ in range(a.runs):
            mx.clear_cache()
            mx.reset_peak_memory()
            t = time.perf_counter()
            dec = fn(x)
            mx.eval(dec)
            times.append(time.perf_counter() - t)
            peaks.append(mx.get_peak_memory() / G)
        res.update(impl=a.impl, impl_opts=opts, time_s=times, time_s_median=sorted(times)[len(times) // 2], peak_mlx_gb=max(peaks),
                   weights_gb=sum(v.nbytes for _, v in __import__("mlx.utils").utils.tree_flatten(vae.parameters())) / G)
        img = to_image(dec)
    from PIL import Image
    pid = a.prompt
    Image.fromarray(img).save(out / f"{pid}.png")
    res["config"] = {"prompt_ids": [pid], "weights": a.weights, "cache_limit_gb": a.cache_limit_gb, "vae_impl": a.impl, **json.loads(a.impl_opts)}
    res["ref"] = a.ref
    if a.ref:
        from bench.verify import compare
        res["verify"] = {pid: compare(img, np.asarray(Image.open(ROOT / a.ref / f"{pid}.png").convert("RGB")), str(out / f"{pid}_diff.png"))}
        res["psnr_db_vs_baseline"] = res["verify"][pid]["psnr_db"]
        res["max_abs_pixel_diff"] = res["verify"][pid]["max_abs_pixel_diff"]
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    summ = {k: res.get(k) for k in ("exp_id", "e2e", "time_s_median", "peak_mlx_gb", "psnr_db_vs_baseline")}
    print(json.dumps(summ, default=str))
    if a.cmd == "profile":
        for r in res["top_by_delta"]:
            print(f"{r['module']:40s} {r['type']:28s} delta {r['delta_gb']:.2f} GB  peak {r['peak_gb']:.2f}  out {r['out_gb']:.3f} {r['out_shape']}  {r['time_s']*1000:.0f} ms")


if __name__ == "__main__":
    main()
