"""Isolated test: is a row-chunked 'same' Conv2d bit-identical to the full conv at a given shape, and does it depend on
the framework's algorithm selection? Model-agnostic; shapes come from the caller.

    python bench/conv_determinism.py --wan-vae models/wan21_mlx --frames 4 --rows 16,32,48,64,96,120,160,240 --out runs/exp-114/<tag>.json
Run once per environment setting (e.g. MLX_CONV_WINOGRAD unset / 0 / 1) in separate processes.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-vae", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--rows", default="12,16,24,48,96,120,240")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    from engine.chunked import conv_rows
    from adapters.wan.vae_tools import load_config
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.vae import Resample

    md = ROOT / a.wan_vae
    vae = load_vae_decoder(md / "vae.safetensors", load_config(md))
    ups = [m for _, m in vae.named_modules() if isinstance(m, Resample) and m.mode.startswith("upsample")]
    mx.random.seed(0)
    h, w, t = 60, 104, 1                                # latent 60x104; one latent frame per streamed chunk
    rows_list = [int(r) for r in a.rows.split(",")]
    results = []
    for i, rs in enumerate(ups):
        if rs.mode == "upsample3d":
            t *= 2
        conv = rs.resample[1]
        cin = conv.weight.shape[-1]
        H, W = h * 2, w * 2
        x = mx.random.normal((t, H, W, cin)).astype(mx.float32)   # the conv input after nearest-2x (values: random)
        mx.eval(x)
        full = conv(x)
        mx.eval(full)
        row = {"upsampler": i, "mode": rs.mode, "input_shape": [t, H, W, cin], "out_channels": conv.weight.shape[0], "strips": {}}
        for r in rows_list:
            if r >= H:
                continue
            y = conv_rows(conv, x, r)
            mx.eval(y)
            d = mx.abs(y - full)
            row["strips"][r] = {"identical": bool(mx.array_equal(y, full).item()), "max_abs": d.max().item()}
            # grid-aligned strips: pad rows once (as the full conv does internally), cut strips starting at multiples of r,
            # conv with zero row padding so each strip's tiles sit exactly where the full conv's tiles sit
            p_ = conv.weight.shape[1] // 2
            xp = mx.pad(x, [(0, 0), (p_, p_), (0, 0), (0, 0)])
            outs = []
            for r0 in range(0, H, r):
                r1 = min(r0 + r, H)
                o = mx.conv2d(xp[:, r0: r1 + 2 * p_], conv.weight, padding=(0, p_)) + conv.bias
                mx.eval(o)
                outs.append(o)
            y2 = mx.concatenate(outs, axis=1)
            row["strips"][f"aligned{r}"] = {"identical": bool(mx.array_equal(y2, full).item()), "max_abs": mx.abs(y2 - full).max().item()}
        results.append(row)
        h, w = H, W
    env = {k: v for k, v in os.environ.items() if k.startswith("MLX_CONV")}
    out = {"env": env, "mlx_version": mx.__version__, "results": results}
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    for r in results:
        print(r["upsampler"], r["mode"], r["input_shape"], {k: ("ID" if v["identical"] else f"{v['max_abs']:.2e}") for k, v in r["strips"].items()})


if __name__ == "__main__":
    main()
