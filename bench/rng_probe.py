"""Does a loader consume the global MLX RNG differently from stock? Seed, construct, then draw.  python bench/rng_probe.py"""
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adapters.wan.vae_tools import load_config  # noqa: E402
from mlx_video.models.wan_2.text_encoder import T5Encoder  # noqa: E402

cfg = load_config(ROOT / "models" / "wan21_mlx")


def draw(tag, build):
    mx.random.seed(42)
    build()
    v = mx.random.normal((4,))
    print(f"{tag:28s}", [round(x, 6) for x in v.tolist()], flush=True)


def stock_ctor():  # stock load_t5_encoder minus the 21 GB fp32 load_weights (which does not touch the RNG)
    T5Encoder(vocab_size=cfg.t5_vocab_size, dim=cfg.t5_dim, dim_attn=cfg.t5_dim_attn, dim_ffn=cfg.t5_dim_ffn,
              num_heads=cfg.t5_num_heads, num_layers=cfg.t5_num_layers, num_buckets=cfg.t5_num_buckets, shared_pos=False)


def streamed():
    from adapters.wan.te_stream import load_streamed_t5
    load_streamed_t5(None, cfg)


draw("nothing", lambda: None)
draw("stock T5 constructor", stock_ctor)
draw("streamed T5 loader", streamed)
