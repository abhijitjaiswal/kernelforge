"""Stock mlx-video's initial noise for a seed without the 21 GB fp32 encoder load: replay its RNG-consuming steps in order
(seed -> T5Encoder constructor -> load_wan_model -> normal(target_shape)). load_weights and forward passes draw no random numbers.
    python bench/rng_stock_noise.py <out.safetensors>"""
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adapters.wan.vae_tools import load_config  # noqa: E402
from mlx_video.models.wan_2.text_encoder import T5Encoder  # noqa: E402
from mlx_video.models.wan_2.utils import load_wan_model  # noqa: E402

md = ROOT / "models" / "wan21_mlx"
cfg = load_config(md)
mx.random.seed(42)
T5Encoder(vocab_size=cfg.t5_vocab_size, dim=cfg.t5_dim, dim_attn=cfg.t5_dim_attn, dim_ffn=cfg.t5_dim_ffn, num_heads=cfg.t5_num_heads,
          num_layers=cfg.t5_num_layers, num_buckets=cfg.t5_num_buckets, shared_pos=False)
load_wan_model(md / "model.safetensors", cfg, None, loras=None)
mx.save_safetensors(sys.argv[1], {"noise": mx.random.normal((16, 9, 60, 104))})
