"""Exact streamed umT5-XXL encoder for mlx-video (same construction verified bit-identical in exp-102).

load_streamed_t5(model_path, config) is a drop-in for mlx_video...utils.load_t5_encoder: 24 blocks streamed from per-block
bf16 files with fp32 upcast at load (pattern upcast-on-load-streaming), 256k-row embedding gathered from file by token id.
Requires `python -m adapters.wan.te_tools prep` (models/wan21-te-blocks/).
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.rowgather import gather_rows  # noqa: E402
from engine.streaming import WeightStreamer  # noqa: E402

BLOCK_DIR = ROOT / "models" / "wan21-te-blocks"


class UpcastStreamer(WeightStreamer):
    def _start_load(self, i):
        params = tree_unflatten([(k, v.astype(mx.float32)) for k, v in mx.load(self.files[i]).items()])
        if self.prefetch:
            mx.async_eval(params)
        self.pending[i] = params


def load_streamed_t5(model_path, config):
    from mlx_video.models.wan_2.text_encoder import T5Encoder
    enc = T5Encoder(vocab_size=config.t5_vocab_size, dim=config.t5_dim, dim_attn=config.t5_dim_attn, dim_ffn=config.t5_dim_ffn,
                    num_heads=config.t5_num_heads, num_layers=config.t5_num_layers, num_buckets=config.t5_num_buckets, shared_pos=False)
    rest = {k: v.astype(mx.float32) for k, v in mx.load(str(BLOCK_DIR / "rest.safetensors")).items()}
    enc.load_weights(list(rest.items()), strict=False)
    st = UpcastStreamer(list(enc.blocks), [BLOCK_DIR / f"block{i:02d}.safetensors" for i in range(len(enc.blocks))], prefetch=True)
    st.release_all()
    ef = str(BLOCK_DIR / "embed.safetensors")

    class RowEmbedding:
        def __call__(self, ids):
            return gather_rows(ef, "weight", np.array(ids)).astype(mx.float32)
    enc.token_embedding = RowEmbedding()
    mx.eval(enc.norm.parameters())
    st.attach()
    enc._kf_streamer = st
    return enc
