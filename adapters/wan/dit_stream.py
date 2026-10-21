"""Wan DiT block streaming via the frozen engine (engine/streaming.py), adapter-level only.

mlx-video calls blocks as block(x, cross_kv_cache=..., **kwargs) with x positional; the frozen engine's compiled path takes
keyword arguments only, so these subclasses fold positional args into keywords using the block's __call__ signature.
Streaming starts at the first denoising step (after mlx-video's cross-attention K/V precompute, which reads every block once).

    python -m adapters.wan.dit_stream prep     # per-block bf16 files -> models/wan21-dit-blocks/
"""

import inspect
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.streaming import BlockCompiler, WeightStreamer, _CompiledCall  # noqa: E402

BLOCK_DIR = ROOT / "models" / "wan21-dit-blocks"


def block_files(n):
    return [BLOCK_DIR / f"block{i:02d}.safetensors" for i in range(n)]


def _fold(cls, args, kw):
    if not args:
        return kw
    names = [p for p in inspect.signature(cls.__call__).parameters if p != "self"]
    return {**dict(zip(names, args)), **kw}


class PosBlockCompiler(BlockCompiler):
    def attach(self):
        for i, b in enumerate(self.blocks):
            cls = type(b)
            cc = _CompiledCall(b, cls.__call__)
            outer = self

            def call(self_, *a, _i=i, _cc=cc, _cls=cls, **k):
                out = _cc(outer.params[_i], _fold(_cls, a, k))
                if outer.barrier:
                    mx.eval(out)
                return out
            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
            self._patched.append((b, cls))


class PosWeightStreamer(WeightStreamer):
    def attach(self):
        streamer = self
        for i, b in enumerate(self.blocks):
            cls = type(b)
            orig = cls.__call__
            cc = _CompiledCall(b, orig) if self.compile else None

            def call(self_, *a, _i=i, _orig=orig, _cc=cc, _cls=cls, **k):
                t = time.perf_counter()
                stall, params = streamer._before(_i)
                out = _cc(params, _fold(_cls, a, k)) if _cc is not None else _orig(self_, *a, **k)
                streamer._after(_i, out, stall, t, params)
                return out
            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
            self._patched.append((b, cls))


def prep(model_dir=ROOT / "models" / "wan21_mlx"):
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    w = mx.load(str(model_dir / "model.safetensors"))
    n = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("blocks."))
    for i in range(n):
        p = f"blocks.{i}."
        mx.save_safetensors(str(BLOCK_DIR / f"block{i:02d}.safetensors"), {k[len(p):]: v for k, v in w.items() if k.startswith(p)})
    print(f"[prep] {n} DiT block files -> {BLOCK_DIR}")


if __name__ == "__main__" and sys.argv[1:] == ["prep"]:
    prep()
