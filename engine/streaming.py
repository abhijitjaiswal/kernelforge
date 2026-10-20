"""Model-agnostic block-wise weight streaming for MLX modules (Tier E: same weights, same ops).

Each block's parameters live in their own safetensors file. Only the block being computed (plus, with
prefetch, the next one) is resident; after a block runs its parameters are replaced by empty placeholders
so the memory is released.

    s = WeightStreamer(blocks, files, prefetch=True)
    s.attach()          # wraps every block's __call__ with load-before / release-after
    ... run the model eagerly ...
    s.detach(); s.stats  # per-block stall (time waiting for weights) and compute times

Prefetch: right after block i's weights are installed, block i+1's (cyclic, so the next step's first
block is fetched during the last block) load is started with mx.async_eval, so file reads can overlap
GPU compute of block i. The deliverable measurement is how much of the load time this hides.
"""

import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def save_block_files(blocks, paths):
    for b, p in zip(blocks, paths):
        mx.save_safetensors(str(p), dict(tree_flatten(b.parameters())))


def _placeholder(tree):
    return tree_unflatten([(k, mx.zeros((0,), dtype=v.dtype)) for k, v in tree_flatten(tree)])


class WeightStreamer:
    def __init__(self, blocks, files, prefetch=True):
        self.blocks, self.files, self.prefetch = blocks, [str(f) for f in files], prefetch
        self.n = len(blocks)
        self.pending = {}
        self.stats = []
        self._patched = []
        self._empty = [_placeholder(b.parameters()) for b in blocks]

    def release_all(self):
        for i, b in enumerate(self.blocks):
            b.update(self._empty[i])

    def _start_load(self, i):
        params = tree_unflatten(list(mx.load(self.files[i]).items()))
        if self.prefetch:
            mx.async_eval(params)
        self.pending[i] = params

    def _before(self, i):
        t0 = time.perf_counter()
        if i not in self.pending:
            self._start_load(i)
        params = self.pending.pop(i)
        mx.eval(params)                      # stall = time spent waiting for this block's weights
        stall = time.perf_counter() - t0
        self.blocks[i].update(params)
        if self.prefetch:
            self._start_load((i + 1) % self.n)
        return stall

    def _after(self, i, out, stall, t_start):
        mx.eval(out)
        self.blocks[i].update(self._empty[i])
        self.stats.append({"block": i, "stall_s": stall, "block_s": time.perf_counter() - t_start})

    def attach(self):
        streamer = self
        for i, b in enumerate(self.blocks):
            cls = type(b)
            orig = cls.__call__

            def call(self_, *a, _i=i, _orig=orig, **k):
                t = time.perf_counter()
                stall = streamer._before(_i)
                out = _orig(self_, *a, **k)
                streamer._after(_i, out, stall, t)
                return out

            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
            self._patched.append((b, cls))

    def detach(self):
        for b, cls in self._patched:
            b.__class__ = cls
        self._patched = []
        for i in list(self.pending):
            self.pending.pop(i)
