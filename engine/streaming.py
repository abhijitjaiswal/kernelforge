"""Model-agnostic block-wise weight streaming for MLX modules (Tier E: same weights, same ops).

Each block's parameters live in their own safetensors file. Only the block being computed (plus, with
prefetch, the next one) is resident; after a block runs its parameters are replaced by empty placeholders
so the memory is released.

    s = WeightStreamer(blocks, files, prefetch=True)
    s.attach()          # wraps every block's __call__ with load-before / release-after
    ... run the model eagerly ...
    s.detach(); s.stats  # per-block stall (time waiting for weights) and compute times

compile=True: every block gets its own mx.compile'd function whose weights are explicit inputs
(fn(params, array_kwargs)), so swapping in streamed weights never retraces and each block keeps
compile's fusion. Use BlockCompiler for the same per-block compilation with resident weights.

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


def _is_array_tree(v):
    if isinstance(v, mx.array):
        return True
    if isinstance(v, (tuple, list)) and v:
        return all(_is_array_tree(x) for x in v)
    return False


class _CompiledCall:
    """mx.compile'd block call with weights and array kwargs as inputs; non-array kwargs are baked in."""

    def __init__(self, block, orig_call):
        self.block, self.orig, self.fn, self.static_key = block, orig_call, None, None

    def __call__(self, params, kw):
        arrays = {k: v for k, v in kw.items() if _is_array_tree(v)}
        static = {k: v for k, v in kw.items() if k not in arrays}
        key = repr(sorted(static.items()))
        if self.fn is None or key != self.static_key:
            block, orig = self.block, self.orig

            def fn(p, arr):
                block.update(p)
                return orig(block, **arr, **static)

            self.fn, self.static_key = mx.compile(fn), key
        return self.fn(params, arrays)


class BlockCompiler:
    """Resident weights, one mx.compile'd function per block (numerics reference for compiled streaming)."""

    def __init__(self, blocks, barrier=False):
        self.blocks, self.barrier, self._patched = blocks, barrier, []
        self.params = [b.parameters() for b in blocks]  # kept apart: tracing installs tracers in the module

    def attach(self):
        for i, b in enumerate(self.blocks):
            cls = type(b)
            cc = _CompiledCall(b, cls.__call__)
            outer = self

            def call(self_, *a, _i=i, _cc=cc, **k):
                assert not a, "block calls must use keyword arguments"
                out = _cc(outer.params[_i], k)
                if outer.barrier:
                    mx.eval(out)
                return out

            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
            self._patched.append((b, cls))

    def detach(self):
        for b, cls in self._patched:
            b.__class__ = cls


class WeightStreamer:
    def __init__(self, blocks, files, prefetch=True, compile=False, lag=0):
        """lag=k (compile mode): block outputs are queued with mx.async_eval and the host only waits for the
        output of block i-k, so the GPU always has k blocks queued (no idle gap at each barrier). Weights of at
        most k+2 blocks are alive (k in flight, the current one, the prefetched one)."""
        self.blocks, self.files, self.prefetch, self.compile = blocks, [str(f) for f in files], prefetch, compile
        self.lag = lag
        self._inflight = []
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
        if not self.compile:
            self.blocks[i].update(params)
        if self.prefetch:
            self._start_load((i + 1) % self.n)
        return stall, params

    def _after(self, i, out, stall, t_start, params=None):
        if self.lag and self.compile:
            # queue this block, wait only for the one `lag` places back; dropping `params` then frees its weights
            mx.async_eval(out)
            self._inflight.append((out, params))
            if len(self._inflight) > self.lag:
                old_out, _ = self._inflight.pop(0)
                mx.eval(old_out)
        else:
            mx.eval(out)
        self.blocks[i].update(self._empty[i])
        self.stats.append({"block": i, "stall_s": stall, "block_s": time.perf_counter() - t_start})

    def flush(self):
        for o, _ in self._inflight:
            mx.eval(o)
        self._inflight = []

    def attach(self):
        streamer = self
        for i, b in enumerate(self.blocks):
            cls = type(b)
            orig = cls.__call__
            cc = _CompiledCall(b, orig) if self.compile else None

            def call(self_, *a, _i=i, _orig=orig, _cc=cc, **k):
                t = time.perf_counter()
                stall, params = streamer._before(_i)
                out = _cc(params, k) if _cc is not None else _orig(self_, *a, **k)
                streamer._after(_i, out, stall, t, params)
                return out

            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
            self._patched.append((b, cls))

    def detach(self):
        for b, cls in self._patched:
            b.__class__ = cls
        self._patched = []
        for i in list(self.pending):
            self.pending.pop(i)
