"""Model-agnostic per-module memory/time profiler for an MLX nn.Module tree.

Every module call (down to `max_depth`) is wrapped: inputs are evaluated, the MLX peak counter is reset,
the call runs, its output is evaluated, and we record
  peak_gb      MLX peak during this call (includes everything still alive: weights, inputs, outputs)
  delta_gb     peak minus memory active just before the call (= the call's own transient + output)
  out_gb       output size
  time_s       wall time of the call (synchronised)
Eager per-module evaluation changes scheduling, so the peaks are per-op working sets, not the fused
graph's peak; the profiler also runs the unwrapped module once to record the true end-to-end peak.
"""

import time

import mlx.core as mx
from mlx.utils import tree_flatten

G = 1024**3


def _nbytes(x):
    if isinstance(x, mx.array):
        return x.nbytes
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(v) for v in x)
    if isinstance(x, dict):
        return sum(_nbytes(v) for v in x.values())
    return 0


def _children(mod, prefix):
    for name, child in mod.children().items():
        items = child if isinstance(child, list) else [child]
        for i, c in enumerate(items):
            if hasattr(c, "children"):
                yield (f"{prefix}.{name}" + (f".{i}" if isinstance(child, list) else "")).lstrip("."), c


def profile(module, inputs, max_depth=3):
    """Run module(*inputs) with per-module instrumentation. Returns (output, records, e2e)."""
    # end-to-end peak with no instrumentation
    mx.eval(inputs)
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    t = time.perf_counter()
    out = module(*inputs)
    mx.eval(out)
    e2e = {"peak_gb": mx.get_peak_memory() / G, "delta_gb": (mx.get_peak_memory() - base) / G, "time_s": time.perf_counter() - t,
           "weights_gb": sum(v.nbytes for _, v in tree_flatten(module.parameters())) / G}
    del out
    mx.clear_cache()

    records, patched = [], []

    def wrap(name, mod, depth):
        cls = type(mod)
        orig = cls.__call__

        def call(self, *a, **k):
            mx.eval([x for x in a if isinstance(x, mx.array)])
            before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            o = orig(self, *a, **k)
            mx.eval(o)
            pk = mx.get_peak_memory()
            arr_in = next((x for x in a if isinstance(x, mx.array)), None)
            w = getattr(self, "weight", None) if "weight" in self else None
            records.append({"module": name, "type": cls.__name__, "depth": depth, "peak_gb": pk / G,
                            "in_shape": list(arr_in.shape) if arr_in is not None else None,
                            "weight_shape": list(w.shape) if isinstance(w, mx.array) else None,
                            "quantized": "scales" in self,
                            "delta_gb": (pk - before) / G, "out_gb": _nbytes(o) / G,
                            "out_shape": list(o.shape) if isinstance(o, mx.array) else None,
                            "time_s": time.perf_counter() - t0})
            return o

        mod.__class__ = type(cls.__name__, (cls,), {"__call__": call})
        patched.append((mod, cls))

    def walk(mod, prefix, depth):
        if depth > max_depth:
            return
        for name, child in _children(mod, prefix):
            wrap(name, child, depth)
            walk(child, name, depth + 1)

    walk(module, "", 1)
    try:
        out = module(*inputs)
        mx.eval(out)
    finally:
        for mod, cls in patched:
            mod.__class__ = cls
    return out, records, e2e


def aggregate(records):
    """Group records by module path with block indices wildcarded (blocks.3.attn -> blocks.*.attn)."""
    import re
    agg = {}
    for r in records:
        key = (re.sub(r"\.\d+(?=\.|$)", ".*", r["module"]), r["type"], r["depth"])
        g = agg.setdefault(key, {"module": key[0], "type": key[1], "depth": key[2], "calls": 0, "time_s": 0.0, "max_delta_gb": 0.0})
        g["calls"] += 1
        g["time_s"] += r["time_s"]
        g["max_delta_gb"] = max(g["max_delta_gb"], r["delta_gb"])
    return sorted(agg.values(), key=lambda g: -g["time_s"])


def linear_shapes(records):
    """(M, K, N, calls, name) for every distinct Linear-like call (has a 2-D weight)."""
    import re
    from math import prod
    out = {}
    for r in records:
        if r["weight_shape"] and len(r["weight_shape"]) == 2 and r["in_shape"] and r["type"].endswith("Linear"):
            M = prod(r["in_shape"][:-1])
            K = r["in_shape"][-1]
            N = r["weight_shape"][0]
            key = (M, K, N)
            name = re.sub(r"\.\d+(?=\.|$)", ".*", r["module"])
            if key in out:
                out[key][3] += 1
            else:
                out[key] = [M, K, N, 1, name]
    return list(out.values())
