"""Model-agnostic per-layer quantization sensitivity scan (MLX).

For each Linear in `module` (by dotted path) and each bit width, swap that one layer for
nn.QuantizedLinear (affine, group_size), run `forward()` and compare its output with the unmodified
reference output. Restores the original layer before moving on. Error metrics on the output tensor:
  rel_l2   ||y - y_ref|| / ||y_ref||
  cos      cosine similarity
Rank layers by rel_l2: the least sensitive ones are the first candidates for lower precision in a
mixed-precision recipe; the most sensitive stay in bf16.
"""

import time

import mlx.core as mx
from mlx import nn


def linear_paths(module):
    return [name for name, m in module.named_modules() if isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear)]


def _get_parent(module, path):
    parts = path.split(".")
    obj = module
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _set(parent, last, value):
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def _get(parent, last):
    return parent[int(last)] if last.isdigit() else getattr(parent, last)


def scan(module, forward, paths=None, bits=(8, 4), group_size=64, log=print):
    y_ref = forward()
    mx.eval(y_ref)
    ref = y_ref.astype(mx.float32).reshape(-1)
    ref_norm = mx.linalg.norm(ref).item()
    rows = []
    paths = paths or linear_paths(module)
    for i, path in enumerate(paths):
        parent, last = _get_parent(module, path)
        orig = _get(parent, last)
        n_params = orig.weight.size
        for b in bits:
            t = time.perf_counter()
            try:
                q = nn.QuantizedLinear.from_linear(orig, group_size=group_size, bits=b)
            except ValueError as e:  # e.g. K not divisible by group size
                rows.append({"layer": path, "bits": b, "error": str(e)})
                continue
            _set(parent, last, q)
            y = forward().astype(mx.float32).reshape(-1)
            d = y - ref
            rel = (mx.linalg.norm(d) / ref_norm).item()
            cos = (mx.sum(y * ref) / (mx.linalg.norm(y) * ref_norm)).item()
            _set(parent, last, orig)
            del q, y, d
            rows.append({"layer": path, "bits": b, "rel_l2": rel, "cos": cos, "params": n_params,
                         "bytes_saved_vs_bf16": n_params * 2 - n_params * b / 8 - 2 * 2 * n_params / group_size,
                         "s": time.perf_counter() - t})
        log(f"[sens] {i + 1}/{len(paths)} {path}: " + "  ".join(f"q{r['bits']} rel {r['rel_l2']:.2e}" for r in rows[-len(bits):] if "rel_l2" in r))
    return rows
