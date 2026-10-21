"""Bit-compare two text-context safetensors (context, context_null).  python bench/ctx_compare.py <a> <b>"""
import sys

import mlx.core as mx

c, f = mx.load(sys.argv[1]), mx.load(sys.argv[2])
for k in ("context", "context_null"):
    x, y = c[k], f[k]
    same = x.shape == y.shape
    print(k, x.shape, x.dtype, y.shape, y.dtype, "identical", same and mx.array_equal(x, y).item(),
          "max abs", mx.abs(x - y).max().item() if same else None)
