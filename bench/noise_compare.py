"""python bench/noise_compare.py <a> <b>: bit-compare two saved noise tensors."""
import sys

import mlx.core as mx

x, y = mx.load(sys.argv[1])["noise"], mx.load(sys.argv[2])["noise"]
print("noise", x.shape, y.shape, "identical", x.shape == y.shape and mx.array_equal(x, y).item())
