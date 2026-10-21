"""python bench/latent_compare.py <a> <b>: bit-compare two saved final-latent tensors (key z)."""
import sys

import mlx.core as mx

x, y = mx.load(sys.argv[1])["z"], mx.load(sys.argv[2])["z"]
d = mx.abs(x.astype(mx.float32) - y.astype(mx.float32))
print("latents", sys.argv[1], "identical", x.shape == y.shape and mx.array_equal(x, y).item(), "max abs %.4g" % d.max().item())
