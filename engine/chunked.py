"""Model-agnostic exact row-chunked primitives for NHWC conv nets (MLX).

All functions take NHWC arrays and process the H dimension in chunks of `rows` output rows, so no
full-resolution temporary (im2col workspace, fp32 copy, upsampled copy) is ever materialised.
Each chunk is evaluated before the next one starts; the chunk outputs are concatenated.

Exactness:
  conv (stride 1, odd kernel, 'same' padding): each output row only reads input rows within the
    kernel radius, so slicing the input with a halo of k//2 rows and discarding the halo rows of the
    output gives the same arithmetic per output pixel as the full conv. Zero padding at the true image
    border is reproduced because the halo is clipped there and the conv pads that edge itself.
  nearest-2x upsample + conv: the upsampled halo rows are rebuilt from input rows, so it is exact too.
  group norm: statistics (mean, then variance about the mean, both fp32) are accumulated over the
    *whole* tensor before any chunk is normalised, i.e. global statistics, not per-tile ones. Only the
    reduction order differs from mx.fast.layer_norm (numerics-changing at float rounding level).
"""

import mlx.core as mx
from mlx import nn


def _cat_rows(chunks):
    return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)


def _ranges(n, rows):
    return [(r0, min(r0 + rows, n)) for r0 in range(0, n, rows)]


# ---------------------------------------------------------------- group norm with global statistics
def group_norm_stats(x, num_groups, rows):
    """Per-(batch, group) mean and variance over H, W and the group's channels, in fp32. Two passes."""
    B, H, W, C = x.shape
    gs = C // num_groups
    n = H * W * gs
    s = mx.zeros((B, num_groups), dtype=mx.float32)
    for r0, r1 in _ranges(H, rows):
        c = x[:, r0:r1].astype(mx.float32).reshape(B, r1 - r0, W, num_groups, gs)
        s = s + c.sum(axis=(1, 2, 4))
        mx.eval(s)
    mean = s / n
    v = mx.zeros((B, num_groups), dtype=mx.float32)
    m5 = mean.reshape(B, 1, 1, num_groups, 1)
    for r0, r1 in _ranges(H, rows):
        c = x[:, r0:r1].astype(mx.float32).reshape(B, r1 - r0, W, num_groups, gs) - m5
        v = v + (c * c).sum(axis=(1, 2, 4))
        mx.eval(v)
    return mean, v / n


def group_norm_apply(gn: nn.GroupNorm, x_rows, mean, var, out_dtype):
    """Normalise a row slice with precomputed statistics; matches GroupNorm(pytorch_compatible) + affine."""
    B, h, W, C = x_rows.shape
    G = gn.num_groups
    xs = x_rows.astype(mx.float32).reshape(B, h, W, G, C // G)
    y = (xs - mean.reshape(B, 1, 1, G, 1)) * mx.rsqrt(var.reshape(B, 1, 1, G, 1) + gn.eps)
    y = y.reshape(B, h, W, C)
    if "weight" in gn:
        y = y * gn.weight + gn.bias
    return y.astype(out_dtype)


# ---------------------------------------------------------------- convs
def _halo(conv):
    k = conv.weight.shape[1]  # MLX conv weight: (out, kh, kw, in)
    return k // 2


def conv_rows(conv: nn.Conv2d, x, rows, pre=None, call=None):
    """Exact H-chunked 'same' conv. `pre(slice) -> slice` is applied to each input slice first
    (e.g. norm + activation) so that intermediate never exists at full size. `call(slice)` overrides
    how the conv is applied (for Conv2d subclasses whose __call__ changes layout)."""
    call = call or conv
    H = x.shape[1]
    p = _halo(conv)
    out = []
    for r0, r1 in _ranges(H, rows):
        s0, s1 = max(r0 - p, 0), min(r1 + p, H)
        xin = x[:, s0:s1]
        if pre is not None:
            xin = pre(xin)
        y = call(xin)[:, r0 - s0: r0 - s0 + (r1 - r0)]
        mx.eval(y)
        out.append(y)
    return _cat_rows(out)


def upsample2x_conv_rows(conv: nn.Conv2d, x, rows):
    """Exact H-chunked nearest-2x upsample followed by a 'same' conv, without the full upsampled tensor."""
    H2 = x.shape[1] * 2
    p = _halo(conv)
    out = []
    for r0, r1 in _ranges(H2, rows):
        u0, u1 = max(r0 - p, 0), min(r1 + p, H2)      # upsampled rows the conv needs
        i0, i1 = u0 // 2, (u1 - 1) // 2 + 1            # input rows that produce them
        up = mx.repeat(mx.repeat(x[:, i0:i1], 2, axis=1), 2, axis=2)[:, u0 - 2 * i0: u1 - 2 * i0]
        y = conv(up)[:, r0 - u0: r0 - u0 + (r1 - r0)]
        mx.eval(y)
        out.append(y)
    return _cat_rows(out)
