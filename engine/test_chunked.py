"""Property tests for engine/chunked.py on random tensors: chunked == reference within tolerance.

    python -m engine.test_chunked
"""

import mlx.core as mx
from mlx import nn

from engine.chunked import conv_rows, group_norm_apply, group_norm_stats, upsample2x_conv_rows


def maxdiff(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


def main(seed=0):
    mx.random.seed(seed)
    res = {}
    for dt in (mx.float32, mx.bfloat16):
        for (H, W, C, rows, k) in [(37, 19, 16, 8, 3), (64, 32, 32, 16, 3), (40, 24, 16, 7, 1), (33, 17, 8, 5, 5)]:
            x = mx.random.normal((1, H, W, C)).astype(dt)
            conv = nn.Conv2d(C, C, kernel_size=k, padding=k // 2)
            conv.update({k_: v.astype(dt) for k_, v in conv.parameters().items()})
            ref = conv(x)
            res[f"conv k{k} {H}x{W} rows{rows} {dt}"] = maxdiff(conv_rows(conv, x, rows), ref)
            up_ref = conv(mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2))
            res[f"up2x+conv k{k} {H}x{W} rows{rows} {dt}"] = maxdiff(upsample2x_conv_rows(conv, x, rows), up_ref)
            gn = nn.GroupNorm(num_groups=4, dims=C, eps=1e-6, pytorch_compatible=True)
            gn.update({"weight": mx.random.normal((C,)), "bias": mx.random.normal((C,))})
            g_ref = gn(x.astype(mx.float32)).astype(dt)
            m, v = group_norm_stats(x, 4, rows)
            g = mx.concatenate([group_norm_apply(gn, x[:, r:r + rows], m, v, dt) for r in range(0, H, rows)], axis=1)
            res[f"groupnorm {H}x{W} rows{rows} {dt}"] = maxdiff(g, g_ref)
    worst_f32 = max(v for k, v in res.items() if "float32" in k)
    for k, v in res.items():
        print(f"{v:.3e}  {k}")
    print(f"worst fp32 max-abs error: {worst_f32:.3e}")
    assert worst_f32 < 1e-4, "chunked result differs beyond fp32 rounding"
    return res


if __name__ == "__main__":
    main()
