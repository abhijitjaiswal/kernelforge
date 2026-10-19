"""Exact low-memory decode for the FLUX.2 VAE decoder (Klein-specific walk over generic primitives).

modes
  barriers  run the stock decoder modules unchanged, with mx.eval after every top-level block and every
            up-block resnet/upsampler, so the lazy graph never keeps more than one block's intermediates.
  chunked   barriers + exact row-chunking (engine/chunked.py) of every resnet, upsampler and the output
            norm/conv in blocks whose height is >= min_h; GroupNorm uses global (full-tensor) statistics.
"""

import mlx.core as mx
from mlx import nn

from engine.chunked import conv_rows, group_norm_apply, group_norm_stats, upsample2x_conv_rows
from mflux.models.common.config.model_config import ModelConfig

nchw = lambda x: mx.transpose(x, (0, 3, 1, 2))
nhwc = lambda x: mx.transpose(x, (0, 2, 3, 1))


def _resnet_chunked(rb, x, rows):
    dt = ModelConfig.precision
    m1, v1 = group_norm_stats(x, rb.norm1.num_groups, rows)
    h = conv_rows(rb.conv1, x, rows, pre=lambda s: nn.silu(group_norm_apply(rb.norm1, s, m1, v1, dt)))
    m2, v2 = group_norm_stats(h, rb.norm2.num_groups, rows)
    res = conv_rows(rb.conv_shortcut, x, rows) if rb.conv_shortcut is not None else x
    out = conv_rows(rb.conv2, h, rows, pre=lambda s: nn.silu(group_norm_apply(rb.norm2, s, m2, v2, dt)))
    del h
    y = []
    for r0 in range(0, out.shape[1], rows):  # residual add, chunked to avoid a third full-size temporary
        c = out[:, r0:r0 + rows] + res[:, r0:r0 + rows]
        mx.eval(c)
        y.append(c)
    return mx.concatenate(y, axis=1) if len(y) > 1 else y[0]


def decode_chunked(decoder, z, mode="chunked", rows=128, min_h=256):
    """decoder: Flux2Decoder; z: NCHW post_quant_conv output. Returns NCHW like decoder(z)."""
    x = decoder.conv_in(z)
    mx.eval(x)
    x = decoder.mid_block(x)
    mx.eval(x)
    for ub in decoder.up_blocks:
        H = x.shape[2]
        if mode == "barriers" or H < min_h:
            for rb in ub.resnets:
                x = rb(x)
                mx.eval(x)
            for up in ub.upsamplers:
                x = up(x)
                mx.eval(x)
            continue
        x = nhwc(x)
        for rb in ub.resnets:
            x = _resnet_chunked(rb, x, rows)
        for up in ub.upsamplers:
            x = upsample2x_conv_rows(up.conv, x, rows)
        x = nchw(x)
        mx.eval(x)
    if mode == "barriers" or x.shape[2] < min_h:
        x = decoder.conv_norm_out(x)
        x = nn.silu(x).astype(ModelConfig.precision)
        x = decoder.conv_out(x)
        mx.eval(x)
        return x
    dt = ModelConfig.precision
    xh = nhwc(x)
    gn = decoder.conv_norm_out
    m, v = group_norm_stats(xh, gn.num_groups, rows)
    y = conv_rows(decoder.conv_out, xh, rows, pre=lambda s: nn.silu(group_norm_apply(gn, s, m, v, dt)).astype(dt),
                  call=lambda s: nn.Conv2d.__call__(decoder.conv_out, s))  # Flux2ConvOut.__call__ expects NCHW
    return nchw(y)


def decode_packed(vae, packed, vae_opts):
    """Drop-in for Flux2VAE.decode_packed_latents (untiled path) using decode_chunked."""
    lat = vae.unpack_packed_latents(packed)
    lat = (lat / vae.scaling_factor) + vae.shift_factor
    lat = nchw(vae.post_quant_conv(nhwc(lat)))
    return decode_chunked(vae.decoder, lat, **vae_opts)
