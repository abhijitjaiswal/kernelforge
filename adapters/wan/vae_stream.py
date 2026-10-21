"""Exact low-memory decode for the Wan 2.1 3D VAE: temporal streaming + height chunking, fp32 throughout.

Temporal streaming: every CausalConv3d with a temporal kernel > 1 gets a state holding the last (kt - 1) frames of its
input, initialised to zeros (identical to its own causal zero padding). The stock decoder (vae.decode) is then run on one
latent frame at a time and the outputs are concatenated. Every output frame is computed from the same input window
(previous kt - 1 frames + current) as in the full decode, and every other decoder op is per-frame or per-position
(RMS_norm over channels, per-frame spatial attention, per-frame spatial upsample), so the result reproduces mlx-video's
full untiled decode, including its zero-padded start (36 output frames for 9 latents, not the official Wan's 33).
Height chunking: CausalConv3d._conv3d is replaced per instance by a version that evaluates one output frame at a time
and runs the frame's conv2d on horizontal strips of output rows (input rows + kernel halo), so the conv's transient
is one strip instead of every window of the clip.
"""

import mlx.core as mx
from mlx import nn

from mlx_video.models.wan_2.vae import CausalConv3d, Resample


def _chunked_conv3d(self, x, rows):
    """Drop-in for CausalConv3d._conv3d. x: [B, T, H, W, C_in] (already causally and spatially padded)."""
    b, t, h, w, c_in = x.shape
    kt, kh, kw = self.kernel_size
    st, sh, sw = self.stride
    t_out = (t - kt) // st + 1
    h_out = (h - kh) // sh + 1
    w_2d = self.weight.transpose(0, 2, 3, 1, 4).reshape(self.weight.shape[0], kh, kw, kt * c_in)
    outputs = []
    for t_i in range(t_out):
        t0 = t_i * st
        window = x[:, t0:t0 + kt].transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)
        if rows and h_out > rows:
            strips = []
            for r0 in range(0, h_out, rows):
                r1 = min(r0 + rows, h_out)
                s = window[:, r0 * sh: (r1 - 1) * sh + kh]       # input rows needed for output rows [r0, r1)
                o = mx.conv2d(s, w_2d, stride=(sh, sw)) + self.bias
                mx.eval(o)
                strips.append(o)
            out = mx.concatenate(strips, axis=1)
        else:
            out = mx.conv2d(window, w_2d, stride=(sh, sw)) + self.bias
        mx.eval(out)
        outputs.append(out)
    return mx.stack(outputs, axis=1)


def aligned_conv2d_rows(conv, x, rows):
    """Row-chunked 'same' Conv2d that stays bit-identical under Winograd: pad rows once (as the full conv does internally),
    start every strip on the full conv's tile grid, and run each strip with zero row padding. rows must be a multiple of 24
    (measured, exp-114): misaligned strips (each strip re-padded, like engine.chunked.conv_rows) round differently."""
    assert rows % 24 == 0, "strip height must be a multiple of 24 to stay on the Winograd tile grid"
    p = conv.weight.shape[1] // 2
    H = x.shape[1]
    xp = mx.pad(x, [(0, 0), (p, p), (0, 0), (0, 0)])
    outs = []
    for r0 in range(0, H, rows):
        r1 = min(r0 + rows, H)
        o = mx.conv2d(xp[:, r0: r1 + 2 * p], conv.weight, padding=(0, p))
        if "bias" in conv:
            o = o + conv.bias
        mx.eval(o)
        outs.append(o)
    return mx.concatenate(outs, axis=1)


def direct_window_conv(conv, state, x, rows):
    """Causal 3D conv over [state | x] without stock's whole-block concat/pad/transpose (option 1).
    For each output frame, build its kt-frame window straight from state + x, apply the same spatial zero padding and the
    same channels-last layout (taps major, channels minor) as stock, then run the identical conv2d (grid-aligned strips).
    Conv inputs are byte-identical to stock's; only the big intermediate copies disappear.
    state: [B, C, kt-1, H, W]; x: [B, C, T, H, W] -> [B, O, T, H', W']"""
    kt, kh, kw = conv.kernel_size
    st, sh, sw = conv.stride
    assert st == 1, "temporal stride 1 only (decoder convs)"
    b, c, t, h, w = x.shape
    w_2d = conv.weight.transpose(0, 2, 3, 1, 4).reshape(conv.weight.shape[0], kh, kw, kt * c)
    ph, pw = conv._pad_h, conv._pad_w

    def frame(i):  # frame i of the virtual sequence [state | x]
        return state[:, :, i] if i < kt - 1 else x[:, :, i - (kt - 1)]

    outs = []
    for ti in range(t):
        fr = mx.stack([frame(ti + k) for k in range(kt)], axis=2)               # [B, C, kt, H, W]
        if ph or pw:
            fr = mx.pad(fr, [(0, 0), (0, 0), (0, 0), (ph, ph), (pw, pw)])
        hp, wp = fr.shape[3], fr.shape[4]
        win = fr.transpose(0, 3, 4, 2, 1).reshape(b, hp, wp, kt * c)            # stock's window layout
        h_out = (hp - kh) // sh + 1
        if rows and h_out > rows:
            strips = []
            for r0 in range(0, h_out, rows):
                r1 = min(r0 + rows, h_out)
                o = mx.conv2d(win[:, r0 * sh: (r1 - 1) * sh + kh], w_2d, stride=(sh, sw)) + conv.bias
                mx.eval(o)
                strips.append(o)
            o = mx.concatenate(strips, axis=1)
        else:
            o = mx.conv2d(win, w_2d, stride=(sh, sw)) + conv.bias
        mx.eval(o)
        outs.append(o)
    return mx.stack(outs, axis=1).transpose(0, 4, 1, 2, 3)


class StreamingDecoder:
    """Wraps a stock WanVAE: decode(z) streams latent frames through it with per-conv causal state."""

    def __init__(self, vae, rows=None, chunk_latents=1, direct=False):
        self.vae, self.rows, self.chunk, self.direct = vae, rows, chunk_latents, direct
        self.convs = [m for _, m in vae.named_modules() if isinstance(m, CausalConv3d)]
        self.resamples = [m for _, m in vae.named_modules() if isinstance(m, Resample) and m.mode.startswith("upsample")]
        self._patched = []

    def _install(self):
        rows = self.rows
        direct = self.direct
        for conv in self.convs:
            cls = type(conv)
            kt = conv.kernel_size[0]
            conv._kf_state = None

            def call(self_, x, cache_x=None, _orig=cls.__call__, _kt=kt):
                if _kt <= 1 or cache_x is not None:          # 1x1x1 convs need no temporal state
                    return _orig(self_, x, cache_x=cache_x)
                if self_._kf_state is None:                  # zeros == the conv's own causal zero padding
                    b, c, _, h, w = x.shape
                    self_._kf_state = mx.zeros((b, c, _kt - 1, h, w), dtype=x.dtype)
                if direct:
                    out = direct_window_conv(self_, self_._kf_state, x, rows)
                    tail = [self_._kf_state[:, :, i] for i in range(_kt - 1)] + [x[:, :, i] for i in range(x.shape[2])]
                    self_._kf_state = mx.contiguous(mx.stack(tail[-(_kt - 1):], axis=2))
                    mx.eval(self_._kf_state)
                    return out
                full = mx.concatenate([self_._kf_state, x], axis=2)
                # copy into its own buffer: a slice would share (and keep alive) the whole chunk input of every conv
                self_._kf_state = mx.contiguous(full[:, :, -(_kt - 1):])
                mx.eval(self_._kf_state)
                return _orig(self_, x, cache_x=full[:, :, : _kt - 1])  # conv concatenates cache_x and skips its own padding

            members = {"__call__": call}
            if rows:
                members["_conv3d"] = lambda self_, x, _r=rows: _chunked_conv3d(self_, x, _r)
            conv.__class__ = type(cls.__name__, (cls,), members)
            self._patched.append((conv, cls))
        if rows:  # the spatial upsamplers' 3x3 Conv2d runs on all frames of a chunk at once: grid-aligned row strips
            for rs in self.resamples:
                c2d = rs.resample[1]
                cls = type(c2d)
                c2d.__class__ = type(cls.__name__, (cls,), {"__call__": lambda self_, x, _r=rows: aligned_conv2d_rows(self_, x, _r)})
                self._patched.append((c2d, cls))

    def _remove(self):
        for conv, cls in self._patched:
            conv.__class__ = cls
            conv._kf_state = None
        self._patched = []

    def decode(self, z, on_chunk=None):
        """z: [B, z_dim, T, H, W]. Returns [B, 3, T_out, H_out, W_out] in [-1, 1], like WanVAE.decode."""
        self._install()
        try:
            outs = []
            for t0 in range(0, z.shape[2], self.chunk):
                o = self.vae.decode(z[:, :, t0:t0 + self.chunk])
                mx.eval(o)
                outs.append(o)
                if on_chunk:
                    on_chunk(t0, o)
            return mx.concatenate(outs, axis=2)
        finally:
            self._remove()
