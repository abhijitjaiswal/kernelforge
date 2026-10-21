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

from mlx_video.models.wan_2.vae import CausalConv3d


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


class StreamingDecoder:
    """Wraps a stock WanVAE: decode(z) streams latent frames through it with per-conv causal state."""

    def __init__(self, vae, rows=None, chunk_latents=1):
        self.vae, self.rows, self.chunk = vae, rows, chunk_latents
        self.convs = [m for _, m in vae.named_modules() if isinstance(m, CausalConv3d)]
        self._patched = []

    def _install(self):
        rows = self.rows
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
                full = mx.concatenate([self_._kf_state, x], axis=2)
                self_._kf_state = full[:, :, -(_kt - 1):]
                mx.eval(self_._kf_state)
                return _orig(self_, x, cache_x=full[:, :, : _kt - 1])  # conv concatenates cache_x and skips its own padding

            members = {"__call__": call}
            if rows:
                members["_conv3d"] = lambda self_, x, _r=rows: _chunked_conv3d(self_, x, _r)
            conv.__class__ = type(cls.__name__, (cls,), members)
            self._patched.append((conv, cls))

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
