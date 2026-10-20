"""Model-agnostic redundancy probes for caching / token-merging decisions (MLX).

cosine(a, b)                  cosine similarity of two tensors, flattened, in fp32
near_duplicate_fraction(x)    fraction of sampled tokens whose most similar *other* token has cos > thresh
"""

import mlx.core as mx


def cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()


def near_duplicate_fraction(x, n_sample=512, thresh=0.95, seed=0, chunk=256, center=False):
    """x: (N, D) tokens. Samples n_sample tokens, finds each one's max cosine to any other token.
    center=True subtracts the mean token first, so a direction shared by all tokens (anisotropy) does not
    count as redundancy."""
    N = x.shape[0]
    xf = x.astype(mx.float32)
    if center:
        xf = xf - xf.mean(axis=0, keepdims=True)
    xn = xf / mx.maximum(mx.linalg.norm(xf, axis=-1, keepdims=True), 1e-12)
    idx = mx.random.permutation(N, key=mx.random.key(seed))[:n_sample]
    best = []
    for s in range(0, idx.shape[0], chunk):
        ii = idx[s:s + chunk]
        sims = xn[ii] @ xn.T                                       # (chunk, N)
        sims = mx.where(mx.arange(N)[None, :] == ii[:, None], -2.0, sims)  # exclude self
        best.append(sims.max(axis=1))
    best = mx.concatenate(best)
    mx.eval(best)
    return {"frac_gt": (best > thresh).mean().item(), "median_max_cos": mx.sort(best)[best.shape[0] // 2].item(),
            "n_sample": int(idx.shape[0]), "thresh": thresh}
