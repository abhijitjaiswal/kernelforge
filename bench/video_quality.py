"""Model-agnostic video quality gate: per-frame PSNR, optional per-frame LPIPS, and a temporal-flicker check.

flicker: compare the frame-to-frame change of the candidate with that of the reference. For each t,
d_c(t) = mean |cand[t] - cand[t-1]|, d_r(t) = mean |ref[t] - ref[t-1]|. A candidate that adds flicker (e.g. tile
or chunk boundaries that shift between frames) shows d_c > d_r; reported as max and mean of (d_c - d_r) in 8-bit levels.
"""

import numpy as np

from bench.verify import compare


def compare_videos(cand, ref, lpips=False):
    if cand.shape != ref.shape:
        return {"error": f"shape mismatch {cand.shape} vs {ref.shape}"}
    per = [compare(cand[t], ref[t]) for t in range(cand.shape[0])]
    ps = [p["psnr_db"] for p in per]
    dc = np.abs(np.diff(cand.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    dr = np.abs(np.diff(ref.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    out = {"identical": bool(np.array_equal(cand, ref)), "psnr_db_min": min(ps), "psnr_db_mean": float(np.mean(ps)),
           "psnr_db_per_frame": ps, "max_abs_pixel_diff": int(max(p["max_abs_pixel_diff"] for p in per)),
           "flicker_excess_max": float((dc - dr).max()) if len(dc) else 0.0, "flicker_excess_mean": float((dc - dr).mean()) if len(dc) else 0.0}
    if lpips:
        from bench.quality import lpips_dist
        lp = [lpips_dist(cand[t], ref[t]) for t in range(cand.shape[0])]
        out.update(lpips_max=max(lp), lpips_mean=float(np.mean(lp)), lpips_per_frame=lp)
    return out
