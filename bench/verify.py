"""Model-agnostic image verification: PSNR, max abs pixel diff, amplified diff image.

    python bench/verify.py candidate.png reference.png [--diff out.png]
"""

import argparse
import json
import math

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def compare(a: np.ndarray, b: np.ndarray, diff_path: str | None = None, amplify: int = 10) -> dict:
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}", "psnr_db": None}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mse = float(np.mean(d.astype(np.float64) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)
    if diff_path:
        Image.fromarray(np.clip(d * amplify, 0, 255).astype(np.uint8)).save(diff_path)
    return {
        "psnr_db": psnr if psnr != math.inf else 999.0,  # 999 = bit-identical
        "max_abs_pixel_diff": int(d.max()),
        "mean_abs_pixel_diff": float(d.mean()),
        "identical": bool(mse == 0),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--diff")
    a = ap.parse_args()
    print(json.dumps(compare(load(a.candidate), load(a.reference), a.diff), indent=2))
