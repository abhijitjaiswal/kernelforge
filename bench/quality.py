"""Model-agnostic Tier A quality check: PSNR + LPIPS (AlexNet) + side-by-side grid.

    python bench/quality.py --cand DIR --ref DIR --prompts p1,p2,p3 --out runs/<exp>/quality [--label "..."]

Grid rows = prompts; columns = reference | candidate | |diff| x10. LPIPS weights are cached under
models/torch (TORCH_HOME), never in the user's home cache.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from bench.verify import compare, load  # noqa: E402

_lp = None


def lpips_dist(a: np.ndarray, b: np.ndarray) -> float:
    global _lp
    import torch
    import lpips

    if _lp is None:
        _lp = lpips.LPIPS(net="alex", verbose=False)
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    with torch.no_grad():
        return float(_lp(t(a), t(b)).item())


def grid(rows, out_path, cell=384, title=""):
    pad, head = 6, 22
    W = 3 * cell + 4 * pad
    H = head + len(rows) * (cell + pad + 16) + pad
    g = Image.new("RGB", (W, H), (245, 246, 248))
    d = ImageDraw.Draw(g)
    d.text((pad, 5), title, fill=(20, 20, 20))
    for i, (pid, ref, cand, m) in enumerate(rows):
        y = head + i * (cell + pad + 16)
        diff = np.clip(np.abs(ref.astype(np.int16) - cand.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
        for j, (img, cap) in enumerate([(ref, "reference"), (cand, "candidate"), (diff, "|diff| x10")]):
            x = pad + j * (cell + pad)
            g.paste(Image.fromarray(img).resize((cell, cell), Image.LANCZOS), (x, y))
            text = f"{pid} {cap}" + (f"  PSNR {m['psnr_db']:.1f} dB  LPIPS {m['lpips']:.4f}" if j == 2 else "")
            d.text((x, y + cell + 2), text, fill=(20, 20, 20))
    g.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prompts", default="p1,p2,p3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    res, rows = {}, []
    for pid in a.prompts.split(","):
        c, r = load(ROOT / a.cand / f"{pid}.png"), load(ROOT / a.ref / f"{pid}.png")
        m = compare(c, r)
        m["lpips"] = lpips_dist(c, r)
        res[pid] = m
        rows.append((pid, r, c, m))
    grid(rows, out / "grid.png", title=f"{a.label}  (candidate {a.cand} vs reference {a.ref})")
    summary = {
        "cand": a.cand, "ref": a.ref, "label": a.label, "per_prompt": res,
        "lpips_max": max(v["lpips"] for v in res.values()),
        "psnr_min": min(v["psnr_db"] for v in res.values()),
        "tier_a_pass_lpips": all(v["lpips"] < 0.05 for v in res.values()),
    }
    (out / "quality.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
