"""Quality gate for an approximate video run vs its exact reference: per-frame PSNR, LPIPS (AlexNet), flicker excess, and final-latent
cosine.  Run with an env that has lpips (~/kf-venv):  python bench/video_gate.py <cand run> <ref run> [--latents-ref <run>] --out <json>"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.video_quality import compare_videos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cand")
ap.add_argument("ref")
ap.add_argument("--latents-ref")
ap.add_argument("--out", required=True)
a = ap.parse_args()
c, r = np.load(ROOT / a.cand / "frames.npy"), np.load(ROOT / a.ref / "frames.npy")
res = {"cand": a.cand, "ref": a.ref, **compare_videos(c, r, lpips=True)}
if a.latents_ref:
    x = load_file(str(ROOT / a.cand / "latents.safetensors"))["z"].astype(np.float64).ravel()
    y = load_file(str(ROOT / a.latents_ref / "latents.safetensors"))["z"].astype(np.float64).ravel()
    res["latent_cosine"] = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    res["latent_rel_l2"] = float(np.linalg.norm(x - y) / np.linalg.norm(y))
res["tierA_lpips_lt_0.05"] = res["lpips_max"] < 0.05
(ROOT / a.out).write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if not k.endswith("per_frame")}))
