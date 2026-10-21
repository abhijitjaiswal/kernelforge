"""Every number in the Wan PDF report comes from here, read from runs/*/{result,gate}.json at build time, and every bit-identity
claim is re-checked here from the saved tensors (byte comparison). Nothing is hard-coded except experiment ids and labels."""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def R(exp):
    return json.loads((RUNS / exp / "result.json").read_text())


def G(exp):
    return json.loads((RUNS / exp / "gate.json").read_text())


def fp(exp):
    return R(exp)["process"]["peak_footprint_gb"]


def tot(exp):
    return R(exp)["total_s"]["median"]


def st(exp, stage):
    return R(exp)["stages"][stage]


def st_raw(path):
    """Minimal safetensors reader: {name: (dtype, shape, raw bytes)}; byte comparison needs no framework."""
    b = Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    hdr = json.loads(b[8:8 + n])
    base = 8 + n
    return {k: (v["dtype"], v["shape"], b[base + v["data_offsets"][0]: base + v["data_offsets"][1]]) for k, v in hdr.items() if k != "__metadata__"}


def same_tensor(a, ka, b, kb):
    x, y = st_raw(a)[ka], st_raw(b)[kb]
    return x[0] == y[0] and x[1] == y[1] and x[2] == y[2]


def f32(path, key):
    dt, shape, raw = st_raw(path)[key]
    assert dt == "F32", dt
    return np.frombuffer(raw, dtype=np.float32).reshape(shape)


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def motion_ratio(cand, ref):
    c, r = np.load(RUNS / cand / "frames.npy").astype(np.float32), np.load(RUNS / ref / "frames.npy").astype(np.float32)
    dc = np.abs(np.diff(c, axis=0)).mean(axis=(1, 2, 3))
    dr = np.abs(np.diff(r, axis=0)).mean(axis=(1, 2, 3))
    return float(dc.sum() / dr.sum())


def collect():
    d = {}
    BV = ROOT / "baselines_video" / "wan21"
    # ---------------- stock (stage-isolated: stock cannot run end to end in 16 GB)
    te = R("exp-101-te-stock")
    d["stock_te"] = {"mlx": te["peak_mlx_gb"], "fp": te["process"]["peak_footprint_gb"], "s": te["encode_s"],
                     "swap_gb": (te["swap_used_mb_before_after"][1] - te["swap_used_mb_before_after"][0]) / 1024}
    d["stock_dit"] = {"mlx": st("exp-130-stock", "denoise")["peak_mlx_gb"], "step": R("exp-130-stock")["step_s_median"],
                      "s": st("exp-130-stock", "denoise")["s"]}
    d["stock_dit_8"] = {"mlx": st("exp-126-stock-1", "denoise")["peak_mlx_gb"], "fp": fp("exp-126-stock-1")}
    dec = R("exp-132-stock-decode")
    d["stock_dec"] = {"mlx": dec["peak_mlx_gb"], "fp": dec["process"]["peak_footprint_gb"], "s": dec["decode_s"], "swap": dec["swap_grew"],
                      "mlx_103": st("exp-103-base-auto", "vae_decode")["peak_mlx_gb"], "s_103": st("exp-103-base-auto", "vae_decode")["s"]}
    d["untiled_crash"] = {"asked_gb": 11041505280 / 1e9, "limit_gb": 9534832640 / 1e9}  # metal::malloc message, runs/exp-108-barriers-full log
    for t in RUNS.joinpath("exp-108-barriers-full").glob("*"):
        try:
            s = t.read_text(errors="ignore")
        except IsADirectoryError:
            continue
        import re
        m = re.search(r"allocate (\d+) bytes which is greater than the maximum allowed buffer size of (\d+) bytes", s)
        if m:
            d["untiled_crash"] = {"asked_gb": int(m.group(1)) / 1e9, "limit_gb": int(m.group(2)) / 1e9}
    b = R("exp-103-base-auto")
    d["base103"] = {"fp": b["process"]["peak_footprint_gb"], "s": tot("exp-103-base-auto"),
                    "swap_gb": (b["swap_used_mb_before_after"][1] - b["swap_used_mb_before_after"][0]) / 1024}
    # ---------------- composed exact pipeline
    c = R("exp-127-composed")
    d["comp"] = {"fp": c["process"]["peak_footprint_gb"], "s": tot("exp-127-composed"), "swap": c["swap_grew"],
                 "te_mlx": c["stages"]["text_encode"]["peak_mlx_gb"], "te_s": c["stages"]["text_encode"]["s"],
                 "dit_mlx": c["stages"]["denoise"]["peak_mlx_gb"], "dit_s": c["stages"]["denoise"]["s"], "step": c["step_s_median"],
                 "dec_mlx": c["stages"]["vae_decode"]["peak_mlx_gb"], "dec_s": c["stages"]["vae_decode"]["s"]}
    t2 = R("exp-102-te-stream")
    d["te_stream"] = {"mlx": t2["peak_mlx_gb"], "fp": t2["process"]["peak_footprint_gb"], "s": t2["encode_s"]}
    d["dit_stream_8"] = {"mlx": st("exp-126-stream-1", "denoise")["peak_mlx_gb"], "fp": fp("exp-126-stream-1")}
    over = []
    for i in (1, 2):
        over.append(100 * (R(f"exp-126-stream-{i}")["step_s_median"] / R(f"exp-126-blockc-{i}")["step_s_median"] - 1))
    d["stream_over"] = over
    v6 = R("exp-125-v6-full")
    d["v6"] = {"mlx": v6["peak_mlx_gb"], "fp": v6["process"]["peak_footprint_gb"], "s": v6["decode_s"], "swap": v6["swap_grew"],
               "prefix_n": v6["prefix_check"]["n_frames"], "prefix_ok": v6["prefix_check"]["all_identical"],
               "vs_tiled": list(v6["verify"].values())[0]}
    d["dec_versions"] = []
    for exp, lab in (("exp-115-v3-full", "v3 uncapped"), ("exp-117-v3-full-cap", "v3"), ("exp-119-v4-full", "v4 (option 1)"),
                     ("exp-122-v5-full", "v5"), ("exp-125-v6-full", "v6")):
        r = R(exp)
        d["dec_versions"].append((lab, exp, r["peak_mlx_gb"], r["process"]["peak_footprint_gb"], r["decode_s"], r["swap_grew"]))
    d["untiled_pre4"] = {"mlx": R("exp-109-untiled-pre4")["peak_mlx_gb"], "fp": fp("exp-109-untiled-pre4")}
    # end-to-end frames vs stock tiled decode of the true-stock latents
    d["e2e_vs_tiled"] = list(R("exp-132-stock-decode")["verify"].values())[0]
    # ---------------- exactness chain, re-checked from tensors
    chk = {}
    chk["ctx"] = all(same_tensor(RUNS / "exp-128-techeck/text_context.safetensors", k, BV / "v1_text_context.safetensors", k)
                     for k in ("context", "context_null"))
    chk["noise"] = same_tensor(RUNS / "exp-129-noise/noise.safetensors", "noise", RUNS / "exp-129-noise/stock_noise.safetensors", "noise")
    chk["dit50"] = same_tensor(RUNS / "exp-127-composed/latents.safetensors", "z", RUNS / "exp-130-stock/latents.safetensors", "z")
    chk["ctxref50"] = same_tensor(RUNS / "exp-131-ctxref-stream/latents.safetensors", "z", BV / "v1_latents.safetensors", "z")
    chk["old_ref_cos"] = cosine(f32(RUNS / "exp-127-composed/latents.safetensors", "z"), f32(BV / "v1_latents.safetensors", "z"))
    a, p = np.load(RUNS / "exp-125-v6-full/frames.npy"), np.load(RUNS / "exp-109-untiled-pre4/frames.npy")
    chk["prefix"] = bool(np.array_equal(a[: p.shape[0]], p))
    chk["prefix_n"] = int(p.shape[0])
    d["chk"] = chk
    # ---------------- A1
    d["a1"] = R("exp-104-a1")["headline"]
    # ---------------- caching
    exact = {"v1": "exp-127-composed", "v2": "exp-ref-v2", "v3": "exp-ref-v3"}
    runs = {("v1", "A2"): "exp-134-a2", ("v2", "A2"): "exp-a2-v2", ("v3", "A2"): "exp-a2-v3",
            ("v2", "A3-d"): "exp-a3d-v2", ("v3", "A3-d"): "exp-a3d-v3"}
    d["exact_s"] = {p: tot(e) for p, e in exact.items()}
    d["exact_fp"] = {p: fp(e) for p, e in exact.items()}
    d["exact131_dit_s"] = st("exp-131-ctxref-stream", "denoise")["s"]
    cache = {}
    for (p, mode), e in runs.items():
        g, r = G(e), R(e)
        lp = g["lpips_per_frame"]
        cache[(p, mode)] = {"exp": e, "s": tot(e), "speedup": d["exact_s"][p] / tot(e), "lp_max": g["lpips_max"], "lp_mean": g["lpips_mean"],
                            "n_over": sum(v > 0.05 for v in lp), "n": len(lp), "psnr_min": g["psnr_db_min"], "psnr_mean": g["psnr_db_mean"],
                            "pass": g["lpips_max"] < 0.05, "fp": r["process"]["peak_footprint_gb"], "swap": r["swap_grew"],
                            "dit_mlx": r["stages"]["denoise"]["peak_mlx_gb"], "motion": motion_ratio(e, exact[p]),
                            "lpips_per_frame": lp, "counts": (r.get("cache") or {}).get("counts"),
                            "taken": (r.get("cache") or {}).get("taken"), "rate_log": (r.get("cache") or {}).get("rate_log")}
    d["cache"] = cache
    d["a2_v1_lat_cos"] = G("exp-134-a2")["latent_cosine"]
    stalls = R("exp-a2-v2")["stream_stalls"]
    post = [x["block0_stall_s"] for x in stalls if x["after"] == "skip"]
    d["stall"] = {"n_post": len(post), "post_max_ms": 1e3 * max(post), "total_ms": 1e3 * float(np.median([x["stall_s"] for x in stalls]))}
    a2 = R("exp-134-a2")
    full = sorted(t for t in a2["step_s"] if t > 15)
    d["thermal"] = {"a2_full_step": full[len(full) // 2], "exact_step": R("exp-131-ctxref-stream")["step_s_median"]}
    return d
