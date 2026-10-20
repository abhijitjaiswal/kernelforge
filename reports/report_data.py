"""Every number in the PDF report comes from here, read from runs/*/result.json at build time.
Nothing is hard-coded except experiment ids and labels."""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def R(exp):
    return json.loads((ROOT / "runs" / exp / "result.json").read_text())


def med(xs):
    return statistics.median(xs)


def fp(exp):
    return R(exp)["process"]["peak_footprint_gb"]


def total(exp):
    r = R(exp)
    return (r.get("total_s") or {}).get("median") or r.get("e2e_image_s")


def collect():
    d = {}
    # --- baseline (exp-011, bf16 staged + cache cap, alternating A/B, cold single image)
    b = [f"exp-011-bf16-{i}" for i in (1, 2, 3)]
    d["base_total"] = med([total(e) for e in b])
    d["base_totals"] = [total(e) for e in b]
    d["base_fp"] = med([fp(e) for e in b])
    d["base_fps"] = [fp(e) for e in b]
    d["base_den"] = med([R(e)["denoise_s"]["median"] for e in b])
    d["base_te"] = med([R(e)["text_encode_s"]["median"] for e in b])
    d["base_vae"] = med([R(e)["vae_s"]["median"] for e in b])

    # --- final pipeline (exp-061 p1 x2, exp-062 p2/p3)
    f = ["exp-061-full-1", "exp-061-full-2"]
    d["final_e2e"] = [R(e)["e2e_image_s"] for e in f]
    d["final_fp"] = [fp(e) for e in f]
    d["final_mlx"] = [R(e)["peak_mlx_gb"] for e in f]
    d["final_swap"] = [R(e)["swap_grew"] for e in f]
    d["final_ident_same_mode"] = [R(e)["verify"]["runs/exp-054-lag1-1"]["identical"] for e in f]
    psnr = {"p1": R("exp-061-full-1")["verify"]["baselines/bf16_free_te"]}
    for p in ("p2", "p3"):
        e = f"exp-062-full-{p}"
        if (ROOT / "runs" / e / "result.json").exists():
            psnr[p] = R(e)["verify"]["baselines/bf16_free_te"]
            d.setdefault("final_fp_p23", []).append(fp(e))
            d.setdefault("final_swap_p23", []).append(R(e)["swap_grew"])
    d["final_psnr"] = psnr
    d["final_fp_all"] = d["final_fp"] + d.get("final_fp_p23", [])
    q = json.loads((ROOT / "runs/exp-064-final-quality/quality.json").read_text())["per_prompt"]
    d["final_lpips"] = {k: v["lpips"] for k, v in q.items()}
    d["diag"] = {t: R(f"exp-063-{t}-p3")["verify"] for t in ("rcomp", "rblockc", "eager")}
    d["diag_rcomp_swap"] = R("exp-063-rcomp-p3")["swap_grew"]
    # cold single image, alternating A/B (exp-065) and p2 uncompiled reference (exp-066)
    cb = [f"exp-065-base-{i}" for i in (1, 2, 3)]
    cf = [f"exp-065-final-{i}" for i in (1, 2, 3)]
    if all((ROOT / "runs" / e / "result.json").exists() for e in cb + cf):
        d["cold_base"] = [total(e) for e in cb]
        d["cold_final"] = [R(e)["e2e_image_s"] for e in cf]
        d["cold_base_fp"] = [fp(e) for e in cb]
        d["cold_final_fp"] = [fp(e) for e in cf]
        d["cold_swap"] = [R(e)["swap_grew"] for e in cb + cf]
        d["cold_final_ident"] = [R(e)["verify"]["runs/exp-061-full-1"]["identical"] for e in cf]
        d["cold_ratio"] = med(d["cold_final"]) / med(d["cold_base"])
        d["final_fp_all"] = d["final_fp_all"] + d["cold_final_fp"]
        d["cold_parts"] = {"decode": [R(f)["decode_s"] - R(b)["vae_s"]["median"] for b, f in zip(cb, cf)],
                           "denoise": [R(f)["denoise_s_median"] - R(b)["denoise_s"]["median"] for b, f in zip(cb, cf)],
                           "encode": [R(f)["encode_s"] - R(b)["text_encode_s"]["median"] for b, f in zip(cb, cf)]}
    if (ROOT / "runs/exp-066-eager-p2/result.json").exists():
        d["p2_eager_ident"] = R("exp-066-eager-p2")["verify"]["runs/exp-062-full-p2"]["identical"]
    d["final_stage_fp"] = {k: v["footprint_peak_gb"] for k, v in R("exp-061-full-1")["memtrace"]["per_stage"].items()}
    # like-for-like resident (same exact pipeline, resident transformer + encoder)
    d["resident_e2e"] = [R(f"exp-054-rblockc-{i}")["e2e_image_s"] for i in (1, 2)]
    d["resident_fp"] = [fp(f"exp-054-rblockc-{i}") for i in (1, 2)]

    # --- streaming overhead per step, same-round references
    def over(cand, ref):
        return 100 * (R(cand)["step_s_median"] / R(ref)["step_s_median"] - 1)
    d["lag1_over"] = [over(f"exp-054-lag1-{i}", f"exp-054-rblockc-{i}") for i in (1, 2)]
    d["lag2_over"] = [over(f"exp-054-lag2-{i}", f"exp-054-rblockc-{i}") for i in (1, 2)]
    d["k2_over"] = [over(f"exp-053-pblockc-{i}", f"exp-053-rblockc-{i}") for i in (1, 2)]
    d["lag1_mlx"] = R("exp-054-lag1-1")["peak_mlx_gb"]
    d["lag2_mlx"] = R("exp-054-lag2-1")["peak_mlx_gb"]
    d["k2_mlx"] = R("exp-053-pblockc-1")["peak_mlx_gb"]
    d["res_mlx"] = R("exp-054-rblockc-1")["peak_mlx_gb"]
    d["stream_ident"] = all(R(e)["verify"][k]["identical"] for e in
                            [f"exp-05{x}-{m}-{i}" for x, m in (("1", "sync"), ("1", "prefetch"), ("2", "sync-cold"), ("2", "prefetch-cold"),
                                                                 ("3", "pblockc"), ("4", "lag1"), ("4", "lag2")) for i in (1, 2)]
                            for k in R(e)["verify"] if k.startswith("runs/"))
    d["n_stream_runs"] = 14
    # eager streaming overhead (barrier lesson)
    res_eager = med([R(f"exp-051-resident-{i}")["step_s_median"] for i in (1, 2)])
    d["eager_res_step"] = res_eager
    d["barrier_over"] = 100 * (med([R(f"exp-052-barrier-{i}")["step_s_median"] for i in (1, 2)]) / res_eager - 1)
    d["prefetch_cold_over"] = 100 * (med([R(f"exp-052-prefetch-cold-{i}")["step_s_median"] for i in (1, 2)]) / res_eager - 1)
    d["prefetch_warm_over"] = 100 * (med([R(f"exp-051-prefetch-{i}")["step_s_median"] for i in (1, 2)]) / res_eager - 1)

    # --- stall hiding (per step)
    d["stall"] = {
        "sync warm": [R(f"exp-051-sync-{i}")["stall_per_step_s"] for i in (1, 2)],
        "prefetch warm": [R(f"exp-051-prefetch-{i}")["stall_per_step_s"] for i in (1, 2)],
        "sync cold": [R(f"exp-052-sync-cold-{i}")["stall_per_step_s"] for i in (1, 2)],
        "prefetch cold": [R(f"exp-052-prefetch-cold-{i}")["stall_per_step_s"] for i in (1, 2)],
    }
    sc, pc = med(d["stall"]["sync cold"]), med(d["stall"]["prefetch cold"])
    d["hidden_pct_cold"] = 100 * (sc - pc) / sc
    disk = R("exp-052-disk")
    d["ssd_cold"] = [disk["disk"]["purged_nocache"]["gbps"], disk["disk"]["purged_cached"]["gbps"]]
    d["ssd_warm"] = [disk["disk"]["nocache"]["gbps"], disk["disk"]["cached"]["gbps"]]
    d["block_gb"] = disk["block_gb_mean"]
    d["block_load_ms"] = 1000 * d["block_gb"] / min(d["ssd_cold"])
    d["block_compute_ms"] = 1000 * res_eager / 25

    # --- per-stage memory before/after (MLX peak, same execution mode within each pair)
    d["stage_pairs"] = {
        "VAE decode": (R("exp-017-stock-p1")["peak_mlx_gb"], R("exp-019-chunk128-p1")["peak_mlx_gb"], "exp-017 / exp-019"),
        "Transformer (denoise)": (R("exp-054-rblockc-1")["peak_mlx_gb"], R("exp-054-lag1-1")["peak_mlx_gb"], "exp-054"),
        "Text encoder": (R("exp-060-stock-1")["peak_mlx_first_gb"], R("exp-060-stream-1")["peak_mlx_first_gb"], "exp-060"),
    }
    d["vae_psnr"] = {p: R(f"exp-019-chunk128-{p}")["verify"][p]["psnr_db"] for p in ("p1", "p2", "p3")}
    d["vae_barrier_mlx"] = R("exp-018-barriers-p1")["peak_mlx_gb"]
    d["vae_profile_max_module"] = max(x["peak_gb"] for x in R("exp-014")["records"])
    d["te"] = {v: R(f"exp-060-{v}-1") for v in ("stock", "barrier", "truncate", "stream")}

    # --- waterfall (bf16 chain, OS peak footprint) + 8-bit reference bar
    d["waterfall_ref"] = ("8-bit stock\n(mflux default)", fp("exp-002"), "exp-002")
    d["waterfall"] = [
        ("Text encoder freed\n(lifecycle)", fp("exp-003"), "exp-003", "E (identical)", R("exp-003")["swap_grew"]),
        ("+ transformer freed\nbefore decode", fp("exp-007"), "exp-007", "E (identical)", R("exp-007")["swap_grew"]),
        ("+ MLX cache cap\n(baseline)", d["base_fp"], "exp-011", "E (identical)", False),
        ("+ exact chunked\nVAE decode", med([fp(f"exp-021-chunk-{i}") for i in (1, 2, 3)]), "exp-021",
         "E ({:.0f}-{:.0f} dB)".format(min(d["vae_psnr"].values()), max(d["vae_psnr"].values())), False),
        ("+ streamed\ntransformer", med([fp(f"exp-054-lag1-{i}") for i in (1, 2)]), "exp-054", "E (identical)", False),
        ("+ streamed\ntext encoder", med(d["final_fp"]), "exp-061", "E (identical)", False),
    ]

    # --- speed: roofline + ceilings
    mb = R("exp-031-arch-default")
    mbx = R("exp-031-arch-g16g")
    prof = R("exp-030")
    d["peak_bf16_tf"] = max(v["tflops"] for k, v in mb["matmul_peak"].items() if k.startswith("bfloat16"))
    d["peak_bf16_tf_nonax"] = max(v["tflops"] for k, v in mbx["matmul_peak"].items() if k.startswith("bfloat16"))
    big = [r for r in mb["linear"] if r["M"] >= 512]
    heavy = [r for r in big if r["M"] >= 4096 and r["N"] >= 3072]
    d["klein_matmul_tf"] = (min(r["bf16"]["tflops"] for r in heavy), max(r["bf16"]["tflops"] for r in heavy))
    d["peak_any_tf"] = max(v["tflops"] for k, v in mb["matmul_peak"].items() if not k.startswith("float32"))
    flops = sum(r["flops"] * r["count_per_step"] for r in mb["linear"])
    d["matmul_tf_weighted"] = flops / sum(r["bf16"]["s"] * r["count_per_step"] for r in mb["linear"]) / 1e12
    d["matmul_eff_vs_bf16_peak"] = 100 * d["matmul_tf_weighted"] / d["peak_bf16_tf"]
    d["matmul_eff_vs_any_peak"] = 100 * d["matmul_tf_weighted"] / d["peak_any_tf"]
    lin_s = sum(r["bf16"]["s"] * r["count_per_step"] for r in mb["linear"])
    sdpa_s = sum(r["s"] * r["count_per_step"] for r in mb["sdpa"])
    step = prof["compiled_step_s"]
    d["step_compiled"] = step
    d["op_split"] = {"matmul": lin_s, "attention": sdpa_s, "other": max(step - lin_s - sdpa_s, 0.0)}
    d["sdpa_tf"] = mb["sdpa"][0]["tflops"]
    d["nax_speedup"] = sum(r["bf16"]["s"] * r["count_per_step"] for r in mbx["linear"]) / lin_s
    d["bandwidth"] = mb["bandwidth"]["gbps_read_plus_write"]
    # top single-op shares (for the op table)
    rows = []
    for r in mb["linear"]:
        rows.append((r["name"], r["count_per_step"], r["bf16"]["s"] * r["count_per_step"], r["bf16"]["tflops"]))
    rows.append(("joint attention (SDPA)", mb["sdpa"][0]["count_per_step"], sdpa_s, d["sdpa_tf"]))
    d["op_rows"] = sorted(rows, key=lambda x: -x[2])
    # ceilings (image level, 4 steps; denoise share from the baseline split)
    other = d["op_split"]["other"]
    den_frac = d["base_den"] / d["base_total"]
    d["ceil_fusion_all"] = 1 / (1 - den_frac * other / step)              # even if all non-matmul work vanished
    d["ceil_fusion_cands"] = 1 / (1 - den_frac * 0.3 / step)              # 0.3 s/step: phase-1 estimate for the top 3 candidates
    dbl = [r for r in mb["linear"] if r["name"].startswith("transformer_blocks.")]
    per_double = sum(r["bf16"]["s"] * r["count_per_step"] for r in dbl) / 5 + mb["sdpa"][0]["s"]
    static_frac = 4 * per_double / step                                   # double blocks 0-3 (the only static ones)
    d["static_frac"] = static_frac
    d["ceil_cache"] = 1 / (1 - den_frac * static_frac * 3 / 4)            # reused on steps 2-4
    a1 = R("exp-040")["headline"]
    a1c = R("exp-041")["headline"]
    d["a1"] = a1
    d["a1c"] = a1c
    d["a1_pp"] = R("exp-040")["per_prompt"]
    d["ceil_merge"] = 1 / (1 - den_frac * a1c["pct_tokens_near_dup_gt_0.95_centered"] / 100)
    q = [R(f"exp-015-q8-{i}")["stages"]["denoise"]["compute_s"]["median"] for i in (1, 2, 3)]
    bf = [R(f"exp-015-bf16-{i}")["stages"]["denoise"]["compute_s"]["median"] for i in (1, 2, 3)]
    d["q8_den"], d["bf16_den"] = med(q), med(bf)
    d["ceil_q8"] = med(bf) / med(q)
    d["eager_vs_compiled"] = R("exp-040")["hooked_vs_compiled_latents"]

    # --- quality of approximations (LPIPS)
    d["lpips_q8"] = json.loads((ROOT / "runs/exp-003/quality/quality.json").read_text())
    d["lpips_tiled"] = json.loads((ROOT / "runs/exp-013/quality_vs_bf16/quality.json").read_text())
    d["thermal"] = {"exp-001": R("exp-001")["first_vs_last_round_total_s"], "exp-002": R("exp-002")["first_vs_last_round_total_s"]}
    d["throttle_mhz"] = (R("exp-015-bf16-1")["power"]["gpu_mhz_busy_median"], R("exp-010-rep-1")["power"]["gpu_mhz_busy_median"])
    d["cap_runs"] = {c: (R(f"exp-022-cap{c}")["peak_mem_mlx_gb"], fp(f"exp-022-cap{c}")) for c in (8, 6)}
    d["sens"] = R("exp-032")
    d["throttle_pct"] = 100 * (1 - d["throttle_mhz"][0] / d["throttle_mhz"][1])
    d["decode_extra_s"] = [R(e)["decode_s"] - d["base_vae"] for e in f]
    d["lifecycle_mlx"] = (R("exp-002")["peak_mem_mlx_gb"], R("exp-001")["peak_mem_mlx_gb"])
    d["cachecap_saving"] = fp("exp-007") - fp("exp-009")
    d["warm_read_gbps"] = d["block_gb"] * 25 / med(d["stall"]["sync warm"])
    d["contention_pct"] = (d["prefetch_warm_over"] - d["barrier_over"], d["prefetch_cold_over"] - d["barrier_over"])
    d["load_compute_ratio"] = d["block_load_ms"] / d["block_compute_ms"]
    return d


def experiment_rows(base_total):
    """Appendix table rows (same logic as scripts/results_table.py)."""
    out = []
    key = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.parent.name)]
    for p in sorted((ROOT / "runs").glob("exp-*/result.json"), key=key):
        r = json.loads(p.read_text())
        c = r.get("config") or {}
        kind = r.get("kind", "")
        cfg = " ".join(str(x) for x in [c.get("weights") or r.get("weights"), c.get("lifecycle") or c.get("stream") or c.get("te_variant"),
                                         "tiledVAE" if c.get("tile_vae") else None, f"vae:{c['vae_impl']}" if c.get("vae_impl") not in (None, "stock") else None] if x) or kind
        t = (r.get("total_s") or {}).get("median") if kind not in ("stages",) else None
        if kind == "streaming":
            t = r.get("e2e_image_s")
        mem_only = r.get("swap_grew") is True
        sp = "memory-only" if mem_only else (f"{base_total / t:.2f}x" if t and r.get("swap_grew") is False and kind in ("", "streaming") else "-")
        ps = " ".join(f"{k}:{'=' if v.get('psnr_db', 0) >= 999 else format(v.get('psnr_db'), '.1f')}" for k, v in (r.get("verify") or {}).items()
                      if not k.startswith("runs/")) or "-"
        same = [v for k, v in (r.get("verify") or {}).items() if k.startswith("runs/")]
        if same:
            ps += " | same-mode: " + ("identical" if all(v.get("identical") for v in same) else "differs")
        cap = c.get("cache_limit_gb")
        out.append([r["exp_id"], cfg[:34], f"{cap:g}" if cap is not None else "-", (r.get("ref") or "-").replace("baselines/", "").replace("runs/", "")[:16],
                    f"{t:.2f}" if t else "-", sp, f"{(r.get('process') or {}).get('peak_footprint_gb'):.2f}" if (r.get("process") or {}).get("peak_footprint_gb") else "-",
                    ps[:40], "grew" if mem_only else ("flat" if r.get("swap_grew") is False else "-")])
    return out
