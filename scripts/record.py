"""Local: turn runs/<exp>/result.json into a results.jsonl row, a RESULTS.md entry and dataset rows.

    python scripts/record.py exp-001 --change "..." --hypothesis "..." --track memory --tier E \
        --verdict keep --baseline exp-001 --lesson "..." [--notes "..."]
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(exp):
    return json.loads((ROOT / "runs" / exp / "result.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--change", required=True)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--track", default="both")
    ap.add_argument("--tier", default="E")
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--baseline", help="exp id whose total_s / peak are the reference")
    ap.add_argument("--lesson", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    r = load(a.exp)
    base = load(a.baseline) if a.baseline else None
    v = r.get("verify", {})
    psnr = r.get("psnr_db_vs_baseline")
    peak_fp = r.get("process", {}).get("peak_footprint_gb")
    row = {
        "exp_id": a.exp,
        "timestamp": r.get("timestamp"),
        "phase": a.phase,
        "model": r.get("model"),
        "change": a.change,
        "hypothesis": a.hypothesis,
        "config": r.get("config"),
        "runs": r.get("runs"),
        "denoise_s": r.get("denoise_s"),
        "vae_s": r.get("vae_s"),
        "text_encode_s": r.get("text_encode_s"),
        "total_s": r.get("total_s"),
        "speedup_vs_baseline": round(base["total_s"]["median"] / r["total_s"]["median"], 4) if base else None,
        "track": a.track,
        "tier": a.tier,
        "peak_mem_gb": peak_fp,
        "peak_mem_mlx_gb": r.get("peak_mem_mlx_gb"),
        "mem_by_stage_gb": r.get("mem_by_stage_gb"),
        "mem_cap_gb": (r.get("config") or {}).get("mem_limit_gb"),
        "mem_reduction_vs_baseline": round(1 - peak_fp / base["process"]["peak_footprint_gb"], 4) if base and peak_fp else None,
        "swap_grew": r.get("swap_grew"),
        "clip_score_delta": None,
        "psnr_db_vs_baseline": psnr,
        "max_abs_pixel_diff": r.get("max_abs_pixel_diff"),
        "lpips_vs_baseline": None,
        "correctness_pass": (psnr is None) or (psnr > 40) if a.tier == "E" else None,
        "verdict": a.verdict,
        "notes": a.notes,
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "deterministic_across_runs": r.get("deterministic_across_runs"),
        "first_vs_last_round_total_s": r.get("first_vs_last_round_total_s"),
        "raw": f"runs/{a.exp}/result.json",
    }
    with open(ROOT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    # experience dataset: one row per (prompt-independent) pipeline-level candidate
    ds = {
        "schema": 1,
        "level": "pipeline",
        "subgraph_sig": None,
        "model": r.get("model"),
        "hardware_id": "apple_m5_8gpu_16gb",
        "shapes": {"resolution": (r.get("config") or {}).get("resolution")},
        "dtype": (r.get("config") or {}).get("weights"),
        "rewrite": a.change,
        "cost_model_pred": None,
        "measured_time_s": r["total_s"]["median"],
        "measured_peak_gb": peak_fp,
        "error": {"psnr_db": psnr, "max_abs": r.get("max_abs_pixel_diff")},
        "exp_id": a.exp,
    }
    with open(ROOT / "dataset" / "candidates.jsonl", "a") as f:
        f.write(json.dumps(ds) + "\n")

    t = r["total_s"]["median"]
    psnr_list = ", ".join("%s %s" % (k, x.get("psnr_db", "n/a")) for k, x in v.items())
    md = (
        f"### {a.exp}: {a.change}\n"
        f"- total {t:.2f} s (denoise {r['denoise_s']['median']:.2f}, VAE {r['vae_s']['median']:.2f}, "
        f"text {r['text_encode_s']['median']:.2f}); peak footprint {peak_fp} GB, MLX peak {r.get('peak_mem_mlx_gb')} GB; "
        f"swap grew: {r.get('swap_grew')}\n"
        + (f"- vs {a.baseline}: {row['speedup_vs_baseline']}x speed, {row['mem_reduction_vs_baseline']} memory reduction\n" if base else "")
        + (f"- PSNR vs ref: {psnr_list} dB, max abs diff {r.get('max_abs_pixel_diff')}\n" if v else "")
        + f"- tier {a.tier}, verdict **{a.verdict}**. {a.lesson}\n\n"
    )
    p = ROOT / "RESULTS.md"
    head = "# RESULTS (newest first)\n\n"
    old = p.read_text()[len(head):] if p.exists() else ""
    p.write_text(head + md + old)
    print(md)


if __name__ == "__main__":
    main()
