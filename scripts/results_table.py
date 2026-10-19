"""Regenerate the summary table at the top of RESULTS.md from runs/*/result.json.

Speedup is against the baseline group (default exp-011-bf16: median of its repeats). Runs whose swap grew
are "memory-only": their timing is invalid and they get no speedup.
"""

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "exp-011-bf16"
START, END = "<!-- summary:start -->", "<!-- summary:end -->"


def group(exp):
    m = re.match(r"^(exp-\d+-[a-z0-9]+)-\d+$", exp)
    return m.group(1) if m else exp


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def main():
    runs = []
    for p in sorted((ROOT / "runs").glob("exp-*/result.json"), key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.parent.name)]):
        runs.append(json.loads(p.read_text()))
    tot = lambda r: (r.get("total_s") or {}).get("median")
    base = [tot(r) for r in runs if group(r["exp_id"]) == BASE and tot(r) is not None and r.get("swap_grew") is False]
    b = statistics.median(base) if base else None
    rows = [f"Speedup vs **{BASE}** = {f(b)} s (median of {len(base)}). *memory-only* = swap grew, so the timing is invalid. "
            f"PSNR is against the run's own quality reference (`=` means bit-identical). Stage-bench and VAE-only runs have no end-to-end total.\n",
            "| Exp | Config | Cache cap | Quality ref | Total s | Speedup | MLX peak GB | OS peak GB | PSNR dB per prompt | Swap |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        c = r.get("config") or {}
        cfg = " · ".join(str(x) for x in [c.get("weights") or r.get("weights"), c.get("lifecycle"), "tiled VAE" if c.get("tile_vae") else None] if x) or r.get("kind", "")
        mem_only = r.get("swap_grew") is True
        sp = "memory-only" if mem_only else (f"{b / tot(r):.2f}×" if b and tot(r) and r.get("swap_grew") is False else "–")
        psnr = " · ".join(f"{k} {'=' if v.get('psnr_db', 0) >= 999 else f(v.get('psnr_db'), 1)}" for k, v in (r.get("verify") or {}).items()) or "–"
        cap = c.get("cache_limit_gb")
        rows.append(f"| {r['exp_id']} | {cfg} | {f(cap, 1) + ' GB' if cap is not None else 'none'} | {(r.get('ref') or '–').replace('baselines/', '')} | "
                    f"{f(tot(r))} | {sp} | {f(r.get('peak_mem_mlx_gb') or r.get('peak_mlx_gb'))} | {f((r.get('process') or {}).get('peak_footprint_gb'))} | "
                    f"{psnr} | {'grew' if mem_only else ('flat' if r.get('swap_grew') is False else '–')} |")
    table = START + "\n" + "\n".join(rows) + "\n" + END
    p = ROOT / "RESULTS.md"
    text = p.read_text()
    if START in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: table, text, flags=re.S)
    else:
        text = text.replace("# RESULTS (newest first)\n\n", "# RESULTS (newest first)\n\n## Summary\n\n" + table + "\n\n## Log\n\n", 1)
    p.write_text(text)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
