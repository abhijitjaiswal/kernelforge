"""Profile one Klein denoise step (transformer forward) at 1024^2 with real inputs.

    python -m adapters.flux2_klein.profile_denoise --exp exp-030 [--weights bf16] [--prompt p1]

Reports (all synchronised with mx.eval):
  compiled_step_s  the mx.compile'd predict closure mflux uses (median of --runs)
  eager_step_s     the same transformer call without compile
  per-module       eager per-module times/memory (bench/layer_profile.py), aggregated across blocks
and writes shapes.json (Linear (M,K,N) and SDPA shapes per step) for bench/microbench.py.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.flux2_klein.pipeline import DEFAULTS, Adapter  # noqa: E402
from bench.layer_profile import aggregate, linear_shapes, profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--weights", default="bf16")
    ap.add_argument("--prompt", default="p1")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--res", type=int, default=1024)
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(int(0.5 * 1024**3))
    prompt = json.loads((ROOT / "bench" / "prompts.json").read_text())[a.prompt]

    ad = Adapter({"weights": a.weights, "lifecycle": "staged"})
    m = ad.model
    emb, txt_ids, _, _ = ad._orig_encode(prompt=prompt, negative_prompt=" ", guidance=DEFAULTS["guidance"])
    mx.eval(emb, txt_ids)
    m.text_encoder = None
    gc.collect()
    mx.clear_cache()

    from mflux.models.common.config.config import Config
    cfg = Config(model_config=m.model_config, num_inference_steps=4, height=a.res, width=a.res, guidance=DEFAULTS["guidance"],
                 image_path=None, image_strength=None, scheduler="flow_match_euler_discrete")
    lat, lat_ids, _, _ = m._prepare_generation_latents(seed=42, config=cfg)
    ts = cfg.scheduler.timesteps[0]
    mx.eval(lat, lat_ids, m.transformer.parameters())
    tr = m.transformer
    args = dict(hidden_states=lat, encoder_hidden_states=emb, timestep=ts, img_ids=lat_ids, txt_ids=txt_ids, guidance=None)

    def timed(fn, n):
        mx.eval(fn())  # warm-up / compile
        xs = []
        for _ in range(n):
            t = time.perf_counter()
            mx.eval(fn())
            xs.append(time.perf_counter() - t)
        return xs

    compiled = m._predict(tr)
    c = timed(lambda: compiled(latents=lat, latent_ids=lat_ids, prompt_embeds=emb, text_ids=txt_ids, negative_prompt_embeds=None,
                               negative_text_ids=None, guidance=DEFAULTS["guidance"], timestep=ts), a.runs)
    del compiled
    e = timed(lambda: tr(**args), a.runs)
    _, records, e2e = profile(tr, [lat, emb, ts, lat_ids, txt_ids], max_depth=4)

    top = [r for r in records if r["depth"] == 1]
    instrumented_total = sum(r["time_s"] for r in top)
    agg = aggregate(records)
    for g in agg:
        g["pct_of_instrumented_step"] = 100 * g["time_s"] / instrumented_total
    L_img, L_txt = lat.shape[1], emb.shape[1]
    heads, dh = tr.transformer_blocks[0].attn.heads, tr.transformer_blocks[0].attn.dim_head
    shapes = {"linear": linear_shapes(records),
              "sdpa": [[1, heads, L_img + L_txt, dh, len(tr.transformer_blocks) + len(tr.single_transformer_blocks), "joint attention"]]}
    (out / "shapes.json").write_text(json.dumps(shapes, indent=1))
    res = {"exp_id": a.exp, "kind": "denoise_profile", "weights": a.weights, "config": {"weights": a.weights, "prompt_ids": [a.prompt], "resolution": a.res},
           "tokens": {"image": L_img, "text": L_txt}, "compiled_step_s": statistics.median(c), "compiled_runs_s": c,
           "eager_step_s": statistics.median(e), "eager_runs_s": e, "instrumented_step_s": instrumented_total, "e2e_profile": e2e,
           "aggregate": agg, "records": records, "mlx_version": mx.__version__}
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"tokens img {L_img} txt {L_txt}; compiled step {res['compiled_step_s']:.3f}s  eager {res['eager_step_s']:.3f}s  "
          f"instrumented {instrumented_total:.3f}s  peak {e2e['peak_gb']:.2f} GB")
    for g in [g for g in agg if g["depth"] >= 2][:25]:
        print(f"{g['module']:55s} {g['type']:30s} x{g['calls']:3d} {g['time_s']*1000:8.1f} ms {g['pct_of_instrumented_step']:5.1f}%")
    print(json.dumps(shapes))


if __name__ == "__main__":
    main()
