"""Klein transformer per-layer quantization sensitivity (engine/sensitivity.py) on one real denoise step.

    python -m adapters.flux2_klein.sensitivity_scan --exp exp-032 [--step 0] [--prompt p1]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.flux2_klein.pipeline import DEFAULTS, Adapter  # noqa: E402
from engine.sensitivity import scan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--prompt", default="p1")
    ap.add_argument("--step", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(int(0.5 * 1024**3))
    prompt = json.loads((ROOT / "bench" / "prompts.json").read_text())[a.prompt]
    ad = Adapter({"weights": "bf16", "lifecycle": "staged"})
    m = ad.model
    emb, txt_ids, _, _ = ad._orig_encode(prompt=prompt, negative_prompt=" ", guidance=DEFAULTS["guidance"])
    mx.eval(emb, txt_ids)
    m.text_encoder = None
    m.vae = None
    gc.collect()
    mx.clear_cache()
    from mflux.models.common.config.config import Config
    cfg = Config(model_config=m.model_config, num_inference_steps=4, height=1024, width=1024, guidance=DEFAULTS["guidance"],
                 image_path=None, image_strength=None, scheduler="flow_match_euler_discrete")
    lat, lat_ids, _, _ = m._prepare_generation_latents(seed=42, config=cfg)
    tr = m.transformer
    mx.eval(lat, lat_ids, tr.parameters())
    ts = cfg.scheduler.timesteps[a.step]
    forward = lambda: tr(hidden_states=lat, encoder_hidden_states=emb, timestep=ts, img_ids=lat_ids, txt_ids=txt_ids, guidance=None)
    rows = scan(tr, forward, bits=(8, 4), group_size=64)
    ok = [r for r in rows if "rel_l2" in r]
    res = {"exp_id": a.exp, "kind": "sensitivity", "config": {"weights": "bf16", "prompt_ids": [a.prompt], "step": a.step, "group_size": 64},
           "metric": "rel L2 of transformer output (predicted velocity) vs bf16, one layer quantized at a time", "rows": rows,
           "ranking_q4": sorted([r for r in ok if r["bits"] == 4], key=lambda r: -r["rel_l2"]),
           "ranking_q8": sorted([r for r in ok if r["bits"] == 8], key=lambda r: -r["rel_l2"]),
           "mlx_version": mx.__version__}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print("most sensitive (q4):")
    for r in res["ranking_q4"][:12]:
        print(f"  {r['layer']:55s} rel {r['rel_l2']:.3e} cos {r['cos']:.6f}")
    print("least sensitive (q4):")
    for r in res["ranking_q4"][-8:]:
        print(f"  {r['layer']:55s} rel {r['rel_l2']:.3e}")


if __name__ == "__main__":
    main()
