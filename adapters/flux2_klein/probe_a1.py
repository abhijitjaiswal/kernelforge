"""Exp A1: redundancy probe on a normal bf16 4-step Klein run (no optimisation).

    python -m adapters.flux2_klein.probe_a1 --exp exp-040 --prompts p1,p2,p3

Per prompt, the eager transformer runs the 4 denoise steps with hooks on all 25 blocks:
  block-step similarity (steps 2-4 vs the previous step, same block):
    cos_out    cosine of the block's output (what the brief asks for; inflated by the residual stream)
    cos_delta  cosine of the block's own contribution, output - input (what output caching actually reuses)
  token near-duplicates (each step, at the input of single-stream blocks 0, 10 and 19, image tokens only):
    fraction of 512 sampled tokens whose most similar other token has cos > 0.95
The final latents are compared with a compiled reference run, so the hooks are shown not to change numerics.
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
from engine.similarity import cosine, near_duplicate_fraction  # noqa: E402

TOKEN_LAYERS = (0, 10, 19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--prompts", default="p1,p2,p3")
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(int(0.5 * 1024**3))
    prompts = json.loads((ROOT / "bench" / "prompts.json").read_text())
    pids = a.prompts.split(",")

    ad = Adapter({"weights": "bf16", "lifecycle": "staged"})
    m = ad.model
    enc = {}
    for p in pids:
        e, t, _, _ = ad._orig_encode(prompt=prompts[p], negative_prompt=" ", guidance=DEFAULTS["guidance"])
        mx.eval(e, t)
        enc[p] = (e, t)
    m.text_encoder = None
    m.vae = None
    gc.collect()
    mx.clear_cache()
    tr = m.transformer
    mx.eval(tr.parameters())
    from mflux.models.common.config.config import Config

    # --- hooks: record block input/output; previous-step copies live in `prev`
    state = {"step": 0, "pid": None}
    prev, rows, tok_rows = {}, [], []
    n_img = None

    def hook(name, kind):
        mod = tr.transformer_blocks[int(name.split(".")[1])] if kind == "double" else tr.single_transformer_blocks[int(name.split(".")[1])]
        cls = type(mod)
        orig = cls.__call__

        def call(self, *args, **kw):
            if kind == "double":
                x_in = mx.concatenate([kw["encoder_hidden_states"], kw["hidden_states"]], axis=1)
            else:
                x_in = kw["hidden_states"] if "hidden_states" in kw else args[0]
                j = int(name.split(".")[1])
                if j in TOKEN_LAYERS:
                    mx.eval(x_in)
                    r = near_duplicate_fraction(x_in[0, -n_img:], n_sample=512, thresh=0.95, seed=state["step"])
                    rc = near_duplicate_fraction(x_in[0, -n_img:], n_sample=512, thresh=0.95, seed=state["step"], center=True)
                    tok_rows.append({"prompt": state["pid"], "step": state["step"] + 1, "layer": name, **r,
                                     "frac_gt_centered": rc["frac_gt"], "median_max_cos_centered": rc["median_max_cos"]})
            o = orig(self, *args, **kw)
            y = mx.concatenate([o[0], o[1]], axis=1) if kind == "double" else o
            mx.eval(y, x_in)
            delta = y - x_in
            if name in prev:
                py, pd = prev[name]
                rows.append({"prompt": state["pid"], "step": state["step"] + 1, "block": name,
                             "cos_out": cosine(y, py), "cos_delta": cosine(delta, pd),
                             "rel_delta_change": (mx.linalg.norm((delta - pd).astype(mx.float32)) / mx.linalg.norm(pd.astype(mx.float32))).item(),
                             "delta_share": (mx.linalg.norm(delta.astype(mx.float32)) / mx.linalg.norm(y.astype(mx.float32))).item()})
            prev[name] = (y, delta)
            return o

        mod.__class__ = type(cls.__name__, (cls,), {"__call__": call})
        return mod, cls

    patched = [hook(f"transformer_blocks.{i}", "double") for i in range(len(tr.transformer_blocks))]
    patched += [hook(f"single_transformer_blocks.{i}", "single") for i in range(len(tr.single_transformer_blocks))]

    check = {}
    try:
        for p in pids:
            e, t = enc[p]
            cfg = Config(model_config=m.model_config, num_inference_steps=a.steps, height=1024, width=1024, guidance=DEFAULTS["guidance"],
                         image_path=None, image_strength=None, scheduler="flow_match_euler_discrete")
            lat0, lid, _, _ = m._prepare_generation_latents(seed=42, config=cfg)
            n_img = lat0.shape[1]
            prev.clear()
            state["pid"] = p
            lat = lat0
            for s in range(a.steps):
                state["step"] = s
                noise = tr(hidden_states=lat, encoder_hidden_states=e, timestep=cfg.scheduler.timesteps[s], img_ids=lid, txt_ids=t, guidance=None)
                lat = cfg.scheduler.step(noise=noise, timestep=s, latents=lat, sigmas=cfg.scheduler.sigmas)
                mx.eval(lat)
            prev.clear()
            check[p] = lat
            print(f"[A1] {p} done", flush=True)
    finally:
        for mod, cls in patched:
            mod.__class__ = cls

    # numerics check: same steps through the compiled predict closure mflux uses (no hooks)
    pred = m._predict(tr)
    lat_match = {}
    for p in pids:
        e, t = enc[p]
        cfg = Config(model_config=m.model_config, num_inference_steps=a.steps, height=1024, width=1024, guidance=DEFAULTS["guidance"],
                     image_path=None, image_strength=None, scheduler="flow_match_euler_discrete")
        lat, lid, _, _ = m._prepare_generation_latents(seed=42, config=cfg)
        for s in range(a.steps):
            noise = pred(latents=lat, latent_ids=lid, prompt_embeds=e, text_ids=t, negative_prompt_embeds=None, negative_text_ids=None,
                         guidance=DEFAULTS["guidance"], timestep=cfg.scheduler.timesteps[s])
            lat = cfg.scheduler.step(noise=noise, timestep=s, latents=lat, sigmas=cfg.scheduler.sigmas)
        mx.eval(lat)
        lat_match[p] = {"cos": cosine(lat, check[p]), "max_abs": mx.abs(lat.astype(mx.float32) - check[p].astype(mx.float32)).max().item()}

    n = len(rows)
    out_pass = sum(r["cos_out"] > 0.98 for r in rows) / n
    delta_pass = sum(r["cos_delta"] > 0.98 for r in rows) / n
    tok_pass = sum(r["frac_gt"] for r in tok_rows) / len(tok_rows)
    tok_pass_c = sum(r["frac_gt_centered"] for r in tok_rows) / len(tok_rows)
    per_prompt = {p: {"block_steps_cos_out_gt_0.98": sum(r["cos_out"] > 0.98 for r in rows if r["prompt"] == p) / max(1, sum(r["prompt"] == p for r in rows)),
                      "block_steps_cos_delta_gt_0.98": sum(r["cos_delta"] > 0.98 for r in rows if r["prompt"] == p) / max(1, sum(r["prompt"] == p for r in rows)),
                      "tokens_near_dup_gt_0.95": sum(r["frac_gt"] for r in tok_rows if r["prompt"] == p) / max(1, sum(r["prompt"] == p for r in tok_rows)),
                      "tokens_near_dup_gt_0.95_centered": sum(r["frac_gt_centered"] for r in tok_rows if r["prompt"] == p) / max(1, sum(r["prompt"] == p for r in tok_rows))}
                  for p in pids}
    res = {"exp_id": a.exp, "kind": "probe", "config": {"weights": "bf16", "prompt_ids": pids, "steps": a.steps, "token_layers": TOKEN_LAYERS},
           "headline": {"pct_block_steps_cos_out_gt_0.98": 100 * out_pass, "pct_block_steps_cos_delta_gt_0.98": 100 * delta_pass,
                        "pct_tokens_near_dup_gt_0.95": 100 * tok_pass, "pct_tokens_near_dup_gt_0.95_centered": 100 * tok_pass_c, "n_block_steps": n, "n_token_samples": sum(r["n_sample"] for r in tok_rows)},
           "per_prompt": per_prompt, "hooked_vs_compiled_latents": lat_match, "block_rows": rows, "token_rows": tok_rows,
           "mlx_version": mx.__version__}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"headline": res["headline"], "per_prompt": per_prompt, "hooked_vs_compiled": lat_match}, indent=1))


if __name__ == "__main__":
    main()
