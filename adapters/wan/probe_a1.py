"""Exp A1 for Wan: step/block redundancy probe on a normal bf16 50-step run (no optimisation), plus final latents.

    python -m adapters.wan.probe_a1 --exp exp-104 --prompt v1 --context-ref baselines_video/wan21/v1_text_context.safetensors

mlx-video's generate_video() runs uncompiled (hooks cannot live inside mx.compile; R1: similarity statistics only, no
bit-identity claim). Every WanAttentionBlock is hooked; all statistics use the conditional half of the CFG batch.
  block-step (steps 2..N vs the previous step, same block): cos_out (block output) and cos_delta (output - input,
  the block's own contribution, which is what output caching reuses); delta_share = |delta| / |output|
  tokens (blocks 0, mid, last; each step): fraction of 512 sampled tokens with a near-duplicate (cos > 0.95), raw and
  mean-centred (engine/similarity.py)
The final latents (decoder input) are saved so VAE experiments can iterate on decode alone. The decode itself is skipped.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


class _Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model-dir", default="models/wan21_mlx")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--context-ref", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    import mlx_video.models.wan_2.generate as gen
    from engine.similarity import cosine, near_duplicate_fraction

    prompt = json.loads((ROOT / "bench" / "video_prompts.json").read_text())[a.prompt]
    ref = mx.load(str(ROOT / a.context_ref))
    queue = [ref["context"], ref["context_null"]]
    gen.load_t5_encoder = lambda *args, **kw: object()
    gen.encode_text = lambda *args, **kw: queue.pop(0)

    state = {"step": 0}
    prev, rows, tok_rows = {}, [], []
    n_blocks = {}

    def hook_model(model):
        blocks = model.blocks
        n = len(blocks)
        n_blocks["n"] = n
        tok_layers = {0, n // 2, n - 1}
        for i, b in enumerate(blocks):
            cls = type(b)

            def call(self_, x, *args, _i=i, _o=cls.__call__, **kw):
                x0 = x[0]
                if _i in tok_layers:
                    mx.eval(x0)
                    for center in (False, True):
                        r = near_duplicate_fraction(x0, n_sample=512, thresh=0.95, seed=state["step"], center=center)
                        tok_rows.append({"step": state["step"] + 1, "block": _i, "centered": center, **r})
                y = _o(self_, x, *args, **kw)
                y0 = y[0]
                d = y0 - x0
                mx.eval(y0, d)
                if _i in prev:
                    py, pd = prev[_i]
                    rows.append({"step": state["step"] + 1, "block": _i, "cos_out": cosine(y0, py), "cos_delta": cosine(d, pd),
                                 "delta_share": (mx.linalg.norm(d.astype(mx.float32)) / mx.linalg.norm(y0.astype(mx.float32))).item()})
                prev[_i] = (y0, d)
                return y
            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
        return model

    orig_load = gen.load_wan_model
    gen.load_wan_model = lambda *args, **kw: hook_model(orig_load(*args, **kw))

    orig_tqdm = gen.tqdm
    t0 = time.perf_counter()

    def tqdm_w(it, *args, **kw):
        if kw.get("desc") != "Diffusion":
            return orig_tqdm(it, *args, **kw)

        def g():
            last = time.perf_counter()
            for k, x in enumerate(it):
                state["step"] = k
                yield x
                now = time.perf_counter()
                print(f"[A1] step {k + 1}/{a.steps} {now - last:.1f}s (elapsed {now - t0:.0f}s)", flush=True)
                last = now
        return g()
    gen.tqdm = tqdm_w

    lat = {}
    orig_vae = gen.load_vae_decoder

    def vae_w(*args, **kw):
        v = orig_vae(*args, **kw)

        def grab(z, *aa, **kk):
            lat["z"] = z
            raise _Stop()
        v.decode = grab
        v.decode_tiled = grab
        return v
    gen.load_vae_decoder = vae_w

    try:
        gen.generate_video(model_dir=str(ROOT / a.model_dir), prompt=prompt, width=a.width, height=a.height, num_frames=a.frames,
                           steps=a.steps, seed=a.seed, output_path=str(out / "unused.mp4"), no_compile=True)
    except _Stop:
        pass
    mx.save_safetensors(str(out / "latents.safetensors"), {"z": lat["z"]})

    n = len(rows)
    late = [r for r in rows if r["step"] > a.steps // 5]  # after the first 20% of steps
    def pct(rs, key, th=0.98):
        return 100 * sum(r[key] > th for r in rs) / max(1, len(rs))
    raw = [r for r in tok_rows if not r["centered"]]
    cen = [r for r in tok_rows if r["centered"]]
    head = {"pct_block_steps_cos_out_gt_0.98": pct(rows, "cos_out"), "pct_block_steps_cos_delta_gt_0.98": pct(rows, "cos_delta"),
            "pct_late_block_steps_cos_delta_gt_0.98": pct(late, "cos_delta"),
            "pct_tokens_near_dup_gt_0.95": 100 * sum(r["frac_gt"] for r in raw) / len(raw),
            "pct_tokens_near_dup_gt_0.95_centered": 100 * sum(r["frac_gt"] for r in cen) / len(cen),
            "n_block_steps": n, "n_blocks": n_blocks.get("n")}
    res = {"exp_id": a.exp, "kind": "probe", "model": Path(a.model_dir).name,
           "config": {"prompt_ids": [a.prompt], "width": a.width, "height": a.height, "frames": a.frames, "steps": a.steps, "seed": a.seed,
                      "compile": False}, "headline": head, "block_rows": rows, "token_rows": tok_rows, "latents": f"runs/{a.exp}/latents.safetensors",
           "latents_shape": list(lat["z"].shape), "wall_s": time.perf_counter() - t0, "mlx_version": mx.__version__}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
