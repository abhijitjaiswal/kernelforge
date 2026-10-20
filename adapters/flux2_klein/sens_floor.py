"""Noise floor of the sensitivity metric: rel L2 between repeated unmodified forwards, and a no-op layer swap."""
import gc, json, sys
from pathlib import Path
import mlx.core as mx
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adapters.flux2_klein.pipeline import DEFAULTS, Adapter  # noqa: E402

mx.set_cache_limit(int(0.5 * 1024**3))
prompt = json.loads((ROOT / "bench" / "prompts.json").read_text())["p1"]
ad = Adapter({"weights": "bf16", "lifecycle": "staged"}); m = ad.model
emb, tid, _, _ = ad._orig_encode(prompt=prompt, negative_prompt=" ", guidance=DEFAULTS["guidance"]); mx.eval(emb, tid)
m.text_encoder = None; m.vae = None; gc.collect(); mx.clear_cache()
from mflux.models.common.config.config import Config
cfg = Config(model_config=m.model_config, num_inference_steps=4, height=1024, width=1024, guidance=DEFAULTS["guidance"], image_path=None, image_strength=None, scheduler="flow_match_euler_discrete")
lat, lid, _, _ = m._prepare_generation_latents(seed=42, config=cfg); tr = m.transformer; mx.eval(lat, lid, tr.parameters())
f = lambda: tr(hidden_states=lat, encoder_hidden_states=emb, timestep=cfg.scheduler.timesteps[0], img_ids=lid, txt_ids=tid, guidance=None).astype(mx.float32).reshape(-1)
ys = [f() for _ in range(4)]; mx.eval(ys)
rel = lambda a, b: (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()
print("repeat forwards rel L2 vs first:", [rel(y, ys[0]) for y in ys[1:]])
print("bitwise identical:", [bool(mx.array_equal(y, ys[0]).item()) for y in ys[1:]])
print("output dtype path: transformer out", tr(hidden_states=lat, encoder_hidden_states=emb, timestep=cfg.scheduler.timesteps[0], img_ids=lid, txt_ids=tid, guidance=None).dtype)

# rounding-only perturbation: recompute one Linear in fp32 and round its output to bf16
from mlx import nn
from engine.sensitivity import _get_parent, _get, _set


class F32Linear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def __call__(self, x):
        y = x.astype(mx.float32) @ self.lin.weight.astype(mx.float32).T
        if "bias" in self.lin:
            y = y + self.lin.bias.astype(mx.float32)
        return y.astype(x.dtype)


for path in ["single_transformer_blocks.10.attn.to_out", "transformer_blocks.2.ff.linear_in", "norm_out.linear", "x_embedder", "single_transformer_blocks.0.attn.to_qkv_mlp_proj"]:
    p, last = _get_parent(tr, path); orig = _get(p, last); _set(p, last, F32Linear(orig))
    y = f(); mx.eval(y); print(f"fp32-recompute floor {path}: rel {rel(y, ys[0]):.3e}"); _set(p, last, orig)
