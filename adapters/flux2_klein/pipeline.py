"""FLUX.2 Klein 4B adapter (mflux). The only place in the repo that knows about Klein.

Exposes the generic adapter interface used by bench/run.py:
    Adapter(opts) -> .prepare(prompts) -> .generate(prompt, seed, steps, width, height) -> dict
generate() returns {"image": np.uint8 HxWx3, "times": {...}, "mem": {...}}.

Stage timing wraps mflux's own generate_image() instead of re-implementing it, so the
baseline is the stock code path: the text encoder and VAE decode are wrapped with an
explicit mx.eval() and timestamps; denoise is the time between them (mflux evaluates the
latents after every step).

opts:
  weights    "q8" (saved 8-bit copy in ~/imagegen/models) | "bf16" (HF snapshot)
  quantize   int | None   quantize at load (only meaningful with weights=bf16)
  lifecycle  "stock"      all components resident (mflux default)
             "free_te"    encode prompts up front, then drop the text encoder
             "staged"     free_te + drop the transformer before VAE decode (one image per process)
  tile_vae   bool         mflux TilingConfig() decode
  mem_limit_gb float|None mx.set_memory_limit (simulated smaller machine)
"""

import gc
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

IMAGEGEN = Path(os.environ.get("IMAGEGEN_HOME", Path.home() / "imagegen"))  # read-only
os.environ.setdefault("HF_HOME", str(IMAGEGEN / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.vae.tiling_config import TilingConfig  # noqa: E402
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein  # noqa: E402

REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
WEIGHT_PATHS = {
    "q8": IMAGEGEN / "models" / "flux2-klein-4b-q8",
    "bf16": IMAGEGEN / "hf" / "hub" / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / REVISION,
}
MODEL_ID = "flux2-klein-4b"
DEFAULTS = dict(steps=4, guidance=1.0)


def _gb(n):
    return round(n / 1024**3, 3)


class Adapter:
    model_id = MODEL_ID

    def __init__(self, opts: dict):
        self.opts = dict(weights="q8", quantize=None, lifecycle="stock", tile_vae=False, mem_limit_gb=None) | opts
        if self.opts["mem_limit_gb"]:
            mx.set_memory_limit(int(self.opts["mem_limit_gb"] * 1024**3))
        path = WEIGHT_PATHS[self.opts["weights"]]
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), model_path=str(path), quantize=self.opts["quantize"])
        if self.opts["lifecycle"] == "stock":
            # Materialize everything up front (stock behaviour). Other lifecycles leave the
            # mmap-backed weights lazy so a freed component never becomes resident at once.
            mx.eval(self.model.parameters())
        self.load_s = time.perf_counter() - t0
        self.mem_after_load_gb = _gb(mx.get_active_memory())
        self.weights_gb = {k: _gb(sum(v.nbytes for _, v in _flat(getattr(self.model, k).parameters())))
                           for k in ("text_encoder", "transformer", "vae")}
        if self.opts["tile_vae"]:
            self.model.tiling_config = TilingConfig()
        self._orig_encode = self.model._encode_prompt_pair
        self._orig_decode = self.model.vae.decode_packed_latents
        self.model._encode_prompt_pair = self._timed_encode
        self.model.vae.decode_packed_latents = self._timed_decode
        self._cache = {}
        self._marks = {}
        self.mark = lambda name: None  # set by the harness (bench/memtrace.py)

    # --- stage wrappers -------------------------------------------------------------
    def _timed_encode(self, *, prompt, negative_prompt, guidance):
        key = (prompt, guidance)
        if key in self._cache:
            out = self._cache[key]
        else:
            if self.model.text_encoder is None:
                raise RuntimeError("text encoder freed and prompt not pre-encoded")
            out = self._orig_encode(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
            mx.eval([a for a in out if a is not None])
            if self.opts["lifecycle"] != "stock":
                self._cache[key] = out
        self._marks["encode_end"] = time.perf_counter()
        self.mark("denoise")
        self._marks["mem_encode_peak"] = mx.get_peak_memory()
        return out

    def _timed_decode(self, packed_latents, tiling_config=None):
        mx.eval(packed_latents)
        self._marks["denoise_end"] = time.perf_counter()
        self._marks["mem_denoise_peak"] = mx.get_peak_memory()
        if self.opts["lifecycle"] == "staged":
            self.model.transformer = None
            gc.collect()
            mx.clear_cache()
        mx.reset_peak_memory()
        self.mark("vae")
        out = self._orig_decode(packed_latents, tiling_config=tiling_config)
        mx.eval(out)
        self._marks["decode_end"] = time.perf_counter()
        self.mark("post")
        self._marks["mem_vae_peak"] = mx.get_peak_memory()
        return out

    # --- adapter interface -------------------------------------------------------------
    def prepare(self, prompts: list[str]) -> dict:
        """free_te: encode every prompt (timed, after one warm-up encode), then free the encoder."""
        info = {"text_encode_s": {}}
        if self.opts["lifecycle"] == "stock":
            return info
        g = DEFAULTS["guidance"]
        self._orig_encode(prompt="warm-up", negative_prompt=" ", guidance=g)
        for p in prompts:
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            out = self._orig_encode(prompt=p, negative_prompt=" ", guidance=g)
            mx.eval([a for a in out if a is not None])
            info["text_encode_s"][p] = time.perf_counter() - t0
            info["text_encode_peak_gb"] = _gb(mx.get_peak_memory())
            self._cache[(p, g)] = out
        self.mark("free_text_encoder")
        info["active_before_free_gb"] = _gb(mx.get_active_memory())
        self.model.text_encoder = None
        gc.collect()
        mx.clear_cache()
        info["active_after_free_gb"] = _gb(mx.get_active_memory())
        self._prep = info
        return info

    def generate(self, prompt: str, seed: int, steps: int = 4, width: int = 1024, height: int = 1024) -> dict:
        self._marks = {}
        mx.reset_peak_memory()
        self.mark("encode")
        t0 = time.perf_counter()
        img = self.model.generate_image(seed=seed, prompt=prompt, num_inference_steps=steps,
                                        width=width, height=height, guidance=DEFAULTS["guidance"])
        arr = np.array(img.image.convert("RGB"))
        t_end = time.perf_counter()
        m = self._marks
        cached = self.opts["lifecycle"] != "stock"
        te = self._prep["text_encode_s"][prompt] if cached else m["encode_end"] - t0
        times = {
            "text_encode_s": te,
            "text_encode_cached": cached,
            "denoise_s": m["denoise_end"] - m["encode_end"],
            "vae_s": m["decode_end"] - m["denoise_end"],
            "post_s": t_end - m["decode_end"],
        }
        # total = encode + everything generate_image did after encoding
        times["total_s"] = te + (t_end - m["encode_end"])
        mem = {
            "peak_mlx_gb": _gb(max(m["mem_denoise_peak"], m["mem_vae_peak"])),
            "denoise_peak_gb": _gb(m["mem_denoise_peak"]),
            "vae_peak_gb": _gb(m["mem_vae_peak"]),
            "active_gb": _gb(mx.get_active_memory()),
        }
        return {"image": arr, "times": times, "mem": mem}

    def describe(self) -> dict:
        return {"model": MODEL_ID, "revision": REVISION, "opts": self.opts, "load_s": self.load_s,
                "mem_after_load_gb": self.mem_after_load_gb, "weights_gb": self.weights_gb}


def _flat(tree):
    from mlx.utils import tree_flatten
    return tree_flatten(tree)
