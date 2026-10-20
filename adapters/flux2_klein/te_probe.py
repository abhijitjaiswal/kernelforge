"""Klein text encoder (Qwen3, bf16) memory anatomy and exact streaming.

    python -m adapters.flux2_klein.te_probe --prep                         # once: per-layer + embedding files
    python -m adapters.flux2_klein.te_probe --exp exp-060-stock --variant stock
    variants:
      stock     mflux get_prompt_embeds as-is (lazy graph; layers after the last used hidden state never run)
      barrier   mx.eval after every decoder layer (resident weights)
      truncate  drop the layers after the last used hidden state explicitly
      stream    truncate + per-layer weight streaming (prefetch) + embedding rows gathered from file
Each variant runs in its own process (lazy-loading state differs) and saves the prompt embedding, so the
outputs can be checked bit for bit against `stock`. The text encoder is eager in mflux, so stock is a
same-mode reference (R1).
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TE_DIR = ROOT / "models" / "klein-te-bf16"
G = 1024**3
LAYERS = (9, 18, 27)   # hidden states used by FLUX.2 (index 0 = embedding output)
N_USED = max(LAYERS)   # decoder layers 0..26 feed hidden state 27


def nbytes(tree):
    return sum(v.nbytes for _, v in tree_flatten(tree))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--variant", choices=["stock", "barrier", "truncate", "stream"])
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--prompt", default="p1")
    ap.add_argument("--ref", help="exp dir with embeds.safetensors to compare against")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    from adapters.flux2_klein.pipeline import DEFAULTS, Adapter
    from engine.streaming import _placeholder

    if a.prep:
        TE_DIR.mkdir(parents=True, exist_ok=True)
        ad = Adapter({"weights": "bf16", "lifecycle": "staged"})
        te = ad.model.text_encoder
        ad.model.transformer = None
        mx.save_safetensors(str(TE_DIR / "embed.safetensors"), {"weight": te.embed_tokens.weight})
        te.embed_tokens.update(_placeholder(te.embed_tokens.parameters()))
        for i in range(N_USED):
            mx.save_safetensors(str(TE_DIR / f"layer{i:03d}.safetensors"), dict(tree_flatten(te.layers[i].parameters())))
            te.layers[i].update(_placeholder(te.layers[i].parameters()))
            print(f"[prep] layer {i}", flush=True)
        return

    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("KF_CHILD") != "1":  # re-exec under /usr/bin/time -l for the OS peak footprint
        from bench.run import parse_time_l
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "adapters.flux2_klein.te_probe", *sys.argv[1:]],
                                env=dict(os.environ, KF_CHILD="1"), stdout=lf, stderr=tf, cwd=ROOT).returncode
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "notes": "crashed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "variant", "peak_mlx_gb", "active_after_gb", "encode_s_median", "identical_to_ref", "process", "returncode")}, default=str))
        sys.exit(rc)

    from bench.memtrace import MemTrace
    from engine.rowgather import gather_rows
    from engine.streaming import WeightStreamer

    mx.set_cache_limit(int(0.5 * G))
    trace = MemTrace().start()
    prompt = json.loads((ROOT / "bench" / "prompts.json").read_text())[a.prompt]
    ad = Adapter({"weights": "bf16", "lifecycle": "staged"})
    m = ad.model
    m.transformer = None
    m.vae = None
    gc.collect()
    te = m.text_encoder
    all_layer_gb = [nbytes(l.parameters()) / G for l in te.layers]  # lazy: shapes only, nothing loaded
    info = {"n_layers": len(te.layers), "layer_gb": all_layer_gb[0], "embed_gb": nbytes(te.embed_tokens.parameters()) / G,
            "all_weights_gb": nbytes(te.parameters()) / G, "used_layers": N_USED,
            "used_weights_gb": sum(all_layer_gb[:N_USED]) + nbytes(te.embed_tokens.parameters()) / G}

    streamer = None
    if a.variant in ("truncate", "stream"):
        te.layers = te.layers[:N_USED]  # hidden state 27 = output of layer index 26; later layers never feed the embedding
    if a.variant == "barrier":
        for l in te.layers:
            cls = type(l)

            def call(self_, *args, _o=cls.__call__, **kw):
                o = _o(self_, *args, **kw)
                mx.eval(o)
                return o

            l.__class__ = type(cls.__name__, (cls,), {"__call__": call})
    if a.variant == "stream":
        streamer = WeightStreamer(list(te.layers), [TE_DIR / f"layer{i:03d}.safetensors" for i in range(N_USED)], prefetch=True)
        streamer.release_all()
        te.embed_tokens.update({"weight": mx.zeros((0,), dtype=mx.bfloat16)})
        embed_file = str(TE_DIR / "embed.safetensors")

        class RowEmbedding:  # exact lookup from file: only the prompt's rows are read
            def __call__(self, ids):
                return gather_rows(embed_file, "weight", __import__("numpy").array(ids))

        te.embed_tokens = RowEmbedding()
        streamer.attach()

    def encode():
        out_ = ad._orig_encode(prompt=prompt, negative_prompt=" ", guidance=DEFAULTS["guidance"])
        mx.eval(out_[0])
        return out_[0]

    times, peaks = [], []
    emb = None
    for r in range(a.runs):
        trace.mark(f"encode{r}")
        mx.reset_peak_memory()
        t = time.perf_counter()
        emb = encode()
        times.append(time.perf_counter() - t)
        peaks.append(mx.get_peak_memory() / G)
    active_after = mx.get_active_memory() / G
    trace.mark("done")
    trace.stop()
    mx.save_safetensors(str(out / "embeds.safetensors"), {"embeds": emb})
    ident = None
    if a.ref:
        ref = mx.load(str(ROOT / a.ref / "embeds.safetensors"))["embeds"]
        ident = {"identical": bool(mx.array_equal(ref, emb).item()),
                 "max_abs": mx.abs(ref.astype(mx.float32) - emb.astype(mx.float32)).max().item()}
    res = {"exp_id": a.exp, "kind": "text_encoder", "variant": a.variant,
           "config": {"weights": "bf16", "prompt_ids": [a.prompt], "te_variant": a.variant, "cache_limit_gb": 0.5},
           "anatomy": info, "embeds_shape": list(emb.shape),
           "encode_s": times, "encode_s_first": times[0], "encode_s_median": sorted(times)[len(times) // 2],
           "peak_mlx_gb": max(peaks), "peak_mlx_first_gb": peaks[0], "active_after_gb": active_after,
           "memtrace": trace.summary(), "identical_to_ref": ident,
           "stream_stall_s": sum(x["stall_s"] for x in streamer.stats) if streamer else None, "mlx_version": mx.__version__}
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
