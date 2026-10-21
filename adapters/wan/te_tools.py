"""Wan umT5-XXL text encoder: stock capture (frozen reference) and exact streaming with per-layer fp32 upcast.

    python -m adapters.wan.te_tools capture --exp exp-101 --prompt v1 --swap-abort-gb 16   # stock mlx-video, writes the frozen reference
    python -m adapters.wan.te_tools prep                                                     # per-block bf16 files + embedding file
    python -m adapters.wan.te_tools stream  --exp exp-102 --prompt v1                        # streamed encoder vs the frozen reference

capture runs mlx-video's own generate_video() and stops it right after the prompt and negative-prompt encodings,
so tokenizer, text cleaning, negative-prompt resolution and the fp32 upcast are exactly stock and the DiT never loads.
stream swaps in an encoder whose 24 blocks are streamed from per-block bf16 files and upcast to fp32 as each loads
(fp32 numerics at roughly bf16-file memory), with the 256k-row embedding gathered from file by token id and upcast.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))  # tokenizer cache stays inside the project
G = 1024**3
MODEL_DIR = ROOT / "models" / "wan21_mlx"
BLOCK_DIR = ROOT / "models" / "wan21-te-blocks"
REF_DIR = ROOT / "baselines_video" / "wan21"


class _Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "prep", "stream"])
    ap.add_argument("--exp")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--swap-abort-gb", type=float, default=8.0)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    if a.mode == "prep":
        BLOCK_DIR.mkdir(parents=True, exist_ok=True)
        w = mx.load(str(MODEL_DIR / "t5_encoder.safetensors"))  # lazy; bf16 as converted
        n = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("blocks."))
        for i in range(n):
            p = f"blocks.{i}."
            mx.save_safetensors(str(BLOCK_DIR / f"block{i:02d}.safetensors"), {k[len(p):]: v for k, v in w.items() if k.startswith(p)})
            print(f"[prep] block {i}", flush=True)
        mx.save_safetensors(str(BLOCK_DIR / "embed.safetensors"), {"weight": w["token_embedding.weight"]})
        rest = {k: v for k, v in w.items() if not k.startswith("blocks.") and k != "token_embedding.weight"}
        mx.save_safetensors(str(BLOCK_DIR / "rest.safetensors"), rest)
        print("[prep] rest:", sorted(rest), "dtypes:", sorted({str(v.dtype) for v in w.values()}), flush=True)
        return

    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("KF_CHILD") != "1":
        from bench.run import parse_time_l
        with open(out / "time.txt", "w") as tf, open(out / "log", "w") as lf:
            rc = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "adapters.wan.te_tools", *sys.argv[1:]],
                                env=dict(os.environ, KF_CHILD="1"), stdout=lf, stderr=tf, cwd=ROOT).returncode
        rp = out / "result.json"
        res = json.loads(rp.read_text()) if rp.exists() else {"exp_id": a.exp, "notes": "crashed; see log"}
        res["process"] = parse_time_l((out / "time.txt").read_text())
        res["returncode"] = rc
        rp.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res.get(k) for k in ("exp_id", "mode", "encode_s", "peak_mlx_gb", "identical_to_ref", "process", "swap_grew", "returncode", "error")}, default=str))
        sys.exit(rc)

    from bench.memtrace import MemTrace
    from bench.run import swap_used_mb
    import mlx_video.models.wan_2.generate as gen

    prompt = json.loads((ROOT / "bench" / "video_prompts.json").read_text())[a.prompt]
    swap0 = swap_used_mb()
    trace = MemTrace().start()
    res = {"exp_id": a.exp, "kind": "video_text_encoder", "mode": a.mode, "config": {"prompt_ids": [a.prompt], "te": a.mode}}

    def watchdog():
        while True:
            time.sleep(2)
            s = swap_used_mb() or 0
            if s - (swap0 or 0) > a.swap_abort_gb * 1024:
                res.update(error=f"ABORTED by swap watchdog (+{(s - swap0) / 1024:.1f} GB, limit {a.swap_abort_gb} GB)", swap_used_mb_before_after=[swap0, s], swap_grew=True)
                (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    marks = {}
    if a.mode == "stream":
        from mlx_video.models.wan_2.text_encoder import T5Encoder
        from engine.rowgather import gather_rows
        from engine.streaming import WeightStreamer
        import numpy as np

        class UpcastStreamer(WeightStreamer):  # pattern variant: store bf16, upcast per layer at load (adapter-level; engine frozen)
            def _start_load(self, i):
                params = tree_unflatten([(k, v.astype(mx.float32)) for k, v in mx.load(self.files[i]).items()])
                if self.prefetch:
                    mx.async_eval(params)
                self.pending[i] = params

        def load_streamed(model_path, config):
            enc = T5Encoder(vocab_size=config.t5_vocab_size, dim=config.t5_dim, dim_attn=config.t5_dim_attn, dim_ffn=config.t5_dim_ffn,
                            num_heads=config.t5_num_heads, num_layers=config.t5_num_layers, num_buckets=config.t5_num_buckets, shared_pos=False)
            rest = {k: v.astype(mx.float32) for k, v in mx.load(str(BLOCK_DIR / "rest.safetensors")).items()}
            enc.load_weights(list(rest.items()), strict=False)
            files = [BLOCK_DIR / f"block{i:02d}.safetensors" for i in range(len(enc.blocks))]
            st = UpcastStreamer(list(enc.blocks), files, prefetch=True)
            st.release_all()
            ef = str(BLOCK_DIR / "embed.safetensors")

            class RowEmbedding:  # stock: whole table upcast to fp32, then lookup; here: lookup in the bf16 file, then upcast
                def __call__(self, ids):
                    return gather_rows(ef, "weight", np.array(ids)).astype(mx.float32)
            enc.token_embedding = RowEmbedding()
            mx.eval(enc.norm.parameters())
            st.attach()
            marks["streamer"] = st
            return enc
        gen.load_t5_encoder = load_streamed

    orig_load = gen.load_t5_encoder

    def load_w(*args, **kw):
        trace.mark("text_encoder_load")
        mx.reset_peak_memory()
        t = time.perf_counter()
        r = orig_load(*args, **kw)
        marks["load_s"] = time.perf_counter() - t
        marks["load_peak"] = mx.get_peak_memory() / G
        marks["weights_gb"] = sum(v.nbytes for _, v in tree_flatten(r.parameters())) / G
        trace.mark("text_encode")
        mx.reset_peak_memory()
        marks["t_enc"] = time.perf_counter()
        return r
    gen.load_t5_encoder = load_w

    outs = []
    orig_enc = gen.encode_text

    def enc_w(*args, **kw):
        r = orig_enc(*args, **kw)
        mx.eval(r)
        outs.append(r)
        if len(outs) == 2:  # prompt + negative prompt (CFG on for Wan2.1 defaults)
            marks["encode_s"] = time.perf_counter() - marks["t_enc"]
            marks["encode_peak"] = mx.get_peak_memory() / G
            raise _Stop()
        return r
    gen.encode_text = enc_w

    try:
        gen.generate_video(model_dir=str(MODEL_DIR), prompt=prompt, width=832, height=480, num_frames=33, steps=50, seed=42,
                           output_path=str(out / "unused.mp4"))
    except _Stop:
        pass
    trace.mark("done")
    trace.stop()
    trace.dump(out / "memtrace.json")
    swap1 = swap_used_mb()
    ctx, ctx_null = outs
    res.update(load_s=marks.get("load_s"), encode_s=marks.get("encode_s"), load_peak_mlx_gb=marks.get("load_peak"),
               peak_mlx_gb=max(marks.get("load_peak", 0), marks.get("encode_peak", 0)), te_weights_gb=marks.get("weights_gb"),
               context_shapes=[list(ctx.shape), list(ctx_null.shape)], context_dtype=str(ctx.dtype), memtrace=trace.summary(),
               swap_used_mb_before_after=[swap0, swap1], swap_grew=(swap1 or 0) - (swap0 or 0) > 32, mlx_version=mx.__version__)
    if "streamer" in marks:
        res["stream_stall_s"] = sum(x["stall_s"] for x in marks["streamer"].stats)
    refp = REF_DIR / f"{a.prompt}_text_context.safetensors"
    if a.mode == "capture":
        REF_DIR.mkdir(parents=True, exist_ok=True)
        if refp.exists():
            sys.exit(f"refusing to overwrite frozen reference {refp}")
        mx.save_safetensors(str(refp), {"context": ctx, "context_null": ctx_null})
        os.chmod(refp, 0o444)
        (REF_DIR / f"{a.prompt}_text_context.json").write_text(json.dumps({"prompt": prompt, "source": a.exp, "mlx_video_commit": "87db56a",
                                                                          "shapes": res["context_shapes"], "dtype": res["context_dtype"]}, indent=2, ensure_ascii=False))
        res["frozen_reference"] = str(refp.relative_to(ROOT))
    else:
        ref = mx.load(str(refp))
        res["identical_to_ref"] = {k: bool(mx.array_equal(ref[k], v).item()) for k, v in (("context", ctx), ("context_null", ctx_null))}
        res["max_abs_vs_ref"] = {k: mx.abs(ref[k].astype(mx.float32) - v.astype(mx.float32)).max().item() for k, v in (("context", ctx), ("context_null", ctx_null))}
        res["ref"] = str(refp.relative_to(ROOT))
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
