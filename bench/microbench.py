"""Model-agnostic MLX microbenchmarks for the hardware profile and bound classification.

    python bench/microbench.py --exp exp-030 [--shapes shapes.json] [--label "..."]

Measures (median of --iters after warm-up, synchronised with mx.eval):
  peak matmul throughput per dtype (square N^3), memory bandwidth (elementwise over 1 GiB), op launch
  overhead, dense vs 8/4-bit quantized matmul at the given (M, K, N) shapes, and SDPA at given shapes.
shapes.json: {"linear": [[M, K, N, count_per_step, "name"], ...], "sdpa": [[B, H, L, D, count, "name"], ...]}
Run with MLX_METAL_GPU_ARCH set to another architecture to test whether arch-specific kernels (e.g. NAX
neural-accelerator matmuls on M5) are responsible for the throughput.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--shapes")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = ROOT / "runs" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    res = {"exp_id": a.exp, "kind": "microbench", "label": a.label, "mlx_version": mx.__version__,
           "env_MLX_METAL_GPU_ARCH": os.environ.get("MLX_METAL_GPU_ARCH"),
           "device": mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()}

    # peak matmul throughput
    peak = {}
    for dt_name, dt in (("float32", mx.float32), ("float16", mx.float16), ("bfloat16", mx.bfloat16)):
        for n in (2048, 4096, 8192):
            x, y = mx.random.normal((n, n)).astype(dt), mx.random.normal((n, n)).astype(dt)
            t = bench(lambda: x @ y, a.iters)
            peak[f"{dt_name}_{n}"] = {"s": t, "tflops": 2 * n**3 / t / 1e12}
            del x, y
    res["matmul_peak"] = peak

    # bandwidth: read 1 GiB + write 1 GiB
    x = mx.random.normal((256 * 1024 * 1024,)).astype(mx.float32)  # 1 GiB
    mx.eval(x)
    t = bench(lambda: x * 2.0, a.iters)
    res["bandwidth"] = {"elementwise_1GiB_s": t, "gbps_read_plus_write": 2 * x.nbytes / t / 1e9}
    del x

    # launch overhead: tiny op, one eval per op
    s = mx.array([1.0])
    t = bench(lambda: s + 1.0, iters=200, warmup=20)
    res["launch_overhead_s"] = t

    if a.shapes:
        shapes = json.loads((ROOT / a.shapes).read_text())
        lin = []
        for M, K, N, count, name in shapes.get("linear", []):
            xin = mx.random.normal((M, K)).astype(mx.bfloat16)
            w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
            mx.eval(xin, w)
            row = {"name": name, "M": M, "K": K, "N": N, "count_per_step": count, "flops": 2 * M * K * N}
            t = bench(lambda: xin @ w.T, a.iters)
            row["bf16"] = {"s": t, "tflops": row["flops"] / t / 1e12, "weight_gbps": w.nbytes / t / 1e9}
            for bits in (8, 4):
                wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
                mx.eval(wq, sc, bi)
                t = bench(lambda: mx.quantized_matmul(xin, wq, sc, bi, transpose=True, group_size=64, bits=bits), a.iters)
                row[f"q{bits}"] = {"s": t, "tflops": row["flops"] / t / 1e12}
            lin.append(row)
            del xin, w
        res["linear"] = lin
        sd = []
        for B, H, L, D, count, name in shapes.get("sdpa", []):
            q, k, v = (mx.random.normal((B, H, L, D)).astype(mx.bfloat16) for _ in range(3))
            mx.eval(q, k, v)
            t = bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5), a.iters)
            fl = 4 * B * H * L * L * D
            sd.append({"name": name, "B": B, "H": H, "L": L, "D": D, "count_per_step": count, "s": t, "tflops": fl / t / 1e12})
        res["sdpa"] = sd
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("matmul_peak", "bandwidth", "launch_overhead_s")}, indent=1))
    for r in res.get("linear", []):
        print(f"{r['name']:28s} M{r['M']} K{r['K']} N{r['N']}  bf16 {r['bf16']['s']*1e3:7.2f} ms {r['bf16']['tflops']:5.2f} TF  "
              f"q8 {r['q8']['s']*1e3:7.2f} ms  q4 {r['q4']['s']*1e3:7.2f} ms")
    for r in res.get("sdpa", []):
        print(f"{r['name']:28s} L{r['L']} sdpa {r['s']*1e3:7.2f} ms {r['tflops']:5.2f} TF")


if __name__ == "__main__":
    main()
