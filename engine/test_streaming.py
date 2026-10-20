"""Streamed vs resident: a toy residual MLP stack must give bit-identical outputs, with and without prefetch.

    python -m engine.test_streaming
"""
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx import nn

from engine.streaming import BlockCompiler, WeightStreamer, save_block_files


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a, self.b = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def __call__(self, x):
        return x + self.b(nn.gelu(self.a(x)))


class Net(nn.Module):
    def __init__(self, d=64, n=6):
        super().__init__()
        self.blocks = [Block(d) for _ in range(n)]

    def __call__(self, x):
        for b in self.blocks:
            x = b(x=x)
        return x


def main():
    mx.random.seed(0)
    net = Net()
    x = mx.random.normal((8, 64))
    ref = [net(x), net(x * 0.5)]
    mx.eval(ref)
    tree_unflatten_params = net.parameters()
    with tempfile.TemporaryDirectory() as d:
        files = [Path(d) / f"{i}.safetensors" for i in range(len(net.blocks))]
        save_block_files(net.blocks, files)
        bc = BlockCompiler(net.blocks)
        bc.attach()
        cref = [net(x), net(x * 0.5)]
        mx.eval(cref)
        bc.detach()
        print("block-compiled resident vs eager: max abs diff", max(mx.abs(a - b).max().item() for a, b in zip(cref, ref)))
        net.update(tree_unflatten_params)  # restore real weights (tracing installed tracers)
        for prefetch, comp in ((False, False), (True, False), (True, True)):
            s = WeightStreamer(net.blocks, files, prefetch=prefetch, compile=comp)
            s.release_all()
            s.attach()
            outs = [net(x), net(x * 0.5)]  # two "steps": exercises the cyclic prefetch
            mx.eval(outs)
            s.detach()
            want = cref if comp else ref
            same = all(mx.array_equal(o, r).item() for o, r in zip(outs, want))
            empty = all(v.size == 0 for b in net.blocks for _, v in __import__("mlx.utils").utils.tree_flatten(b.parameters()))
            print(f"prefetch={prefetch} compile={comp}: bit-identical to {'block-compiled' if comp else 'eager'} resident={same}  blocks released after run={empty}  block calls={len(s.stats)}")
            assert same and empty


if __name__ == "__main__":
    main()
