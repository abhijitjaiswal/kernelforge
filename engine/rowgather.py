"""Model-agnostic exact embedding lookup straight from a safetensors file (MLX).

An embedding table is (vocab, dim), but one forward pass reads only the rows of the token ids it sees.
gather_rows memory-maps the file and copies just those rows, so the table never becomes resident.
Bit-identical to table[ids] (same bytes, no arithmetic).
"""

import json
import struct

import mlx.core as mx
import numpy as np

_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}
_MX = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


def tensor_view(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header[key]
    start, end = meta["data_offsets"]
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + start, shape=(end - start,))
    return mm.view(_NP[meta["dtype"]]).reshape(meta["shape"]), meta["dtype"]


def gather_rows(path, key, ids):
    table, dtype = tensor_view(path, key)
    flat = np.asarray(ids).reshape(-1)
    rows = np.ascontiguousarray(table[flat])          # only these rows are read from disk / page cache
    out = mx.array(rows)
    if dtype == "BF16":
        out = out.view(mx.bfloat16)
    return out.reshape(*np.asarray(ids).shape, table.shape[-1])
