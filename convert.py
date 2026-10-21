"""
convert.py -- stories260K.bin (llama2.c, little-endian fp32)
           -> stories260k.a5 (big-endian, int8 weights, Q16.16 tables)

Quantisation:
  * every weight matrix is int8, symmetric, with one scale per OUTPUT ROW
  * each row scale is a uint16 mantissa plus one per-tensor shift, so the
    rescale after a dot product is a 16x16 multiply and an arithmetic shift
  * activations are int8 with a power-of-two scale, so quantising a vector
    on the 68000 costs one shift per element and nothing else

Usage:  python3 convert.py stories260K.bin stories260k.a5
"""
import struct
import sys
import math
import numpy as np

import fx
from ref_fp32 import read_checkpoint

MAGIC = b"A5LM"
VERSION = 1


def q_tensor(w):
    """w: (rows, cols) fp32 -> (shift, scales uint16, int8 data)."""
    rows, cols = w.shape
    amax = np.abs(w).max(axis=1)
    amax[amax == 0] = 1e-30
    scale = amax / 127.0                       # real value of one int8 step
    smax = float(scale.max())
    shift = 0
    while smax * (1 << (shift + 1)) <= 65535.0 and shift < 62:
        shift += 1
    sfix = np.rint(scale * (1 << shift)).astype(np.int64)
    sfix = np.clip(sfix, 1, 65535).astype(np.uint16)
    real = sfix.astype(np.float64) / (1 << shift)
    q = np.rint(w / real[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return shift, sfix, q


def pack_qtensor(shift, sfix, q):
    rows, cols = q.shape
    out = struct.pack(">HHHH", shift, rows, cols, 0)
    out += sfix.astype(">u2").tobytes()
    body = q.tobytes()
    if len(body) & 1:
        body += b"\0"
    return out + body


def i32arr(a):
    return np.asarray(a, dtype=">i4").tobytes()


def to_q16(a):
    return np.rint(np.asarray(a, dtype=np.float64) * fx.ONE).astype(np.int64)


def convert(src, dst):
    cfg, w = read_checkpoint(src)
    print(cfg)
    hs2 = cfg.head_size // 2

    cos = np.zeros((cfg.seq_len, hs2))
    sin = np.zeros((cfg.seq_len, hs2))
    for pos in range(cfg.seq_len):
        for j in range(hs2):
            freq = 1.0 / (10000.0 ** ((2 * j) / cfg.head_size))
            cos[pos, j] = math.cos(pos * freq)
            sin[pos, j] = math.sin(pos * freq)

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack(">HH", VERSION, 1 if cfg.shared_classifier else 0)
    blob += struct.pack(">8H", cfg.dim, cfg.hidden_dim, cfg.n_layers,
                        cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size,
                        cfg.seq_len, cfg.head_size)
    blob += i32arr(to_q16(cos).ravel())
    blob += i32arr(to_q16(sin).ravel())
    blob += i32arr(fx.build_exp_table())
    blob += i32arr(fx.build_sigmoid_table())
    blob += i32arr(to_q16(w["rms_att"]).ravel())
    blob += i32arr(to_q16(w["rms_ffn"]).ravel())
    blob += i32arr(to_q16(w["rms_final"]).ravel())
    blob += pack_qtensor(*q_tensor(w["tok_emb"]))
    for l in range(cfg.n_layers):
        for k in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            blob += pack_qtensor(*q_tensor(w[k][l]))
    if not cfg.shared_classifier:
        blob += pack_qtensor(*q_tensor(w["wcls"]))

    with open(dst, "wb") as f:
        f.write(blob)
    print(f"wrote {dst}: {len(blob)} bytes ({len(blob)/1024:.1f} KB)")
    return cfg


# ------------------------------------------------------------------ reader

class QTensor:
    __slots__ = ("shift", "rows", "cols", "scale", "q")


class Model:
    pass


def load_a5(path):
    raw = open(path, "rb").read()
    assert raw[:4] == MAGIC, "bad magic"
    ver, flags = struct.unpack_from(">HH", raw, 4)
    assert ver == VERSION
    (dim, hidden_dim, n_layers, n_heads, n_kv_heads,
     vocab_size, seq_len, head_size) = struct.unpack_from(">8H", raw, 8)

    m = Model()
    m.dim, m.hidden_dim, m.n_layers = dim, hidden_dim, n_layers
    m.n_heads, m.n_kv_heads = n_heads, n_kv_heads
    m.vocab_size, m.seq_len, m.head_size = vocab_size, seq_len, head_size
    m.shared = bool(flags & 1)
    m.kv_dim = (dim * n_kv_heads) // n_heads
    off = 24
    hs2 = head_size // 2

    def ints(n):
        nonlocal off
        a = np.frombuffer(raw, dtype=">i4", count=n, offset=off).astype(np.int64)
        off += 4 * n
        return a

    m.rope_cos = ints(seq_len * hs2).reshape(seq_len, hs2)
    m.rope_sin = ints(seq_len * hs2).reshape(seq_len, hs2)
    m.exp_tab = list(ints(fx.EXP_N))
    m.sig_tab = list(ints(fx.EXP_N))
    m.rms_att = ints(n_layers * dim).reshape(n_layers, dim)
    m.rms_ffn = ints(n_layers * dim).reshape(n_layers, dim)
    m.rms_final = ints(dim)

    def qt():
        nonlocal off
        t = QTensor()
        t.shift, t.rows, t.cols, _ = struct.unpack_from(">HHHH", raw, off)
        off += 8
        t.scale = np.frombuffer(raw, dtype=">u2", count=t.rows,
                                offset=off).astype(np.int64)
        off += 2 * t.rows
        n = t.rows * t.cols
        t.q = np.frombuffer(raw, dtype=np.int8, count=n,
                            offset=off).reshape(t.rows, t.cols).astype(np.int64)
        off += n + (n & 1)
        return t

    m.tok_emb = qt()
    m.layers = []
    for _ in range(n_layers):
        m.layers.append({k: qt() for k in
                         ("wq", "wk", "wv", "wo", "w1", "w2", "w3")})
    m.wcls = m.tok_emb if m.shared else qt()
    assert off == len(raw), f"trailing bytes: {off} vs {len(raw)}"
    return m


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
