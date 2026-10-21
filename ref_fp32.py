"""
Reference: llama2.c checkpoint reader + fp32 forward pass.

Two jobs:
  1. parse the legacy llama2.c .bin format (little-endian fp32)
  2. run an fp32 forward pass, to compare against the quantised oracle

Also builds a synthetic checkpoint with stories260K's geometry so the whole
pipeline can be tested without the real weights.
"""
import struct
import numpy as np

HDR = "<7i"


class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                 vocab_size, seq_len, shared_classifier=True):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.shared_classifier = shared_classifier
        self.head_size = dim // n_heads
        self.kv_dim = (dim * n_kv_heads) // n_heads

    def __repr__(self):
        return (f"Config(dim={self.dim} hidden={self.hidden_dim} "
                f"layers={self.n_layers} heads={self.n_heads} "
                f"kv_heads={self.n_kv_heads} vocab={self.vocab_size} "
                f"seq={self.seq_len} head_size={self.head_size} "
                f"shared={self.shared_classifier})")


def read_checkpoint(path):
    """Parse a legacy llama2.c export. Returns (Config, dict of fp32 arrays)."""
    with open(path, "rb") as f:
        raw = f.read()
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
        struct.unpack_from(HDR, raw, 0)
    shared = vocab_size > 0
    vocab_size = abs(vocab_size)
    cfg = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                 vocab_size, seq_len, shared)

    data = np.frombuffer(raw, dtype="<f4", offset=struct.calcsize(HDR))
    pos = 0

    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape))
        out = data[pos:pos + n].reshape(shape).astype(np.float32)
        pos += n
        return out

    hs, kvd = cfg.head_size, cfg.kv_dim
    w = {}
    w["tok_emb"] = take(vocab_size, dim)
    w["rms_att"] = take(n_layers, dim)
    w["wq"] = take(n_layers, dim, dim)
    w["wk"] = take(n_layers, kvd, dim)
    w["wv"] = take(n_layers, kvd, dim)
    w["wo"] = take(n_layers, dim, dim)
    w["rms_ffn"] = take(n_layers, dim)
    w["w1"] = take(n_layers, hidden_dim, dim)
    w["w2"] = take(n_layers, dim, hidden_dim)
    w["w3"] = take(n_layers, hidden_dim, dim)
    w["rms_final"] = take(dim)
    pos += seq_len * hs // 2      # skip legacy freq_cis_real
    pos += seq_len * hs // 2      # skip legacy freq_cis_imag
    w["wcls"] = w["tok_emb"] if shared else take(vocab_size, dim)
    return cfg, w


def write_checkpoint(path, cfg, w):
    """Write arrays back out in legacy llama2.c format (for the synthetic test)."""
    with open(path, "wb") as f:
        v = cfg.vocab_size if cfg.shared_classifier else -cfg.vocab_size
        f.write(struct.pack(HDR, cfg.dim, cfg.hidden_dim, cfg.n_layers,
                            cfg.n_heads, cfg.n_kv_heads, v, cfg.seq_len))
        order = ["tok_emb", "rms_att", "wq", "wk", "wv", "wo",
                 "rms_ffn", "w1", "w2", "w3", "rms_final"]
        for k in order:
            f.write(np.asarray(w[k], dtype="<f4").tobytes())
        f.write(np.zeros(cfg.seq_len * cfg.head_size, dtype="<f4").tobytes())
        if not cfg.shared_classifier:
            f.write(np.asarray(w["wcls"], dtype="<f4").tobytes())


def make_synthetic(path, seed=1234):
    """stories260K geometry, random weights. Same shapes, so same code paths."""
    rng = np.random.default_rng(seed)
    cfg = Config(dim=64, hidden_dim=192, n_layers=5, n_heads=8, n_kv_heads=4,
                 vocab_size=512, seq_len=512, shared_classifier=True)
    hs, kvd = cfg.head_size, cfg.kv_dim

    def r(*shape, s=0.08):
        return rng.normal(0, s, shape).astype(np.float32)

    w = {
        "tok_emb": r(cfg.vocab_size, cfg.dim, s=0.10),
        "rms_att": (1.0 + rng.normal(0, 0.15, (cfg.n_layers, cfg.dim))).astype(np.float32),
        "wq": r(cfg.n_layers, cfg.dim, cfg.dim),
        "wk": r(cfg.n_layers, kvd, cfg.dim),
        "wv": r(cfg.n_layers, kvd, cfg.dim),
        "wo": r(cfg.n_layers, cfg.dim, cfg.dim),
        "rms_ffn": (1.0 + rng.normal(0, 0.15, (cfg.n_layers, cfg.dim))).astype(np.float32),
        "w1": r(cfg.n_layers, cfg.hidden_dim, cfg.dim),
        "w2": r(cfg.n_layers, cfg.dim, cfg.hidden_dim),
        "w3": r(cfg.n_layers, cfg.hidden_dim, cfg.dim),
        "rms_final": (1.0 + rng.normal(0, 0.10, cfg.dim)).astype(np.float32),
    }
    w["wcls"] = w["tok_emb"]
    write_checkpoint(path, cfg, w)
    return cfg, w


# ---------------------------------------------------------------- fp32 forward

def rmsnorm(x, weight, eps=1e-5):
    ss = float(np.mean(x * x)) + eps
    return weight * (x / np.sqrt(ss))


def softmax(x):
    m = float(np.max(x))
    e = np.exp(x - m)
    return e / np.sum(e)


class RunStateF:
    def __init__(self, cfg):
        self.kc = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
        self.vc = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)


def forward_fp32(cfg, w, st, token, pos):
    dim, hs, kvd = cfg.dim, cfg.head_size, cfg.kv_dim
    kv_mul = cfg.n_heads // cfg.n_kv_heads
    x = w["tok_emb"][token].copy()

    for l in range(cfg.n_layers):
        xb = rmsnorm(x, w["rms_att"][l])
        q = w["wq"][l] @ xb
        k = w["wk"][l] @ xb
        v = w["wv"][l] @ xb

        # RoPE
        for i in range(0, dim, 2):
            hd = i % hs
            freq = 1.0 / (10000.0 ** (hd / hs))
            val = pos * freq
            fcr, fci = np.cos(val), np.sin(val)
            rot = 2 if i < kvd else 1
            for vv in range(rot):
                vec = q if vv == 0 else k
                v0, v1 = vec[i], vec[i + 1]
                vec[i] = v0 * fcr - v1 * fci
                vec[i + 1] = v0 * fci + v1 * fcr

        st.kc[l, pos] = k
        st.vc[l, pos] = v

        xb2 = np.zeros(dim, dtype=np.float32)
        for h in range(cfg.n_heads):
            qh = q[h * hs:(h + 1) * hs]
            kvh = (h // kv_mul) * hs
            att = np.array([float(qh @ st.kc[l, t, kvh:kvh + hs]) / np.sqrt(hs)
                            for t in range(pos + 1)], dtype=np.float32)
            att = softmax(att)
            acc = np.zeros(hs, dtype=np.float32)
            for t in range(pos + 1):
                acc += att[t] * st.vc[l, t, kvh:kvh + hs]
            xb2[h * hs:(h + 1) * hs] = acc

        x = x + w["wo"][l] @ xb2

        xb = rmsnorm(x, w["rms_ffn"][l])
        h1 = w["w1"][l] @ xb
        h3 = w["w3"][l] @ xb
        h1 = h1 * (1.0 / (1.0 + np.exp(-h1)))
        x = x + w["w2"][l] @ (h1 * h3)

    x = rmsnorm(x, w["rms_final"])
    return w["wcls"] @ x


if __name__ == "__main__":
    cfg, w = make_synthetic("synthetic260k.bin")
    print(cfg)
    cfg2, w2 = read_checkpoint("synthetic260k.bin")
    print(cfg2)
    for k in w:
        assert np.allclose(w[k], w2[k]), k
    print("roundtrip ok")
    st = RunStateF(cfg2)
    for p, tok in enumerate([1, 5, 9, 42]):
        lg = forward_fp32(cfg2, w2, st, tok, p)
        print(f"pos {p}: argmax={int(np.argmax(lg))} max={lg.max():.4f}")
