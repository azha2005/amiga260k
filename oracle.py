"""
oracle.py -- the integer forward pass, in Python.

This is the specification. Whatever the C and the 68000 assembly do, they must
produce these exact int32 logits. Every operation is Q16.16 fixed point using
only the primitives in fx.py.
"""
import sys
import numpy as np

import fx
from convert import load_a5


def matmul(t, xq, xsh, out):
    """out[d] = sum_n W[d,n] * x[n], in Q16.16. The hot loop."""
    sh = t.shift - xsh
    q = t.q
    sc = t.scale
    for d in range(t.rows):
        acc = int(np.dot(q[d], xq))          # exact int32 accumulation
        out[d] = fx.mulshift(acc, int(sc[d]), sh)


def rmsnorm(x, weight, n):
    ss = 0
    for j in range(n):
        ss += fx.fxmul(x[j], x[j])
    ss = fx.i32(ss // n + 1)                 # +1 == eps, one Q16.16 unit
    inv = fx.fxrsqrt(ss)
    return [fx.fxmul(weight[j], fx.fxmul(x[j], inv)) for j in range(n)]


def softmax_inplace(a, n, tab):
    m = max(a[:n])
    s = 0
    for i in range(n):
        a[i] = fx.fxexp_neg(tab, m - a[i])
        s += a[i]
    s = fx.i32(s)
    inv = fx.fxdiv(fx.ONE, s)          # one division, not n of them
    for i in range(n):
        a[i] = fx.fxmul(a[i], inv)


class State:
    def __init__(self, m):
        self.kc = [[[0] * m.kv_dim for _ in range(m.seq_len)]
                   for _ in range(m.n_layers)]
        self.vc = [[[0] * m.kv_dim for _ in range(m.seq_len)]
                   for _ in range(m.n_layers)]


def forward(m, st, token, pos):
    dim, hs, kvd = m.dim, m.head_size, m.kv_dim
    kv_mul = m.n_heads // m.n_kv_heads
    inv_sqrt_hs = fx.fxdiv(fx.ONE, fx.isqrt32(hs << 16) << 8)   # same formula as the C

    e = m.tok_emb
    x = [fx.mulshift(int(e.q[token][j]), int(e.scale[token]), e.shift - 16)
         for j in range(dim)]

    q = [0] * dim
    k = [0] * kvd
    v = [0] * kvd
    xb2 = [0] * dim
    h1 = [0] * m.hidden_dim
    h3 = [0] * m.hidden_dim

    for l in range(m.n_layers):
        L = m.layers[l]
        xb = rmsnorm(x, m.rms_att[l], dim)
        xq, xsh = fx.quantize_act(xb)
        xqa = np.array(xq, dtype=np.int64)
        matmul(L["wq"], xqa, xsh, q)
        matmul(L["wk"], xqa, xsh, k)
        matmul(L["wv"], xqa, xsh, v)

        for i in range(0, dim, 2):
            j = (i % hs) >> 1
            c = int(m.rope_cos[pos][j])
            s = int(m.rope_sin[pos][j])
            v0, v1 = q[i], q[i + 1]
            q[i] = fx.i32(fx.fxmul(v0, c) - fx.fxmul(v1, s))
            q[i + 1] = fx.i32(fx.fxmul(v0, s) + fx.fxmul(v1, c))
            if i < kvd:
                v0, v1 = k[i], k[i + 1]
                k[i] = fx.i32(fx.fxmul(v0, c) - fx.fxmul(v1, s))
                k[i + 1] = fx.i32(fx.fxmul(v0, s) + fx.fxmul(v1, c))

        st.kc[l][pos] = list(k)
        st.vc[l][pos] = list(v)

        att = [0] * (pos + 1)
        for h in range(m.n_heads):
            qo = h * hs
            ko = (h // kv_mul) * hs
            for t in range(pos + 1):
                kt = st.kc[l][t]
                sc = 0
                for j in range(hs):
                    sc += fx.fxmul(q[qo + j], kt[ko + j])
                att[t] = fx.fxmul(fx.i32(sc), inv_sqrt_hs)
            softmax_inplace(att, pos + 1, m.exp_tab)
            for j in range(hs):
                acc = 0
                for t in range(pos + 1):
                    acc += fx.fxmul(att[t], st.vc[l][t][ko + j])
                xb2[qo + j] = fx.i32(acc)

        bq, bsh = fx.quantize_act(xb2)
        tmp = [0] * dim
        matmul(L["wo"], np.array(bq, dtype=np.int64), bsh, tmp)
        for j in range(dim):
            x[j] = fx.i32(x[j] + tmp[j])

        xb = rmsnorm(x, m.rms_ffn[l], dim)
        xq, xsh = fx.quantize_act(xb)
        xqa = np.array(xq, dtype=np.int64)
        matmul(L["w1"], xqa, xsh, h1)
        matmul(L["w3"], xqa, xsh, h3)
        for j in range(m.hidden_dim):
            g = fx.fxmul(h1[j], fx.fxsigmoid(m.sig_tab, h1[j]))
            h1[j] = fx.fxmul(g, h3[j])
        hq, hsh = fx.quantize_act(h1)
        matmul(L["w2"], np.array(hq, dtype=np.int64), hsh, tmp)
        for j in range(dim):
            x[j] = fx.i32(x[j] + tmp[j])

    x = rmsnorm(x, m.rms_final, dim)
    xq, xsh = fx.quantize_act(x)
    logits = [0] * m.vocab_size
    matmul(m.wcls, np.array(xq, dtype=np.int64), xsh, logits)
    return logits


if __name__ == "__main__":
    m = load_a5(sys.argv[1] if len(sys.argv) > 1 else "stories260k.a5")
    st = State(m)
    toks = [int(a) for a in sys.argv[2:]] or [1, 5, 9, 42]
    for p, t in enumerate(toks):
        lg = forward(m, st, t, p)
        print(f"pos {p} tok {t}: argmax={int(np.argmax(lg))} "
              f"top={max(lg)} sum={sum(lg) & 0xFFFFFFFF:08x}")
