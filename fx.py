"""
Fixed-point primitives, Q16.16 throughout.

Every routine here is written the way the 68000 will execute it: no 64-bit
types, no floating point, all products built from 16x16->32 multiplies (MULU).
The C in llama68k.c mirrors this line for line, and test_bitexact.py asserts
the two agree exactly.
"""

ONE = 1 << 16          # 1.0 in Q16.16
EXP_BITS = 6           # exp table spacing = 1/64
EXP_MAX = 16           # table covers u in [0, 16]
EXP_N = (EXP_MAX << EXP_BITS) + 1


def i32(v):
    """Wrap to signed 32-bit, the way a 68000 D-register does."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def fxmul(a, b):
    """Q16.16 * Q16.16 -> Q16.16, built from four 16x16 multiplies."""
    neg = 0
    if a < 0:
        a = -a
        neg ^= 1
    if b < 0:
        b = -b
        neg ^= 1
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    ah, al = a >> 16, a & 0xFFFF
    bh, bl = b >> 16, b & 0xFFFF
    r = ((ah * bh) << 16) + ah * bl + al * bh + ((al * bl) >> 16)
    r &= 0xFFFFFFFF
    return i32(-r if neg else r)


def mulshift(acc, m, sh):
    """(acc * m) >> sh with a 48-bit intermediate. acc int32, m uint16.

    sh may be negative, meaning shift left. This is the matmul rescale, run
    once per output element, so two MULUs here cost nothing next to the dot
    product itself.
    """
    neg = 0
    if acc < 0:
        acc = -acc
        neg = 1
    lo = (acc & 0xFFFF) * m
    hi = (acc >> 16) * m
    full = (hi << 16) + lo            # 48 bits, exact
    r = (full >> sh) if sh >= 0 else (full << -sh)
    r &= 0xFFFFFFFF
    return i32(-r if neg else r)


def isqrt32(v):
    """Integer square root, restoring shift-subtract. 16 iterations."""
    if v <= 0:
        return 0
    rem = 0
    root = 0
    for i in range(16):
        root <<= 1
        rem = (rem << 2) | ((v >> 30) & 3)
        v = (v << 2) & 0xFFFFFFFF
        if root < rem:
            root += 1
            rem -= root
            root += 1
    return root >> 1


def fxdiv(a, b):
    """(a << 16) / b in Q16.16, long division on a 64-bit-free numerator."""
    if b == 0:
        return 0x7FFFFFFF
    neg = 0
    if a < 0:
        a = -a
        neg ^= 1
    if b < 0:
        b = -b
        neg ^= 1
    nh, nl = a >> 16, (a << 16) & 0xFFFFFFFF
    q = 0
    rem = 0
    for i in range(64):
        bit = (nh >> 31) & 1 if i < 32 else (nl >> (63 - i)) & 1
        if i < 32:
            nh = (nh << 1) & 0xFFFFFFFF
        rem = (rem << 1) | bit
        q <<= 1
        if rem >= b:
            rem -= b
            q |= 1
    q &= 0xFFFFFFFF
    return i32(-q if neg else q)


def fxrsqrt(ss):
    """1/sqrt(ss) in Q16.16. isqrt estimate plus one Newton refinement."""
    if ss <= 0:
        return 0
    r = isqrt32(ss) << 8                       # sqrt(ss) in Q16.16, coarse
    if r == 0:
        return 0x7FFFFFFF
    r = (r + fxdiv(ss, r)) >> 1                # Newton on sqrt
    if r == 0:
        return 0x7FFFFFFF
    return fxdiv(ONE, r)


def build_exp_table():
    """exp(-u) in Q16.16 for u = 0 .. EXP_MAX, spaced 1/64. Shipped in the model."""
    import math
    return [int(round(math.exp(-i / (1 << EXP_BITS)) * ONE)) for i in range(EXP_N)]


def fxexp_neg(tab, u):
    """exp(-u) for u >= 0, Q16.16, linear interpolation between table entries."""
    if u <= 0:
        return ONE
    if u >= (EXP_MAX << 16):
        return 0
    idx = u >> (16 - EXP_BITS)
    frac = (u << EXP_BITS) & 0xFFFF
    a = tab[idx]
    b = tab[idx + 1]
    return i32(a + (((b - a) * frac) >> 16))


def build_sigmoid_table():
    """sigmoid(u) in Q16.16 for u = 0 .. EXP_MAX. Shipped in the model so the
    68000 never has to divide inside SiLU."""
    import math
    return [int(round((1.0 / (1.0 + math.exp(-i / (1 << EXP_BITS)))) * ONE))
            for i in range(EXP_N)]


def fxsigmoid(tab, x):
    """1/(1+exp(-x)) in Q16.16, table lookup only. tab is the sigmoid table."""
    neg = x < 0
    if neg:
        x = -x
    if x >= (EXP_MAX << 16):
        v = ONE
    else:
        idx = x >> (16 - EXP_BITS)
        frac = (x << EXP_BITS) & 0xFFFF
        a = tab[idx]
        b = tab[idx + 1]
        v = i32(a + (((b - a) * frac) >> 16))
    return i32(ONE - v) if neg else v


def quantize_act(x):
    """Q16.16 vector -> (int8 list, shift). Scale is a power of two, so the
    only work per element on the 68000 is an arithmetic shift."""
    amax = 0
    for v in x:
        a = -v if v < 0 else v
        if a > amax:
            amax = a
    if amax == 0:
        return [0] * len(x), 0
    sh = 0
    while (amax >> sh) > 127:
        sh += 1
    out = []
    for v in x:
        q = v >> sh if v >= 0 else -((-v) >> sh)
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        out.append(q)
    return out, sh
