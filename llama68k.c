/* llama68k.c -- stories260K inference for the Amiga 500 (68000, no FPU)
 *
 * Everything is Q16.16 fixed point. No floating point, no 64-bit types, no
 * libm. Every product is built from 16x16->32 multiplies so it maps straight
 * onto MULU/MULS.
 *
 * Host test build (little-endian, C matmul):
 *   gcc -O2 -DA5_LITTLE_ENDIAN -o llama68k llama68k.c
 * Amiga build: see build.sh
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef signed char    s8;
typedef unsigned char  u8;
typedef short          s16;
typedef unsigned short u16;

/* s32 must be exactly 32 bits. On the Amiga that is `long` (68000 C compilers
 * use 32-bit int and 32-bit long). On a modern 64-bit host `long` is 64 bits,
 * which would silently break every wraparound the fixed-point math relies on,
 * so the host test build uses `int`. */
#ifdef A5_HOST
typedef int            s32;
typedef unsigned int   u32;
#else
typedef long           s32;
typedef unsigned long  u32;
#endif

#define ONE        65536L
#define EXP_BITS   6
#define EXP_MAX    16
#define EXP_N      ((EXP_MAX << EXP_BITS) + 1)

#ifdef A5_HOST
static u16 be16(const void *p){ const u8*b=(const u8*)p; return (u16)((b[0]<<8)|b[1]); }
static s32 be32(const void *p){ const u8*b=(const u8*)p;
    return (s32)(((u32)b[0]<<24)|((u32)b[1]<<16)|((u32)b[2]<<8)|b[3]); }
static void fixup32(s32 *a, long n){ long i; for(i=0;i<n;i++) a[i]=be32(&a[i]); }
static void fixup16(u16 *a, long n){ long i; for(i=0;i<n;i++) a[i]=be16(&a[i]); }
#else
#define be16(p)  (*(const u16*)(p))
#define be32(p)  (*(const s32*)(p))
#define fixup32(a,n) ((void)0)
#define fixup16(a,n) ((void)0)
#endif

/* ---------------------------------------------------------- fixed point */

/* Q16.16 * Q16.16 -> Q16.16, four 16x16 multiplies. */
s32 fxmul(s32 a, s32 b)
{
    int neg = 0;
    u32 ua, ub, ah, al, bh, bl, r;
    if (a < 0) { a = -a; neg ^= 1; }
    if (b < 0) { b = -b; neg ^= 1; }
    ua = (u32)a; ub = (u32)b;
    ah = ua >> 16; al = ua & 0xFFFFUL;
    bh = ub >> 16; bl = ub & 0xFFFFUL;
    r = ((ah * bh) << 16) + ah * bl + al * bh + ((al * bl) >> 16);
    return neg ? -(s32)r : (s32)r;
}

/* (acc * m) >> sh with an exact 48-bit intermediate. sh<0 shifts left.
 * Runs once per matmul output element, not per MAC. */
s32 mulshift(s32 acc, u16 m, int sh)
{
    int neg = 0;
    u32 ua, lo, hi, rl, rh;
    if (acc < 0) { acc = -acc; neg = 1; }
    ua = (u32)acc;
    lo = (ua & 0xFFFFUL) * m;
    hi = (ua >> 16) * m;
    rl = lo + (hi << 16);
    rh = (hi >> 16) + ((rl < lo) ? 1UL : 0UL);
    if (sh >= 0) {
        rl = (sh == 0) ? rl : ((rl >> sh) | (rh << (32 - sh)));
    } else {
        rl = rl << (-sh);
    }
    return neg ? -(s32)rl : (s32)rl;
}

static u16 isqrt32(u32 v)
{
    u32 rem = 0, root = 0;
    int i;
    for (i = 0; i < 16; i++) {
        root <<= 1;
        rem = (rem << 2) | ((v >> 30) & 3);
        v <<= 2;
        if (root < rem) { root++; rem -= root; root++; }
    }
    return (u16)(root >> 1);
}

/* (a << 16) / b in Q16.16, restoring long division over a 64-bit numerator
 * held as two 32-bit halves. */
s32 fxdiv(s32 a, s32 b)
{
    int neg = 0, i;
    u32 nh, nl, q = 0, rem = 0, bit;
    if (b == 0) return 0x7FFFFFFFL;
    if (a < 0) { a = -a; neg ^= 1; }
    if (b < 0) { b = -b; neg ^= 1; }
    nh = (u32)a >> 16;
    nl = (u32)a << 16;
    for (i = 0; i < 64; i++) {
        if (i < 32) { bit = (nh >> 31) & 1; nh <<= 1; }
        else        { bit = (nl >> (63 - i)) & 1; }
        rem = (rem << 1) | bit;
        q <<= 1;
        if (rem >= (u32)b) { rem -= (u32)b; q |= 1; }
    }
    return neg ? -(s32)q : (s32)q;
}

s32 fxrsqrt(s32 ss)
{
    s32 r;
    if (ss <= 0) return 0;
    r = (s32)isqrt32((u32)ss) << 8;
    if (r == 0) return 0x7FFFFFFFL;
    r = (r + fxdiv(ss, r)) >> 1;
    if (r == 0) return 0x7FFFFFFFL;
    return fxdiv(ONE, r);
}

/* -------------------------------------------------------------- model */

typedef struct { u16 shift, rows, cols; u16 pad; u16 *scale; s8 *q; } QT;

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len, head_size, kv_dim, shared;
    s32 *rope_cos, *rope_sin, *exp_tab, *sig_tab;
    s32 *rms_att, *rms_ffn, *rms_final;
    QT tok_emb, wcls;
    QT *L;                       /* n_layers * 7, order wq wk wv wo w1 w2 w3 */
    u8 *blob;
} Model;

#define WQ 0
#define WK 1
#define WV 2
#define WO 3
#define W1 4
#define W2 5
#define W3 6

static u8 *cursor;

static s32 *take_i32(long n)
{
    s32 *p = (s32 *)cursor;
    cursor += 4 * n;
    fixup32(p, n);
    return p;
}

static void take_qt(QT *t)
{
    long n;
    t->shift = be16(cursor);
    t->rows  = be16(cursor + 2);
    t->cols  = be16(cursor + 4);
    cursor += 8;
    t->scale = (u16 *)cursor;
    fixup16(t->scale, t->rows);
    cursor += 2L * t->rows;
    n = (long)t->rows * t->cols;
    t->q = (s8 *)cursor;
    cursor += n + (n & 1);
}

int load_model(Model *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    long size, i, j;
    u16 flags;
    if (!f) return 0;
    fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
    m->blob = (u8 *)malloc(size);
    if (!m->blob) { fclose(f); return 0; }
    if (fread(m->blob, 1, size, f) != (size_t)size) { fclose(f); return 0; }
    fclose(f);
    if (memcmp(m->blob, "A5LM", 4) != 0) return 0;

    flags          = be16(m->blob + 6);
    m->dim         = be16(m->blob + 8);
    m->hidden_dim  = be16(m->blob + 10);
    m->n_layers    = be16(m->blob + 12);
    m->n_heads     = be16(m->blob + 14);
    m->n_kv_heads  = be16(m->blob + 16);
    m->vocab_size  = be16(m->blob + 18);
    m->seq_len     = be16(m->blob + 20);
    m->head_size   = be16(m->blob + 22);
    m->shared      = flags & 1;
    m->kv_dim      = (m->dim * m->n_kv_heads) / m->n_heads;

    cursor = m->blob + 24;
    m->rope_cos  = take_i32((long)m->seq_len * (m->head_size / 2));
    m->rope_sin  = take_i32((long)m->seq_len * (m->head_size / 2));
    m->exp_tab   = take_i32(EXP_N);
    m->sig_tab   = take_i32(EXP_N);
    m->rms_att   = take_i32((long)m->n_layers * m->dim);
    m->rms_ffn   = take_i32((long)m->n_layers * m->dim);
    m->rms_final = take_i32(m->dim);
    take_qt(&m->tok_emb);
    m->L = (QT *)malloc(sizeof(QT) * m->n_layers * 7);
    for (i = 0; i < m->n_layers; i++)
        for (j = 0; j < 7; j++)
            take_qt(&m->L[i * 7 + j]);
    if (m->shared) m->wcls = m->tok_emb; else take_qt(&m->wcls);
    return 1;
}

/* ------------------------------------------------------------ kernels */

/* x is sign-extended to 16-bit words once per matmul, not once per row: it
 * removes a MOVE.B and an EXT.W from every multiply-accumulate. */
#define MAXCOLS 512
static s16 xexp[MAXCOLS];

#ifdef USE_ASM
extern s32 dot_i8_x16(const s8 *w, const s16 *x, int n);   /* matmul68k.s */
#else
static s32 dot_i8_x16(const s8 *w, const s16 *x, int n)
{
    s32 acc = 0;
    int i;
    for (i = 0; i < n; i++) acc += (s32)w[i] * (s32)x[i];
    return acc;
}
#endif

static void matmul(const QT *t, const s8 *xq, int xsh, s32 *out)
{
    int sh = (int)t->shift - xsh;
    int d, i, cols = t->cols;
    const s8 *w = t->q;
    for (i = 0; i < cols; i++) xexp[i] = (s16)xq[i];
    for (d = 0; d < t->rows; d++, w += cols)
        out[d] = mulshift(dot_i8_x16(w, xexp, cols), t->scale[d], sh);
}

/* Q16.16 vector -> int8 with a power-of-two scale: one shift per element. */
static int quantize_act(const s32 *x, int n, s8 *out)
{
    s32 amax = 0, a;
    int i, sh = 0;
    for (i = 0; i < n; i++) { a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }
    if (amax == 0) { for (i = 0; i < n; i++) out[i] = 0; return 0; }
    while ((amax >> sh) > 127) sh++;
    for (i = 0; i < n; i++) {
        s32 q = x[i] >= 0 ? (x[i] >> sh) : -((-x[i]) >> sh);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        out[i] = (s8)q;
    }
    return sh;
}

static void rmsnorm(s32 *out, const s32 *x, const s32 *w, int n)
{
    s32 ss = 0, inv;
    int j;
    for (j = 0; j < n; j++) ss += fxmul(x[j], x[j]);
    ss = ss / n + 1;
    inv = fxrsqrt(ss);
    for (j = 0; j < n; j++) out[j] = fxmul(w[j], fxmul(x[j], inv));
}

static s32 fxexp_neg(const s32 *tab, s32 u)
{
    s32 a, b, frac;
    long idx;
    if (u <= 0) return ONE;
    if (u >= ((s32)EXP_MAX << 16)) return 0;
    idx  = u >> (16 - EXP_BITS);
    frac = (u << EXP_BITS) & 0xFFFFL;
    a = tab[idx]; b = tab[idx + 1];
    return a + (((b - a) * frac) >> 16);
}

/* sigmoid straight from its own table: no division inside SiLU. */
static s32 fxsigmoid(const s32 *tab, s32 x)
{
    s32 a, b, frac, v;
    long idx;
    int neg = x < 0;
    if (neg) x = -x;
    if (x >= ((s32)EXP_MAX << 16)) v = ONE;
    else {
        idx  = x >> (16 - EXP_BITS);
        frac = (x << EXP_BITS) & 0xFFFFL;
        a = tab[idx]; b = tab[idx + 1];
        v = a + (((b - a) * frac) >> 16);
    }
    return neg ? (ONE - v) : v;
}

static void softmax(s32 *a, int n, const s32 *tab)
{
    s32 mx = a[0], s = 0, inv;
    int i;
    for (i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    for (i = 0; i < n; i++) { a[i] = fxexp_neg(tab, mx - a[i]); s += a[i]; }
    inv = fxdiv(ONE, s);
    for (i = 0; i < n; i++) a[i] = fxmul(a[i], inv);
}

/* -------------------------------------------------------------- state */

typedef struct {
    s32 *x, *xb, *xb2, *q, *k, *v, *h1, *h3, *att, *logits, *tmp;
    s8  *xq, *hq;
    s32 *kc, *vc;
    int n_ctx;                     /* positions the KV cache was sized for */
} State;

/* n_ctx is the number of positions this run will actually reach. The KV cache
   dwarfs every other allocation -- at seq_len 512 it is 640 KB against 294 KB
   for the weights -- so sizing it for the run is what lets a 512K + A501
   machine hold the model at all. Nothing about the arithmetic changes: the
   cache stride becomes n_ctx instead of seq_len, and pos never reaches it. */
static void state_init(State *s, Model *m, int n_ctx)
{
    long kv = (long)m->n_layers * n_ctx * m->kv_dim;
    s->n_ctx = n_ctx;
    s->x   = (s32*)calloc(m->dim, 4);      s->xb  = (s32*)calloc(m->dim, 4);
    s->xb2 = (s32*)calloc(m->dim, 4);      s->q   = (s32*)calloc(m->dim, 4);
    s->tmp = (s32*)calloc(m->dim, 4);
    s->k   = (s32*)calloc(m->kv_dim, 4);   s->v   = (s32*)calloc(m->kv_dim, 4);
    s->h1  = (s32*)calloc(m->hidden_dim,4);s->h3  = (s32*)calloc(m->hidden_dim,4);
    s->att = (s32*)calloc(n_ctx, 4);
    s->logits = (s32*)calloc(m->vocab_size, 4);
    s->xq  = (s8*)calloc(m->hidden_dim + m->dim, 1);
    s->hq  = (s8*)calloc(m->hidden_dim, 1);
    s->kc  = (s32*)calloc(kv, 4);          s->vc  = (s32*)calloc(kv, 4);
}

s32 *forward(Model *m, State *s, int token, int pos)
{
    int dim = m->dim, hs = m->head_size, kvd = m->kv_dim;
    int kv_mul = m->n_heads / m->n_kv_heads;
    s32 inv_sqrt_hs = fxdiv(ONE, (s32)isqrt32((u32)hs << 16) << 8);
    long l, i, j, h, t, loff;
    int xsh;

    for (j = 0; j < dim; j++)
        s->x[j] = mulshift((s32)m->tok_emb.q[(long)token * dim + j],
                           m->tok_emb.scale[token], (int)m->tok_emb.shift - 16);

    for (l = 0; l < m->n_layers; l++) {
        QT *L = &m->L[l * 7];
        s32 *kcl = s->kc + (l * (long)s->n_ctx + pos) * (long)kvd;
        s32 *vcl = s->vc + (l * (long)s->n_ctx + pos) * (long)kvd;

        rmsnorm(s->xb, s->x, m->rms_att + l * (long)dim, dim);
        xsh = quantize_act(s->xb, dim, s->xq);
        matmul(&L[WQ], s->xq, xsh, s->q);
        matmul(&L[WK], s->xq, xsh, s->k);
        matmul(&L[WV], s->xq, xsh, s->v);

        for (i = 0; i < dim; i += 2) {
            long jj = (i % hs) >> 1;
            s32 c = m->rope_cos[pos * (long)(hs / 2) + jj];
            s32 sn = m->rope_sin[pos * (long)(hs / 2) + jj];
            s32 v0 = s->q[i], v1 = s->q[i + 1];
            s->q[i]     = fxmul(v0, c) - fxmul(v1, sn);
            s->q[i + 1] = fxmul(v0, sn) + fxmul(v1, c);
            if (i < kvd) {
                v0 = s->k[i]; v1 = s->k[i + 1];
                s->k[i]     = fxmul(v0, c) - fxmul(v1, sn);
                s->k[i + 1] = fxmul(v0, sn) + fxmul(v1, c);
            }
        }
        for (j = 0; j < kvd; j++) { kcl[j] = s->k[j]; vcl[j] = s->v[j]; }

        loff = l * (long)s->n_ctx * kvd;
        for (h = 0; h < m->n_heads; h++) {
            long qo = h * (long)hs, ko = (h / kv_mul) * (long)hs;
            for (t = 0; t <= pos; t++) {
                s32 *kt = s->kc + loff + t * (long)kvd;
                s32 sc = 0;
                for (j = 0; j < hs; j++) sc += fxmul(s->q[qo + j], kt[ko + j]);
                s->att[t] = fxmul(sc, inv_sqrt_hs);
            }
            softmax(s->att, (int)pos + 1, m->exp_tab);
            for (j = 0; j < hs; j++) {
                s32 acc = 0;
                for (t = 0; t <= pos; t++)
                    acc += fxmul(s->att[t], s->vc[loff + t * (long)kvd + ko + j]);
                s->xb2[qo + j] = acc;
            }
        }

        xsh = quantize_act(s->xb2, dim, s->xq);
        matmul(&L[WO], s->xq, xsh, s->tmp);
        for (j = 0; j < dim; j++) s->x[j] += s->tmp[j];

        rmsnorm(s->xb, s->x, m->rms_ffn + l * (long)dim, dim);
        xsh = quantize_act(s->xb, dim, s->xq);
        matmul(&L[W1], s->xq, xsh, s->h1);
        matmul(&L[W3], s->xq, xsh, s->h3);
        for (j = 0; j < m->hidden_dim; j++) {
            s32 g = fxmul(s->h1[j], fxsigmoid(m->sig_tab, s->h1[j]));
            s->h1[j] = fxmul(g, s->h3[j]);
        }
        xsh = quantize_act(s->h1, m->hidden_dim, s->hq);
        matmul(&L[W2], s->hq, xsh, s->tmp);
        for (j = 0; j < dim; j++) s->x[j] += s->tmp[j];
    }

    rmsnorm(s->xb, s->x, m->rms_final, dim);
    xsh = quantize_act(s->xb, dim, s->xq);
    matmul(&m->wcls, s->xq, xsh, s->logits);
    return s->logits;
}

/* ------------------------------------------------------------ sampler */

static u32 rng_state = 1;
static u32 rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int sample(s32 *logits, int n, s32 temp, const s32 *tab)
{
    int i, best = 0;
    s32 cdf = 0, r;
    if (temp == 0) {
        for (i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (i = 0; i < n; i++) logits[i] = fxdiv(logits[i], temp);
    softmax(logits, n, tab);
    r = (s32)(rnd() >> 16);
    for (i = 0; i < n; i++) { cdf += logits[i]; if (r < cdf) return i; }
    return n - 1;
}

/* ---------------------------------------------------------- tokenizer */

typedef struct { char **word; int n; } Vocab;

static int load_vocab(Vocab *v, const char *path, int n)
{
    FILE *f = fopen(path, "rb");
    int i, len;
    u8 hdr[4];
    if (!f) return 0;
    v->n = n;
    v->word = (char **)calloc(n, sizeof(char *));
    if (fread(hdr, 1, 4, f) != 4) { fclose(f); return 0; }
    for (i = 0; i < n; i++) {
        if (fread(hdr, 1, 2, f) != 2) break;
        len = (hdr[0] << 8) | hdr[1];
        v->word[i] = (char *)malloc(len + 1);
        if (fread(v->word[i], 1, len, f) != (size_t)len) break;
        v->word[i][len] = 0;
    }
    fclose(f);
    return 1;
}

static void emit(Vocab *v, int prev, int tok)
{
    char *p = v->word[tok];
    if (!p) return;
    if (prev == 1 && *p == ' ') p++;      /* llama2.c whitespace rule */
    fputs(p, stdout);
    fflush(stdout);
}

/* Parse a decimal like "0.8" into Q16.16 without touching floating point. */
static s32 parse_fixed(const char *p)
{
    s32 whole = 0, frac = 0, scale = 1;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') whole = whole * 10 + (*p++ - '0');
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9' && scale < 100000L) {
            frac = frac * 10 + (*p++ - '0');
            scale *= 10;
        }
    }
    whole = (whole << 16) + (frac * ONE) / scale;
    return neg ? -whole : whole;
}

/* ------------------------------------------------------------- banner */

/* Drawn from a rasterised silhouette, in a density ramp of . : I O X M @.
   No backslashes and no quotes anywhere in it, so it needs no escaping and
   survives every layer of quoting between this file and the Amiga.

   Printed only when generating text. -dump emits nothing but logits, which
   verify.py parses a line at a time; a banner in that stream breaks the gate. */
static void banner(void)
{
    static const char *const art[] = {
    "    I .:      stories260K",
    "   .XOXX",
    " :IXMMMX      260,000 parameters, int8 weights",
    " .II:IMX:     Q16.16 fixed point, no FPU",
    "      .XMX:   68000 @ 7.09 MHz, no math library",
    "        IMMO.    .::::::    :.",
    "         .XMMIOXMMMMMMMMMMXMMMO",
    "         IM@@@@@@@@@@@@@@@@@@X:",
    "         I@@@@M@@@@@@@@@@@@MI",
    "          .IIXX.I@OOOOOX@I @:",
    "             OM :@     :@: @:",
    "             OM :@     :@: @:",
    "             :I            O:",
    0
    };
    int i;
    for (i = 0; art[i]; i++) puts(art[i]);
    putchar(10);
    fflush(stdout);
}

/* --------------------------------------------------------------- main */

/* The positions verify.py diffs against the oracle. -dump walks all of them
   whatever -n says, so the cache has to cover them. */
static const int probe_tok[8] = { 1, 5, 9, 42, 7, 100, 3, 55 };
#define NPROBE ((int)(sizeof probe_tok / sizeof probe_tok[0]))

int main(int argc, char **argv)
{
    Model m;
    State s;
    Vocab v;
    int steps = 128, i, tok = 1, next, has_vocab, dump = 0;
    s32 temp = (s32)((8L * ONE) / 10);
    const char *mp = argc > 1 ? argv[1] : "stories260k.a5";
    const char *vp = argc > 2 ? argv[2] : "tok512.a5t";

    for (i = 3; i < argc; i += 2) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            temp = parse_fixed(argv[i + 1]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)
            { int w; rng_state = (u32)atol(argv[i + 1]); if (!rng_state) rng_state = 1;
              for (w = 0; w < 16; w++) rnd(); }   /* xorshift32 is weak on small seeds */
        else if (!strcmp(argv[i], "-dump")) { dump = 1; i--; }
    }

    if (!dump) banner();
    if (!load_model(&m, mp)) { fprintf(stderr, "cannot load %s\n", mp); return 1; }
    if (steps > m.seq_len) steps = m.seq_len;
    if (steps < 1) steps = 1;
    state_init(&s, &m, dump ? NPROBE : steps);
    has_vocab = load_vocab(&v, vp, m.vocab_size);

    if (dump) {
        for (i = 0; i < NPROBE; i++) {
            s32 *lg = forward(&m, &s, probe_tok[i], i);
            int j;
            for (j = 0; j < m.vocab_size; j++) printf("%ld\n", (long)lg[j]);
        }
        return 0;
    }

    for (i = 0; i < steps; i++) {
        s32 *lg = forward(&m, &s, tok, i);
        next = sample(lg, m.vocab_size, temp, m.exp_tab);
        if (next == 1 || next == 2) break;          /* BOS / EOS */
        if (has_vocab) emit(&v, tok, next);
        else printf("%d ", next);
        tok = next;
    }
    printf("\n");
    return 0;
}
