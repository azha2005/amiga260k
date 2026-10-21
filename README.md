# stories260K on an Amiga 500

A 260K-parameter Llama running on a 68000 at 7.09 MHz, in integer arithmetic,
with no FPU and no operating system dependencies beyond `fopen`.

## The files

| File | What it is |
|---|---|
| `ref_fp32.py` | llama2.c checkpoint reader, fp32 forward pass, synthetic-model generator |
| `fx.py` | Q16.16 fixed-point primitives, written the way the 68000 will run them |
| `convert.py` | `stories260K.bin` -> `stories260k.a5` (big-endian, int8) |
| `convert_tok.py` | `tok512.bin` -> `tok512.a5t` |
| `oracle.py` | The integer forward pass in Python. **This is the specification.** |
| `llama68k.c` | The engine. Same arithmetic, in C. |
| `matmul68k.s` | The 68000 dot-product kernel |
| `build.py` | Host build, verification, Amiga cross build |

## Step by step

### 1. Get the weights

```
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin
```

### 2. Convert

```
python3 convert.py stories260K.bin stories260k.a5
python3 convert_tok.py tok512.bin tok512.a5t 512
```

The `.a5` comes out around 300 KB. Big-endian throughout, so the 68000 reads
it with plain pointer casts and no byte swapping at all.

### 3. Verify on the host, before touching a 68000

```
python build.py host
python build.py verify stories260K.bin
```

This runs the same eight token positions through the Python oracle and through
the C, and diffs all 4096 resulting int32 logits. They must be identical. Any
difference means the C has drifted from the specification, and debugging that
on an Amiga instead of here is a bad way to spend a weekend.

The one bug this caught during development is instructive: the C computed
`1/sqrt(head_size)` with an integer square root and the oracle with
`math.sqrt`. They differed by 2 units in Q16.16 — invisible at position 0,
and it corrupted every token from position 4 onward.

### 4. Cross-compile for the Amiga

Install [vbcc](http://sun.hasenbraten.de/vbcc/) with the `aos68k` target and
[vasm](http://sun.hasenbraten.de/vasm/), then:

```
python build.py amiga
```

Copy `llama68k`, `stories260k.a5` and `tok512.a5t` to an ADF or a Gotek.
They total about 310 KB, so a standard 880 KB DD disk holds the lot with room
for a startup-sequence.

### 5. Run

```
llama68k stories260k.a5 tok512.a5t -n 128 -t 0.8 -s 42
```

`-t 0` gives greedy decoding, which is the right setting for A/B-ing the asm
kernel against the C one: the output must be character-identical.

## How the arithmetic works

Everything is Q16.16 in `s32`. No floats, no 64-bit types, and every product
is built from 16x16 multiplies so it maps onto `MULS`/`MULU`.

**Weights** are int8 with one scale per output row. Each scale is a `uint16`
mantissa plus a single per-tensor shift, so the rescale after a dot product is
one multiply and one shift.

**Activations** are int8 with a power-of-two scale. Quantising a vector
therefore costs one arithmetic shift per element and nothing else. It gives up
at most one bit of range compared to a proper scale, which does not matter at
this model size.

**The dot product** accumulates in a full int32, so it is exact. The rescale
uses a 48-bit intermediate assembled from two `MULU`s — that runs once per
output element, not once per multiply-accumulate, so its cost disappears.

**Exp and sigmoid** are 1025-entry tables with linear interpolation, built at
conversion time and shipped inside the `.a5`. Nothing calls `libm` at runtime.

**`rsqrt`** is an integer square root plus one Newton step, then a long
division. It runs 11 times per token, so its cost is irrelevant.

Measured against fp32 on a model with this geometry: correlation 0.998 on the
logits, and 8 to 10 of the top 10 tokens preserved.

## What it should cost

Operation counts, measured, per token:

| | pos 0 | pos 128 | pos 256 |
|---|---|---|---|
| int8 multiply-accumulates | 278,528 | 278,528 | 278,528 |
| `fxmul` (Q16.16) | 5,712 | 97,872 | 190,032 |
| `fxdiv` | 63 | 63 | 63 |
| `mulshift` | 3,776 | 3,776 | 3,776 |

At 78 cycles per int8 MAC, 220 per `fxmul`, 1600 per `fxdiv` and 130 per
`mulshift`, on a 7.09 MHz 68000:

| Position | Cycles | Seconds per token |
|---|---|---|
| 0 | 23.6M | 3.3 |
| 128 | 43.8M | 6.2 |
| 256 | 64.1M | 9.0 |

So roughly **0.3 tokens/s at the start of a story, falling to 0.11 by the
512-token mark**. The matmuls are a fixed cost; what grows is attention, which
is still in Q16.16 and re-reads the whole KV cache on every token.

These are static cycle estimates from the 68000 instruction timings, not
measurements on hardware. Expect real numbers to be worse, mostly from chip-RAM
bus contention (see below).

## Two things worth doing next

**Quantise the KV cache to int8.** Attention is the only part still doing
Q16.16 multiplies, and at position 256 it is 164,000 of the 190,000 `fxmul`
calls. Routing those through `dot_i8_x16` at 78 cycles instead of 220 takes
position 256 from about 9.0 to about 5.7 seconds per token, and shrinks the
cache from 40 KB to 10 KB at the same time.

**Turn off bitplane DMA while computing.** On the A500 the 68000 shares the
chip-RAM bus with Agnus, and the display steals cycles in proportion to the
number of active bitplanes. Clearing the bitplane bits in `DMACON` around the
forward pass and restoring them to print the token is worth roughly 25%.

Note also that the A501 trapdoor expansion is *slow RAM*, not Fast RAM: it sits
at `$C00000` but is still on the chip bus. Genuine Fast RAM needs a side
expansion, and it is worth about the same 25% again.

## Note on the sampler

`rnd()` is a 32-bit xorshift, seeded by `-s`, with 16 warm-up iterations
because xorshift32 produces poor output for the first few steps from a small
seed. It is deterministic: the same seed gives the same story, on the host and
on the Amiga alike. That is what makes the two comparable.
