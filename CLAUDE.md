# stories260K on Amiga 500 (68000)

A 260K-parameter Llama in integer arithmetic, targeting a 7.09 MHz 68000 with
no FPU. Read this before touching anything.

## The one rule

`oracle.py` is the specification. `llama68k.c` is an implementation of it.
They must produce identical int32 logits, always.

`python scripts/verify.py` is the ground truth. It exits 0 when they agree on all
4096 logits and 1 when they do not. **A change that leaves verify.py failing is
not finished, no matter how correct it looks.**

## Order of work for any numerical change

1. Change `oracle.py` (and `fx.py` if a primitive is involved) first.
2. Regenerate expectations and confirm the accuracy is still acceptable:
   `python3 cmp_fp32.py` — correlation must stay above 0.99 and top-10 overlap
   at 8/10 or better.
3. Port the same change to `llama68k.c`.
4. `python scripts/verify.py` must pass.

Never change the C first and then adjust the oracle to match it. That turns the
oracle into a mirror and destroys the only signal in this repo.

## Invariants that are easy to break

- **`s32` must be exactly 32 bits.** On the Amiga it is `long`; the host build
  uses `int` under `-DA5_HOST` because `long` is 64-bit there. Any new integer
  variable in the numeric path uses `s32`, never bare `int` or `long`.
- **No floating point anywhere in `llama68k.c`.** No `float`, no `double`, no
  `libm`, no `atof`. The 68000 has no FPU and there is no math library on the
  target. `parse_fixed()` exists precisely so argument parsing does not need one.
- **No 64-bit types.** `long long` is not available. Wider intermediates are
  assembled from 16x16 multiplies — see `mulshift()` and `fxdiv()`.
- **Every product must decompose into 16x16 multiplies**, because that is what
  `MULU`/`MULS` give us. If a change introduces a plain 32x32 multiply, it is
  wrong even if it compiles.
- **Python and C integer semantics differ.** Python `//` and `%` floor; C
  truncates toward zero. Python integers are unbounded; `s32` wraps. Use
  `fx.i32()` in the oracle wherever the C would wrap.
- **The `.a5` file is big-endian.** The Amiga reads it with plain pointer casts.
  Only the host build byte-swaps, under `A5_HOST`.

## The bug class to watch for

The worst failures here are two implementations computing the same quantity by
different routes. The real example: `1/sqrt(head_size)` was computed with
`isqrt32` in the C and `math.sqrt` in the oracle. They differed by 2 units in
Q16.16. Positions 0 through 3 matched exactly; everything from position 4 on
was wrong.

When verify.py reports a first divergence at position N > 0, suspect a constant
or a rounding path that only matters once values grow — not a structural bug.

## Files

| File | Role |
|---|---|
| `fx.py` | Q16.16 primitives, Python side |
| `oracle.py` | **The specification.** Integer forward pass |
| `ref_fp32.py` | fp32 reference + synthetic model generator |
| `convert.py` | checkpoint -> `.a5`, and the `.a5` reader |
| `convert_tok.py` | tokenizer -> `.a5t` |
| `llama68k.c` | The engine |
| `matmul68k.s` | 68000 dot-product kernel |
| `cmp_fp32.py` | quantisation accuracy check |
| `scripts/verify.py` | bit-exactness gate |

## matmul68k.s is outside the verification loop

There is no m68k assembler or emulator in this environment, so `verify.py`
cannot check the assembly. Treat any change to it as unverified.

The only acceptable evidence that the asm is correct is: build both
`python build.py amiga` and `python build.py amiga-c`, run each with `-t 0` (greedy,
deterministic), and confirm the output is character-identical. Say plainly when
a change to the asm has not been checked this way. Do not describe it as
working.

## Performance model

Cost per token is dominated by 278,528 int8 multiply-accumulates (fixed) plus
attention in Q16.16 (grows with position). At 78 cycles per MAC and 220 per
`fxmul`: about 3.3 s/token at position 0, 9.0 s/token at position 256.

An optimisation only counts if it moves those counts. Measure with the counters
before claiming a speedup, and re-run `cmp_fp32.py` after — a speedup that costs
accuracy needs to be reported as the trade it is.

## Commands

```
python scripts/verify.py     # the gate
python build.py host         # host build
python3 cmp_fp32.py      # quantisation accuracy vs fp32
python build.py amiga        # cross build (needs vbcc + vasm)
```
