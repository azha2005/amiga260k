---
description: Make a change to the numerical path, oracle first
argument-hint: [what to change]
---

Implement this change to the numerical path: $ARGUMENTS

Follow this order strictly. Do not skip ahead.

1. **Oracle first.** Change `oracle.py`, and `fx.py` if a primitive is involved.
   Nothing in `llama68k.c` yet.
2. **Check accuracy.** Run `python3 cmp_fp32.py`. Correlation must stay above
   0.99 and top-10 overlap at 8/10 or better. If it drops, stop and report the
   trade-off rather than continuing.
3. **Port to C.** Now change `llama68k.c` to match, line for line where possible.
   Watch the invariants in CLAUDE.md: `s32` only, no floats, no 64-bit types,
   no plain 32x32 multiplies.
4. **Verify.** `sh scripts/verify.sh` must pass. If it does not, use the
   bisection approach in `/verify` — do not guess.
5. **Measure**, if this was a performance change. Instrument the operation
   counters and report the before and after, plus the resulting cycle estimate.

If the change touches `matmul68k.s`, say explicitly that it is unverified: the
gate cannot check assembly.
