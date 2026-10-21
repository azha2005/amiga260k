---
description: Run the bit-exactness gate and interpret the result
---

Run `sh scripts/verify.sh`.

If it passes, say so in one line and stop.

If it fails, do not start editing. First read the reported position of the
first divergence and reason about what it implies:

- **Position 0** — the divergence is structural and hits the very first token.
  Look for a changed constant, a wrong table offset, or a shape mismatch.
- **Position N > 0** — everything up to N-1 matched exactly, so the arithmetic
  is broadly right and something diverges only once values grow. Suspect a
  rounding path, an overflow, or the same quantity computed two different ways
  in `oracle.py` and `llama68k.c`.

Then bisect by probing an intermediate rather than guessing: pick a tensor
partway through the forward pass, print it from both sides at the failing
position, and narrow down. The primitives in `fx.py` can be differential-tested
directly against their C counterparts on random inputs — that is usually faster
than reasoning about the full network.

Report what you found before you fix it.
