"""
convert_tok.py -- tok512.bin (llama2.c) -> tok512.a5t

The llama2.c tokenizer file stores a float32 merge score next to every token.
Generation from BOS never needs the scores (they only matter when encoding a
prompt), so they are dropped: what ships to the Amiga is just the strings.

Output format, big-endian:
    'A5TK'                    magic
    per token: u16 len, then len bytes (no terminator)

Usage:  python3 convert_tok.py tok512.bin tok512.a5t 512
"""
import struct
import sys


def convert(src, dst, vocab_size):
    raw = open(src, "rb").read()
    off = 4                                   # skip max_token_length
    out = bytearray(b"A5TK")
    for i in range(vocab_size):
        (score,) = struct.unpack_from("<f", raw, off)
        off += 4
        (ln,) = struct.unpack_from("<i", raw, off)
        off += 4
        tok = raw[off:off + ln]
        off += ln
        out += struct.pack(">H", ln) + tok
    open(dst, "wb").write(out)
    print(f"wrote {dst}: {vocab_size} tokens, {len(out)} bytes")


def make_synthetic(dst, vocab_size):
    """A stand-in vocabulary so the pipeline can be exercised without the real
    tokenizer file."""
    out = bytearray(b"A5TK")
    for i in range(vocab_size):
        if i == 0:
            tok = b"<unk>"
        elif i == 1:
            tok = b"<s>"
        elif i == 2:
            tok = b"</s>"
        elif i < 32:
            tok = b"?"
        else:
            tok = (" w%d" % i).encode()
        out += struct.pack(">H", len(tok)) + tok
    open(dst, "wb").write(out)
    print(f"wrote synthetic {dst}: {vocab_size} tokens")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--synthetic":
        make_synthetic(sys.argv[2], 512)
    elif len(sys.argv) == 4:
        convert(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
