#!/usr/bin/env python3
"""
build.py -- host build and Amiga cross build. Cross-platform replacement for
build.sh; no shell required.

    python build.py host                  compile the host verification binary
    python build.py verify [ckpt.bin]     convert a checkpoint and run the gate
    python build.py amiga                 cross build with the asm kernel
    python build.py amiga-c               cross build with the C kernel

The Amiga builds use vbcc's kick13 target, not aos68k. aos68k links with
-Rshort and emits HUNK_RELOC32SHORT, which Kickstart 1.x cannot load: the
machine answers "error 121" and nothing runs. kick13 emits plain HUNK_RELOC32
and works on 1.2 and 1.3 as well as later.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""


def run(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""

    if what == "host":
        run(["gcc", "-O2", "-Wall", "-DA5_HOST", "-o", "llama68k" + EXE,
             "llama68k.c"])

    elif what == "verify":
        if len(sys.argv) > 2:
            run([sys.executable, "convert.py", sys.argv[2], "model.a5"])
            run([sys.executable, "scripts/verify.py", "--model", "model.a5"])
        else:
            run([sys.executable, "scripts/verify.py"])

    elif what == "amiga":
        run(["vasmm68k_mot", "-Fhunk", "-kick1hunks", "-m68000", "-no-opt",
             "matmul68k.s", "-o", "matmul68k.o"])
        run(["vc", "+kick13", "-c99", "-O2", "-DUSE_ASM", "-c",
             "llama68k.c", "-o", "llama68k.o"])
        run(["vc", "+kick13", "-o", "llama68k", "llama68k.o", "matmul68k.o"])

    elif what == "amiga-c":
        run(["vc", "+kick13", "-c99", "-O2", "-c",
             "llama68k.c", "-o", "llama68k_c.o"])
        run(["vc", "+kick13", "-o", "llama68k_c", "llama68k_c.o"])

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
