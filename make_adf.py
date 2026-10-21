#!/usr/bin/env python3
"""
make_adf.py -- pack the runtime files into an 880 KB bootable ADF.

    python make_adf.py                       -> llama260k.adf
    python make_adf.py mydisk.adf            -> named image
    python make_adf.py --ffs                 -> FFS instead of OFS
    python make_adf.py --boot2x              -> Kickstart 2.0+ boot code
    python make_adf.py --shell out.adf       -> boot straight to the 1> prompt

The disk carries the Amiga executable, the model and the tokenizer, plus an
s/startup-sequence that runs the thing on boot. OFS and the 1.x boot code are
the defaults because a Kickstart 1.3 A500 can boot from neither an FFS floppy
nor a 2.x bootblock.

The executable must be an AmigaOS hunk binary from `python build.py amiga`.
A Windows or Linux host build will be rejected: it would waste a disk write
and fail on the target with no useful message.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

EXE_NAME = "llama68k"
ALT_EXE = "llama68k_c"        # C-kernel build, for A/B-ing the asm kernel
DATA = ["stories260k.a5", "tok512.a5t"]

STARTUP = (
    "llama68k stories260k.a5 tok512.a5t -n 128 -t 0.8 -s 42\n"
)

# --shell ships a startup-sequence that runs nothing, so the boot CLI
# drops straight to its prompt. Both file names always have to be typed:
# argv[1] and argv[2] are the model and the tokenizer and the option scan
# starts at argv[3], so "llama68k -t 0.5" tries to load a model called -t.
SHELL_STARTUP = (
    "; llama68k stories260k.a5 tok512.a5t -n 128 -t 0.8 -s 42\n"
)


def is_hunk(path):
    """AmigaOS load files start with HUNK_HEADER, 0x000003f3."""
    with open(path, "rb") as f:
        return f.read(4) == b"\x00\x00\x03\xf3"


def xdf(image, *cmds):
    argv = [sys.executable, "-m", "amitools.tools.xdftool", "-f", image]
    argv += list(cmds)
    print("+ xdftool", image, " ".join(cmds))
    r = subprocess.run(argv, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ffs = "--ffs" in sys.argv
    boot2x = "--boot2x" in sys.argv
    shell = "--shell" in sys.argv
    image = args[0] if args else "llama260k.adf"

    files = list(DATA)
    exe = os.path.join(ROOT, EXE_NAME)
    if os.path.exists(exe):
        if not is_hunk(exe):
            sys.exit("%s is not an AmigaOS hunk binary. Build it with "
                     "`python build.py amiga`; the host build will not run "
                     "on a 68000." % EXE_NAME)
        files.insert(0, EXE_NAME)
    else:
        print("warning: no %s in the project. Writing a data-only disk; "
              "cross-compile and re-run to make it bootable." % EXE_NAME)

    # The asm kernel is outside the verification loop. Ship the C-kernel build
    # alongside it so the two can be run with -t 0 on the target and diffed.
    alt = os.path.join(ROOT, ALT_EXE)
    if os.path.exists(alt) and is_hunk(alt):
        files.append(ALT_EXE)

    missing = [f for f in files if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        sys.exit("missing: " + ", ".join(missing))

    total = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
    # OFS carries 488 data bytes per 512-byte block, FFS the full 512.
    per_block = 512 if ffs else 488
    blocks = sum((os.path.getsize(os.path.join(ROOT, f)) + per_block - 1)
                 // per_block + 1 for f in files)
    if blocks > 1700:
        sys.exit("%d blocks of payload will not fit on an 880 KB disk" % blocks)
    print("payload %d bytes, about %d of 1758 blocks" % (total, blocks))

    if os.path.exists(os.path.join(ROOT, image)):
        os.remove(os.path.join(ROOT, image))

    xdf(image, "create")
    xdf(image, "format", "LLAMA260K", "ffs" if ffs else "ofs")
    for f in files:
        xdf(image, "write", f)

    startup = os.path.join(ROOT, "startup-sequence")
    with open(startup, "w", newline="\n") as fh:
        fh.write(SHELL_STARTUP if shell else STARTUP)
    try:
        xdf(image, "makedir", "s")
        xdf(image, "write", "startup-sequence", "s/startup-sequence")
    finally:
        os.remove(startup)

    # boot1x opens dos.library with OldOpenLibrary and checks no version, so it
    # boots on Kickstart 1.3 and on 2.x/3.x alike. The xdftool default,
    # boot2x3x, asks for version 37 and simply fails on a stock A500.
    xdf(image, "boot", "install", "boot2x3x" if boot2x else "boot1x")
    xdf(image, "list")
    print("\nwrote %s (%d bytes)"
          % (image, os.path.getsize(os.path.join(ROOT, image))))


if __name__ == "__main__":
    main()
