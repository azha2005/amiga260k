#!/usr/bin/env python3
"""
scripts/verify.py -- the ground truth for this repo.

Exit 0 = llama68k.c matches oracle.py exactly. Exit 1 = it does not.
Everything an agent does here is judged by this script.

Pure Python on purpose: no shell, no chmod, no line-ending problems, and the
same behaviour on Windows, macOS and Linux.

    python scripts/verify.py
    python scripts/verify.py --model stories260k.a5
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROBE_TOKENS = [1, 5, 9, 42, 7, 100, 3, 55]


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def find_compiler():
    for cc in ("gcc", "cc", "clang"):
        exe = cc + (".exe" if os.name == "nt" else "")
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if d and os.path.isfile(os.path.join(d, exe)):
                return cc
    die("No C compiler on PATH. Install MSYS2/mingw-w64 (Windows), Xcode "
        "command line tools (macOS), or build-essential (Linux).")


def ensure_model(path):
    if os.path.isfile(path):
        return
    print(f"{os.path.basename(path)} not found, generating a synthetic model...")
    from ref_fp32 import make_synthetic
    import convert
    src = os.path.join(ROOT, "synthetic260k.bin")
    make_synthetic(src)
    convert.convert(src, path)


def build(cc):
    outdir = os.path.join(tempfile.gettempdir(), "a5build")
    os.makedirs(outdir, exist_ok=True)
    exe = os.path.join(outdir, "llama68k" + (".exe" if os.name == "nt" else ""))
    cmd = [cc, "-O2", "-Wall", "-Werror", "-DA5_HOST",
           "-o", exe, os.path.join(ROOT, "llama68k.c")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        die("BUILD FAILED\n" + r.stdout + r.stderr)
    return exe


def run_oracle(model):
    import oracle
    from convert import load_a5
    m = load_a5(model)
    st = oracle.State(m)
    out = []
    for pos, tok in enumerate(PROBE_TOKENS):
        out += oracle.forward(m, st, tok, pos)
    return [str(v) for v in out]


def run_c(exe, model):
    r = subprocess.run([exe, model, "none", "-dump"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        die("C binary failed to run\n" + r.stderr)
    return r.stdout.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "synthetic260k.a5"))
    args = ap.parse_args()

    os.chdir(ROOT)
    cc = find_compiler()
    ensure_model(args.model)
    exe = build(cc)

    py = run_oracle(args.model)
    c = run_c(exe, args.model)

    if len(py) != len(c):
        die(f"FAIL: oracle produced {len(py)} logits, C produced {len(c)}")

    bad = [i for i, (a, b) in enumerate(zip(c, py)) if a != b]
    if not bad:
        print(f"OK: C and oracle agree on all {len(py)} logits")
        return 0

    first = bad[0]
    vocab = len(py) // len(PROBE_TOKENS)
    print(f"FAIL: {len(bad)} of {len(py)} logits differ.", file=sys.stderr)
    print(f"First divergence at index {first} (position {first // vocab}, "
          f"token slot {first % vocab}): C={c[first]} oracle={py[first]}",
          file=sys.stderr)
    print("Do not proceed. The C has drifted from oracle.py.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
