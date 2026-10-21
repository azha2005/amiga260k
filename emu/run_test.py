#!/usr/bin/env python3
"""
run_test.py -- run both kernels under WinUAE and diff their output.

    python emu/run_test.py [--rom PATH] [--timeout SECONDS] [--steps N]
                           [--cpu 68000|68020]

This is the A/B that CLAUDE.md demands for matmul68k.s: build llama68k (asm
kernel) and llama68k_c (C kernel), run both with -t 0, and require the output
to be character-identical. It is automated here by booting a test floppy whose
startup-sequence redirects each run to a host directory mounted as DH0:, then
calls UAEquit to shut the emulator down.

Without --rom it asks WinUAE for its built-in AROS ROM replacement, which needs
more RAM than an A500 and so runs with 2 MB chip. That is fine for comparing
two kernels -- they do the same arithmetic whatever the machine -- but it is
not a test of the 512K + A501 memory budget. For that, pass a real Kickstart
1.3 image dumped from your own machine.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMU = os.path.join(ROOT, "emu")
OUT = os.path.join(EMU, "out")
WINUAE = r"C:\Program Files\WinUAE\winuae64.exe"
ADF = os.path.join(EMU, "llama260k-test.adf")
CFG = os.path.join(EMU, "llama260k.uae")

DISK_FILES = ["llama68k", "llama68k_c", "stories260k.a5", "tok512.a5t"]

STARTUP = """llama68k >%(dev)sout_asm.txt stories260k.a5 tok512.a5t -n %(n)d -t 0
llama68k_c >%(dev)sout_c.txt stories260k.a5 tok512.a5t -n %(n)d -t 0
%(quit)s"""


def xdf(*cmds):
    argv = [sys.executable, "-m", "amitools.tools.xdftool", "-f", ADF]
    argv += list(cmds)
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("xdftool failed: " + " ".join(cmds) + "\n" + r.stdout + r.stderr)


def build_disk(steps, dev, can_quit):
    ss = os.path.join(EMU, "startup-sequence")
    with open(ss, "w", newline='\n') as f:
        f.write(STARTUP % {"n": steps, "dev": dev,
                           "quit": "UAEquit" + NL if can_quit else ""})
    if os.path.exists(ADF):
        os.remove(ADF)
    xdf("create")
    xdf("format", "LLAMATEST", "ofs")
    for f in DISK_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            sys.exit("missing %s -- run `python build.py amiga` and `amiga-c`" % f)
        xdf("write", p)
    if can_quit:
        xdf("write", os.path.join(EMU, "UAEquit"))
    xdf("makedir", "s")
    xdf("write", os.path.join(EMU, "startup-sequence"), "s/startup-sequence")
    xdf("boot", "install", "boot1x")
    print("test disk:", ADF)


def write_config(rom, cpu, a501, out):
    # 2 MB chip for the AROS replacement, which will not fit an A500's memory.
    # With a real Kickstart this drops to chipmem_size=1 + bogomem_size=2.
    real_rom = rom is not None
    is68000 = cpu == "68000"
    cfg = [
        "config_description=llama260k asm-vs-C kernel test",
        "use_gui=no",
        "chipset=ocs" if is68000 else "chipset=aga",
        "chipset_compatible=A500" if is68000 else "chipset_compatible=A1200",
        "cpu_type=" + cpu,
        "cpu_compatible=false",
        "cpu_speed=max",
        # Nothing here depends on chipset timing, so turn off the
        # cycle-exact machinery: it is pure overhead for a batch run.
        "cycle_exact=false",
        "blitter_cycle_exact=false",
        "immediate_blits=true",
        "gfx_framerate=10",
        "chipmem_size=%d" % (1 if real_rom else 4),
        "bogomem_size=%d" % ((2 if a501 else 0) if real_rom else 0),
        "fastmem_size=%d" % (0 if real_rom else 8),
        "kickstart_rom_file=%s" % (rom if real_rom else ":AROS"),
        "floppy0=%s" % ADF,
        "floppy0type=0",
        "floppy0wp=false",
        "nr_floppies=1",
        "floppy_speed=800",
        # A directory filesystem needs WinUAE's boot ROM, which relies on
        # autoboot support that Kickstart 1.2 does not have -- configuring
        # one stops a 1.2 machine from booting at all. With DF0: output it
        # is not needed, so leave it out entirely.
        ("filesystem2=rw,DH0:DH0:%s,0" % OUT) if out == "DH0:" else None,
        "win32.start_not_captured=true",
        "win32.nonotificationicon=true",
    ]
    cfg = [line for line in cfg if line is not None]
    with open(CFG, "w", newline="\n") as f:
        f.write("\n".join(cfg) + "\n")
    print("config:", CFG)


def main():
    rom = None
    timeout = 300
    steps = 8
    cpu = None
    a501 = True
    out = "DH0:"
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--rom" and i + 1 < len(argv):
            rom = os.path.abspath(argv[i + 1])
        elif a == "--timeout" and i + 1 < len(argv):
            timeout = int(argv[i + 1])
        elif a == "--steps" and i + 1 < len(argv):
            steps = int(argv[i + 1])
        elif a == "--cpu" and i + 1 < len(argv):
            cpu = argv[i + 1]
        elif a == "--no-a501":
            a501 = False
        elif a == "--out" and i + 1 < len(argv):
            out = {"df0": "DF0:", "dh0": "DH0:"}[argv[i + 1].lower()]
    if rom and not os.path.exists(rom):
        sys.exit("no such ROM: " + rom)
    # A real Kickstart means a real A500; AROS needs a 68020 to boot at all.
    if cpu is None:
        cpu = "68000" if rom else "68020"

    os.makedirs(OUT, exist_ok=True)
    for f in ("out_asm.txt", "out_c.txt"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            os.remove(p)

    if not os.path.exists(WINUAE):
        sys.exit("WinUAE not found at " + WINUAE)

    build_disk(steps, out, rom is None)
    write_config(rom, cpu, a501, out)

    mem = "512K chip" + (" + 512K A501" if (rom and a501) else "")           if rom else "2M chip + 8M fast"
    print("launching WinUAE (%s ROM, %s, %s, -n %d), timeout %ds..."
          % ("Kickstart" if rom else "AROS replacement", cpu, mem, steps,
             timeout))
    log = os.path.join(EMU, "winuaelog.txt")
    if os.path.exists(log):
        os.remove(log)
    p = subprocess.Popen([WINUAE, "-f", CFG, "-log"], cwd=EMU)
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < timeout:
        time.sleep(2)
    if p.poll() is None:
        if rom:
            print("wait elapsed -- stopping WinUAE "
                  "(UAEquit cannot load on Kickstart 1.x)")
        else:
            print("timeout -- killing WinUAE")
        p.kill()
        time.sleep(2)
    else:
        print("WinUAE exited after %.0fs (UAEquit worked)" % (time.time() - t0))

    if out == "DF0:":
        for name in ("out_asm.txt", "out_c.txt"):
            dst = os.path.join(OUT, name)
            if os.path.exists(dst):
                os.remove(dst)
            r = subprocess.run(
                [sys.executable, "-m", "amitools.tools.xdftool", ADF,
                 "read", name, dst], capture_output=True, text=True)
            if r.returncode != 0:
                print("not on the floppy: %s" % name)
    a = os.path.join(OUT, "out_asm.txt")
    c = os.path.join(OUT, "out_c.txt")
    if not os.path.exists(a) or not os.path.exists(c):
        print("\nNO OUTPUT. The Amiga side never wrote to DH0:.")
        print("Files in %s: %s" % (OUT, os.listdir(OUT)))
        return 2
    da, dc = open(a, "rb").read(), open(c, "rb").read()
    print("\nasm kernel (%d bytes): %r" % (len(da), da[:120]))
    print("C   kernel (%d bytes): %r" % (len(dc), dc[:120]))
    if da == dc and da:
        print("\nIDENTICAL -- matmul68k.s agrees with the C kernel")
        return 0
    print("\nDIFFERENT -- matmul68k.s does not match the C kernel")
    return 1


if __name__ == "__main__":
    sys.exit(main())
