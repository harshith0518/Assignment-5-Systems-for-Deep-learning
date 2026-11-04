#!/usr/bin/env python3
"""
tools/bench.py -- cross-platform stand-in for the Makefile.

The handout's Makefile assumes a POSIX shell (mkdir -p, rm -f, basename,
[ -x ... ]) and clang.  Neither make nor clang exists on this Windows box, so
this script performs exactly the same steps with gcc so the numbers, the
output?*.bin files and the `make check` comparison are reproducible here.
On a POSIX machine `make all && make check` remains the canonical path and the
Makefile is unchanged.

Usage:
    python tools/bench.py build [variant ...]      compile only
    python tools/bench.py run   [variant ...]      compile + time every input
    python tools/bench.py check                    port of the Makefile `check`
    python tools/bench.py csv                      regenerate results.csv
    python tools/bench.py all                      build + run + check + csv

Environment overrides: CC, CFLAGS, OMP_THREADS (default "1,2,4,8,16")
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, BUILD, INP, OUT = ROOT / "src", ROOT / "build", ROOT / "input", ROOT / "output"
RESULTS_JSON = BUILD / "results.json"

VARIANTS = ["naive", "loops", "tiled", "fused", "best", "omp"]
NEEDS_OMP = {"omp"}

GCC_CANDIDATES = [
    os.environ.get("CC"),
    "clang",
    "gcc",
    r"C:\msys64\ucrt64\bin\gcc.exe",
    r"C:\msys64\mingw64\bin\gcc.exe",
    "/usr/bin/clang",
    "/usr/bin/gcc",
]


def find_cc():
    for c in GCC_CANDIDATES:
        if not c:
            continue
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    sys.exit("no C compiler found; set CC")


CC = find_cc()
CFLAGS = os.environ.get("CFLAGS", "-O3 -march=native").split()

TIME_RE = re.compile(r"N=(\d+)\s+d=(\d+)\s+threads=(\d+)\s+time_ms=\s*([\d.]+)\s+GFLOP/s=\s*([\d.]+)")


def inputs():
    return sorted(INP.glob("*.bin"))


def parse_input_name(p):
    m = re.search(r"N(\d+)_d(\d+)", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def build(variant, quiet=False):
    exe = BUILD / (variant + (".exe" if os.name == "nt" else ""))
    cmd = [CC, *CFLAGS]
    if variant in NEEDS_OMP:
        cmd.append("-fopenmp")
    cmd += ["-o", str(exe), str(SRC / f"attn_{variant}.c"), "-lm"]
    if not quiet:
        print("  " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)
    return exe, None


def run_one(exe, inp, variant, threads=1):
    of = OUT / f"{variant}_{inp.name}"
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    # Do NOT set OMP_PROC_BIND=close. Windows numbers logical CPUs so that the
    # two threads of one core are adjacent, so "close" packs a 2-thread run onto
    # a single physical core: measured 53.9 ms instead of 45.3 ms at N=4096 d=64.
    env.pop("OMP_PROC_BIND", None)
    r = subprocess.run([str(exe), str(inp), str(of)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return None
    m = TIME_RE.search(r.stdout)
    if not m:
        return None
    return dict(N=int(m.group(1)), d=int(m.group(2)), threads=int(m.group(3)),
                time_ms=float(m.group(4)), gflops=float(m.group(5)))


def cmd_build(variants):
    BUILD.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    ok = []
    for v in variants:
        exe, err = build(v)
        if exe:
            ok.append(v)
            print(f"  [{v}] built")
        else:
            print(f"  [{v}] FAILED\n{err}")
    return ok


def cmd_run(variants, threads_list):
    BUILD.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    res = {}
    if RESULTS_JSON.exists():
        try:
            res = json.loads(RESULTS_JSON.read_text())
        except Exception:
            res = {}

    for v in variants:
        exe, err = build(v)
        if exe is None:
            print(f"  [{v}] compile FAILED")
            print(err)
            continue
        tl = threads_list if v in NEEDS_OMP else [1]
        for p in inputs():
            for t in tl:
                key = f"{v}|{p.name}|{t}"
                print(f"  running {v:6s} {p.name:22s} threads={t:<3d}", end="", flush=True)
                r = run_one(exe, p, v, t)
                if r is None:
                    print("  -> FAILED")
                    continue
                r["input"] = p.name
                r["variant"] = v
                res[key] = r
                print(f"  -> {r['time_ms']:10.3f} ms  {r['gflops']:7.2f} GFLOP/s")
                RESULTS_JSON.write_text(json.dumps(res, indent=1))
    return res


def load_array(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        a = array("f")
        a.fromfile(f, n * d)
    return a


def cmd_check():
    """Port of the Makefile CHECK block, byte for byte identical logic."""
    variants = ["loops", "tiled", "fused", "best", "omp"]
    refs = sorted(OUT.glob("naive_*.bin"))
    if not refs:
        print("no naive output: run make naive first")
        return
    failed = 0
    for ref_path in refs:
        ref = load_array(ref_path)
        scale = max(abs(x) for x in ref)
        inp = ref_path.name[len("naive_"):]
        for v in variants:
            path = OUT / (v + "_" + inp)
            if not path.exists():
                continue
            out = load_array(path)
            if len(out) != len(ref):
                err = float("inf")
            else:
                err = max(abs(a - b) for a, b in zip(out, ref)) / scale
            ok = err <= 1e-4
            failed += 0 if ok else 1
            print("%-20s %-12s %s  (error %.1e)" % (inp, v, "PASS" if ok else "FAIL", err))
    missing = [v for v in variants if not list(OUT.glob(v + "_*.bin"))]
    if missing:
        print("no output (not implemented, did not compile, or crashed): " + " ".join(missing))
    print("\n%s" % ("ALL PASS" if not failed and not missing else "SOME FAILED"))


def arithmetic_intensity(variant, n, d):
    """FLOP / byte, using the same byte model as plots.py and the report (Q5).

    Bytes counted are the ones this variant must move at least once:
      naive/loops/tiled : Q,K,V,O once + the N x N score matrix written once
                          and read back twice (softmax, then S V)
      fused             : Q,K,V,O only -- no N x N matrix is materialised
      best/omp          : Q,K,V,O + K^T written once + V re-read once per
                          query block of BI = 64
    """
    flop = 4.0 * n * n * d
    compulsory = 4.0 * n * d
    if variant in ("naive", "loops", "tiled"):
        nbytes = (compulsory + 3.0 * n * n) * 4.0
    elif variant == "fused":
        nbytes = compulsory * 4.0
    else:  # best, omp
        nbytes = (compulsory + 1.0 * n * d + (n / 64.0) * n * d) * 4.0
    return flop / nbytes


def cmd_csv():
    if not RESULTS_JSON.exists():
        sys.exit("no results yet: run tools/bench.py run")
    res = json.loads(RESULTS_JSON.read_text())
    rows = []
    hdr = "variant,N,d,threads,time_ms,gflops,arithmetic_intensity_flop_per_byte,make_check"
    check = check_status()
    for v in VARIANTS:
        for p in inputs():
            n, d = parse_input_name(p)
            keys = sorted([k for k in res if k.startswith(f"{v}|{p.name}|")])
            for k in keys:
                r = res[k]
                ai = arithmetic_intensity(v, n, d)
                if v == "naive":
                    st = "reference"   # naive is the reference, nothing to check
                else:
                    st = check.get((v, p.name), "")
                rows.append(f"{v},{n},{d},{r['threads']},{r['time_ms']:.3f},{r['gflops']:.2f},"
                            f"{ai:.2f},{st}")
    (ROOT / "results.csv").write_text(hdr + "\n" + "\n".join(rows) + "\n")
    print(f"wrote results.csv ({len(rows)} rows)")


def check_status():
    """(variant, input) -> PASS/FAIL/blank, from the current output/ files."""
    st = {}
    for ref_path in sorted(OUT.glob("naive_*.bin")):
        ref = load_array(ref_path)
        scale = max(abs(x) for x in ref)
        inp = ref_path.name[len("naive_"):]
        for v in ["loops", "tiled", "fused", "best", "omp"]:
            path = OUT / (v + "_" + inp)
            if not path.exists():
                continue
            out = load_array(path)
            err = (max(abs(a - b) for a, b in zip(out, ref)) / scale
                   if len(out) == len(ref) else float("inf"))
            st[(v, inp)] = "PASS" if err <= 1e-4 else "FAIL"
    return st


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "all"
    args = argv[1:]
    threads_list = [int(t) for t in os.environ.get("OMP_THREADS", "1,2,4,8,16").split(",")]

    if cmd == "build":
        cmd_build(args or VARIANTS)
    elif cmd == "run":
        cmd_run(args or VARIANTS, threads_list)
    elif cmd == "check":
        cmd_check()
    elif cmd == "csv":
        cmd_csv()
    elif cmd == "all":
        vs = args or VARIANTS
        cmd_run(vs, threads_list)
        cmd_check()
        cmd_csv()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
