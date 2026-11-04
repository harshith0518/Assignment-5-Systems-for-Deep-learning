#!/usr/bin/env python3
"""tools/plots.py -- build plots/roofline.png and plots/time_N<N>_d<d>.png.

Reads build/results.json (written by tools/bench.py) and tools/machine.json.
Run with a Python that has matplotlib, e.g. C:\\Python314\\python.exe.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

res = json.loads((ROOT / "build" / "results.json").read_text())
mach = json.loads((ROOT / "tools" / "machine.json").read_text())

VARIANTS = ["naive", "loops", "tiled", "fused", "best", "omp"]
COLOUR = {"naive": "#b03030", "loops": "#d08020", "tiled": "#c0a000",
          "fused": "#309040", "best": "#2060c0", "omp": "#8030b0"}


def get(variant, inp, threads=None):
    keys = [k for k in res if k.startswith(f"{variant}|{inp}|")]
    if not keys:
        return None
    if threads is not None:
        for k in keys:
            if res[k]["threads"] == threads:
                return res[k]
        return None
    return res[sorted(keys)[0]]


def parse(name):
    n = int(name.split("N")[1].split("_")[0])
    d = int(name.split("_d")[1].split(".")[0])
    return n, d


INPUTS = sorted({k.split("|")[1] for k in res})

# --------------------------------------------------------------------------
# 1. time plots, one per input
# --------------------------------------------------------------------------
for inp in INPUTS:
    n, d = parse(inp)
    names, times, gflops, colours = [], [], [], []
    for v in VARIANTS:
        r = get(v, inp, threads=16 if v == "omp" else None)
        if r is None:
            continue
        names.append(v + ("\n16 thr" if v == "omp" else ""))
        times.append(r["time_ms"])
        gflops.append(r["gflops"])
        colours.append(COLOUR[v])
    base = times[0]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(names, times, color=colours,
                  edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("median time over 5 runs  [ms]  (log scale)")
    ax.set_title(f"single-head attention, N={n}, d={d}   "
                 f"({4*n*n*d/1e9:.2f} GFLOP of useful work)")
    ax.grid(axis="y", which="both", alpha=0.3)
    for b, t, g, v in zip(bars, times, gflops, names):
        ax.text(b.get_x() + b.get_width() / 2, t * 1.06,
                f"{t:.2f} ms\n{g:.1f} GFLOP/s\n{base/t:.1f}x",
                ha="center", va="bottom", fontsize=7.2)
    ax.set_ylim(min(times) / 3, max(times) * 4)
    fig.tight_layout()
    fig.savefig(OUT / f"time_N{n}_d{d}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote plots/time_N{n}_d{d}.png")

# --------------------------------------------------------------------------
# 2. roofline at N=4096, d=64
# --------------------------------------------------------------------------
N, d = 4096, 64
FLOP = 4.0 * N * N * d          # 2N^2 d for each of the two GEMMs
compulsory = 4.0 * N * d        # Q + K + V + O, in floats

# Byte model per variant -- see the report's Q5 section for the justification.
# Every term is "bytes this variant must move at least once".
def traffic(v):
    S = float(N * N)                     # one N x N score matrix, floats
    if v in ("naive", "loops", "tiled"):
        # S is written once and read back twice (softmax, then S V)
        return (compulsory + 3.0 * S) * 4
    if v == "fused":
        return compulsory * 4
    # best/omp: no N x N in DRAM; K^T is written once, and V is re-read once
    # per query block of BI=64
    return (compulsory + 1.0 * N * d + (N / 64.0) * N * d) * 4


POINTS = []
for v in VARIANTS:
    r = get(v, f"qkv_N{N}_d{d}.bin", threads=16 if v == "omp" else None)
    if r is None:
        continue
    b = traffic(v)
    POINTS.append((v, FLOP / b, r["gflops"], r["time_ms"]))

fig, ax = plt.subplots(figsize=(8.6, 5.4))
xs = [10 ** (i / 60) for i in range(-40, 200)]

# measured single-core roofline
p1, b1 = mach["peak_1core_gflops"], mach["measured_bw_1thread_gbs"]
# measured all-core roofline
p16, b16 = mach["peak_allcore_gflops"], mach["measured_bw_16thread_gbs"]
ax.plot(xs, [min(p1, b1 * x) for x in xs], "k-", lw=2.0,
        label=f"roofline, 1 core  (peak {p1:.0f} GFLOP/s, BW {b1:.1f} GB/s)")
ax.plot(xs, [min(p16, b16 * x) for x in xs], "k--", lw=1.4, alpha=0.75,
        label=f"roofline, all 16 threads (peak {p16:.0f} GFLOP/s, BW {b16:.1f} GB/s)")
# rated (spec-sheet) roofline for comparison
ax.plot(xs, [min(mach["rated_peak_1core_gflops"], mach["theoretical_bw_gbs"] * x)
             for x in xs], ":", color="gray", lw=1.2,
        label=f"rated (not achievable): {mach['rated_peak_1core_gflops']:.0f} GFLOP/s, "
              f"{mach['theoretical_bw_gbs']:.1f} GB/s")

for v, ai, g, t in POINTS:
    ax.plot(ai, g, "o", ms=8, color=COLOUR[v], markeredgecolor="black", zorder=5)
    # per-point label offset so the annotations do not collide
    off = {"naive": (8, 6), "loops": (-12, -6), "tiled": (12, 4), "fused": (-10, 12),
           "best": (10, 8), "omp": (10, 8)}[v]
    ha = "right" if v in ("fused", "loops") else "left"
    va = "top" if v == "loops" else "bottom"
    ax.annotate(f"{v}\n{ai:.1f} F/B, {g:.1f} GFLOP/s ({t:.0f} ms)",
                (ai, g), textcoords="offset points", xytext=off, fontsize=7.4, ha=ha,
                va=va)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity  [FLOP / byte, DRAM traffic]")
ax.set_ylabel("attained performance  [GFLOP/s]")
ax.set_title(f"Roofline, {mach['cpu']}, N={N}, d={d}  "
             f"(useful work {FLOP/1e9:.2f} GFLOP)")
ax.grid(which="both", alpha=0.28)
ax.set_xlim(1, 4000)
ax.set_ylim(0.8, 1200)
ax.legend(loc="upper left", fontsize=7.6)
ax.annotate("points below the roof are limited by something the\n"
            "DRAM roofline does not model: dependency chains (naive),\n"
            "L3 bandwidth (fused), expf throughput (tiled)",
            xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6e3", ec="#bbb", lw=0.6))
fig.tight_layout()
fig.savefig(OUT / "roofline.png", dpi=150)
plt.close(fig)
print("  wrote plots/roofline.png")
for v, ai, g, t in POINTS:
    print(f"    {v:6s} AI={ai:7.1f} FLOP/B  attained={g:7.2f} GFLOP/s  {t:8.2f} ms")

# --------------------------------------------------------------------------
# 3. OpenMP strong scaling
# --------------------------------------------------------------------------
for inp in ("qkv_N4096_d64.bin", "qkv_N4096_d128.bin"):
    if not any(k.startswith("omp|" + inp + "|") for k in res):
        continue
    n, d = parse(inp)
    keys = sorted([k for k in res if k.startswith(f"omp|{inp}|")],
                  key=lambda k: res[k]["threads"])
    thr = [res[k]["threads"] for k in keys]
    tms = [res[k]["time_ms"] for k in keys]
    # baseline: this same binary with 1 thread (strong scaling)
    t1 = tms[thr.index(1)] if 1 in thr else tms[0]
    sp = [t1 / t for t in tms]
    eff = [s / t * 100 for s, t in zip(sp, thr)]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(thr, sp, "o-", color="#8030b0", label="speedup")
    ax1.plot(thr, thr, "k:", lw=1, label="linear")
    ax1.set_xlabel("OpenMP threads")
    ax1.set_ylabel("speedup over 1 thread")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(thr, eff, "s--", color="#d08020", ms=5, label="parallel efficiency")
    ax2.set_ylabel("parallel efficiency  [%]")
    ax2.set_ylim(0, 110)
    for x, s, e, t in zip(thr, sp, eff, tms):
        if x == 1:
            continue  # the baseline point collides with the axis
        ax1.annotate(f"{s:.2f}x\n{t:.1f} ms", (x, s), textcoords="offset points",
                     xytext=(9, -20), ha="left", fontsize=6.6)
    ax1.set_title(f"OpenMP strong scaling, N={n}, d={d}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"threads_N{n}_d{d}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote plots/threads_N{n}_d{d}.png")

# --------------------------------------------------------------------------
# 4. no-S-matrix illustration is not a plot; skip.
# --------------------------------------------------------------------------
print("done")
