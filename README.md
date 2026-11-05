# Assignment 5 — Optimising Attention on a CPU

CS6886 Systems for Deep Learning, IIT Madras. Single-head attention,
`O = softmax(Q K^T / sqrt(d)) V`, made fast one measured step at a time on an
AMD Ryzen 7 6800H (8C/16T Zen 3+, 16 GB DDR5-4800, Windows 11).

**See [`report.pdf`](report.pdf) for the answers to Q1–Q5.** The source is
[`report.tex`](report.tex); its tables are generated from the measured data by
`tools/make_report.py` and compiled with Tectonic (XeTeX), so the report cannot
disagree with `results.csv`.

## Results at a glance (N=4096, d=64, 4.29 GFLOP of useful work)

| variant | time | GFLOP/s | speed-up vs naive | what it is |
|---|---|---|---|---|
| `naive` | 1375.9 ms | 3.1 | 1.0x | baseline, as given |
| `loops` | 188.2 ms | 22.8 | 7.3x | loop interchange + `K^T` so the inner loops are unit-stride axpys |
| `tiled` | 115.9 ms | 37.0 | 11.9x | cache + register tiling of both GEMMs (4x16 YMM tile, key-blocked outer product) |
| `fused` | 131.4 ms | 32.7 | 10.5x | the three steps fused row by row, no N x N matrix |
| `best` | **47.8 ms** | **89.8** | **28.8x** | query blocking + AVX2 exp + both register-tiled kernels |
| `omp` (16 thr) | **10.6 ms** | **405.5** | **129.9x** | `best` parallelised over query blocks |

Measured FP32 peak on this machine is 134.4 GFLOP/s on one core and 932 GFLOP/s on all eight cores, both taken in the same session as the table above, so `best` reaches 67 % of peak and `omp` reaches 44 % of the all-core peak.

`fused` lands *behind* `tiled`. That is the honest result: fusion saves ~200 MB
of score-matrix traffic but re-reads V per row (4 GB of L3 at d=64, 8 GB at
d=128). §3.3 of the report explains it.

> The sustained clock on this laptop follows its power state: 2.38 GHz in one
> session, 4.29 GHz in another. The peak is therefore re-measured every session
> and every table in the report comes from one session. Re-running
> `python tools/bench.py run && python tools/bench.py csv && python tools/plots.py
> && python tools/make_report.py` regenerates everything from fresh numbers.

## Build and run

The handout's `Makefile` is submitted unmodified and is the canonical build on a
POSIX machine:

```sh
make all      # build + time every variant on every input
make check    # compare every variant against naive (must print PASS)
```

This development machine has neither `make` (MSYS2's `make` cannot start its
signal pipe under the session sandbox) nor `clang`, so every number in the
report was produced with **gcc 14.1.0** through an equivalent driver that
performs exactly the same steps:

```sh
python tools/bench.py run      # build + time all variants, all inputs, all thread counts
python tools/bench.py check    # byte-for-byte port of the Makefile `check` rule
python tools/bench.py csv      # regenerate results.csv
```

`-march=native` resolves to `znver3` (AVX2 + FMA, no AVX-512), which is the same
ISA the Makefile would select with clang.

## Layout

| path | contents |
|---|---|
| `src/attn_*.c` | the six variants; each is self-contained with the handout's driver |
| `input/` | the four given `qkv_*.bin` inputs |
| `machine_spec.txt` | Q1, every field measured on this machine |
| `results.csv` | 40 rows: 6 variants × 4 inputs, plus every OpenMP thread count |
| `plots/roofline.png` | Q5 roofline with all variants at N=4096, d=64 |
| `plots/time_N*_d*.png` | one time plot per input (required) |
| `plots/threads_N*_d*.png` | extra: OpenMP strong scaling |
| `report.pdf` | answers to Q1–Q5 |
| `tools/probe.c` | peak FLOPS, clock, bandwidth, cache-hierarchy microbenchmarks |
| `tools/naive_phases.c` | where the naive kernel's time goes |
| `tools/phases.c` | per-kernel timing; `#include`s the shipped source |
| `tools/exp_test.c` | accuracy and throughput of the AVX2 exponential |
| `tools/bench.py` | Makefile-equivalent build/run/check/csv driver |
| `tools/plots.py`, `tools/make_report.py` | regenerate the figures and the report from measured data |
| `tools/gen_omp.py` | generates `attn_omp.c` from `attn_best.c` so the kernels cannot drift |
