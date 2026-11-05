# Assignment 5 — Optimising Attention on a CPU

CS6886 Systems for Deep Learning, IIT Madras.

Single-head attention `O = softmax(Q K^T / sqrt(d)) V` in C, taken from the given
naive version to a blocked AVX2 kernel. Measured on an AMD Ryzen 7 6800H
(8C/16T Zen 3+, 16 GB DDR5-4800, Windows 11, gcc 14.1.0 `-O3 -march=native`).

## Start here

| If you want | Open |
|---|---|
| the answers to Q1–Q5 | **[`report.pdf`](report.pdf)** — 3 pages. Source: [`report.tex`](report.tex) |
| the raw numbers | [`results.csv`](results.csv) — 40 rows: variant × input × thread count |
| the machine details | [`machine_spec.txt`](machine_spec.txt) — the Q1 table |
| the code | [`src/`](src/) — six files, one per variant |
| the plots | [`plots/`](plots/) — roofline, plus one time plot per input |

## Results at a glance

N=4096, d=64, 4.29 GFLOP of useful work. All six variants pass `make check`
(max relative error 5.0e-6 against the 1e-4 limit).

| variant | time | GFLOP/s | vs naive | what changed |
|---|---|---|---|---|
| `naive` | 1375.9 ms | 3.1 | 1.0× | given baseline, three obvious loops |
| `loops` | 188.2 ms | 22.8 | 7.3× | loop interchange + `K^T`, so the inner loops are unit-stride axpys |
| `tiled` | 115.9 ms | 37.0 | 11.9× | cache and register tiling: 4×16 YMM tile, key-blocked S·V |
| `fused` | 131.4 ms | 32.7 | 10.5× | the three steps fused row by row, no N×N matrix |
| `best` | **47.8 ms** | **89.8** | **28.8×** | query blocking + AVX2 exp + both tiled kernels |
| `omp` (16 thr) | **10.6 ms** | **405.5** | **129.9×** | `best`, parallel over query blocks |

Measured peak is 134.4 GFLOP/s on one core and 932 GFLOP/s on all eight, so
`best` reaches 67 % of the single-core peak.

## Repository map

| path | contents |
|---|---|
| `src/attn_naive.c` | the given baseline, unchanged |
| `src/attn_loops.c`, `attn_tiled.c`, `attn_fused.c`, `attn_best.c` | Q3(a)–(d), in that order |
| `src/attn_omp.c` | Q4: `attn_best` with OpenMP over query blocks |
| `input/` | the four given `qkv_*.bin` inputs |
| `plots/roofline.png` | Q5 roofline, every variant at N=4096, d=64 |
| `plots/time_N*_d*.png` | one time plot per input; `threads_*.png` is extra scaling data |
| `Makefile` | the handout Makefile, unmodified |
| `tools/` | what produced the numbers — not needed to run the assignment |

`tools/`: `probe.c` (clock, peak FLOPS, bandwidth, caches), `phases.c` (kernel
time split, `naive` and `tiled` modes), `exp_test.c` (AVX2 exp accuracy),
`bench.py` (build/run/check), `plots.py` and `make_report.py` (regenerate the
figures and report from the measured data), `gen_omp.py` (writes `attn_omp.c`
from `attn_best.c`, so the two share their kernels by construction).

## Running it

```sh
make all      # build and time every variant on every input
make check    # compare every variant against naive; must print PASS
```

If `make` is not available, `python tools/bench.py run` and
`python tools/bench.py check` do the same steps.

## Two things to know

The driver in each `src/attn_*.c` is byte-identical to the one in `attn_naive.c`.
Only `attention()` differs, plus two small static helpers in the tiled and best
variants.

The sustained clock on this laptop follows its power state (2.38 GHz in one
session, 4.29 GHz in another), so the peak was re-measured per session and all
tables above come from a single session. Re-running on a cold laptop will give
proportionally different absolute times; the speed-ups stay close.
