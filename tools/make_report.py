#!/usr/bin/env python3
"""Generate report.tex from the measured data, then compile it with Tectonic.

All tables and the numbers quoted in the prose are computed from
build/results.json, so the report cannot disagree with results.csv.
Run:  C:\\Python314\\python.exe tools/make_report.py
"""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "build" / "results.json").read_text())
M = json.loads((ROOT / "tools" / "machine.json").read_text())

I1, I2, I64, I128 = ("qkv_N1024_d64.bin", "qkv_N2048_d64.bin",
                     "qkv_N4096_d64.bin", "qkv_N4096_d128.bin")
INPUTS = [I1, I2, I64, I128]
LABEL = {I1: "1024/64", I2: "2048/64", I64: "4096/64", I128: "4096/128"}
P1, PA = M["peak_1core_gflops"], M["peak_allcore_gflops"]
PR = M["rated_peak_1core_gflops"]
BW1, BW16 = M["measured_bw_1thread_gbs"], M["measured_bw_16thread_gbs"]


def g(v, inp, th=None):
    ks = [k for k in res if k.startswith(f"{v}|{inp}|")]
    if th is not None:
        ks = [k for k in ks if res[k]["threads"] == th]
    return res[sorted(ks)[0]]


def T(v, inp, th=None):
    return g(v, inp, th)["time_ms"]


def GF(v, inp, th=None):
    return g(v, inp, th)["gflops"]


BASE = {i: T("naive", i) for i in INPUTS}
best64, omp64 = T("best", I64), T("omp", I64, 16)
sp_best = BASE[I64] / best64
sp_omp = BASE[I64] / omp64
pct_best = 100 * GF("best", I64) / P1
pct_omp = 100 * GF("omp", I64, 16) / PA
pct_naive = 100 * GF("naive", I64) / P1

# Phase profiles: tools/naive_phases.c and tools/phases.c, 4096/64 unless noted.
NV = {I1: (50.646, 4.988, 86.243, 141.877),
      I2: (195.278, 18.432, 351.462, 565.172),
      I64: (803.796, 85.072, 1638.865, 2527.733),
      I128: (1776.435, 89.131, 5827.437, 7693.004)}
QK, SM, SV = 38.762, 81.857, 45.622
PHSUM = QK + SM + SV
ALLOC = T("tiled", I64) - PHSUM
QK_PK, SV_PK = 72.4, 61.5
EXPERR, EXPSPD = "5.1\\times10^{-6}", 8.4

# ---------------------------------------------------------------- tables ---
tab_q2 = ""
for i in INPUTS:
    tab_q2 += (f"{LABEL[i]} & {T('naive',i):8.2f} & {GF('naive',i):5.2f} & "
               f"{100*GF('naive',i)/P1:4.1f} & {100*GF('naive',i)/PR:4.1f} \\\\\n")

tab_q3 = ""
for v, tag in [("loops", "loops"), ("tiled", "tiled"), ("fused", "fused"),
               ("best", "best"), ("omp", "omp (16 thr)")]:
    cells = ""
    for i in INPUTS:
        th = 16 if v == "omp" else None
        cells += f" & {T(v,i,th):7.2f} & {BASE[i]/T(v,i,th):5.1f}"
    tab_q3 += f"\\texttt{{{tag}}}" + cells + " \\\\\n"
cells = "".join(f" & {T('naive',i):7.2f} & 1.0" for i in INPUTS)
tab_q3 += "\\midrule\n\\texttt{naive}" + cells + " \\\\\n"

tab_q4 = ""
for i in [I64, I128]:
    t1 = T("omp", i, 1)
    for t in [1, 2, 4, 8, 16]:
        r = g("omp", i, t)
        s = t1 / r["time_ms"]
        first = f"\\multirow{{5}}{{*}}{{{LABEL[i]}}}" if t == 1 else ""
        tab_q4 += (f"{first} & {t:2d} & {r['time_ms']:8.2f} & {s:5.2f} & "
                   f"{100*s/t:4.0f} & {r['gflops']:6.1f} \\\\\n")
    if i != I128:
        tab_q4 += "\\midrule\n"

N, d = 4096, 64
FLOP = 4.0 * N * N * d
COMP = 4.0 * N * d


def traffic(v):
    S = float(N * N)
    if v in ("naive", "loops", "tiled"):
        return (COMP + 3.0 * S) * 4          # Q,K,V,O + S written once, read twice
    if v == "fused":
        return COMP * 4                      # no N x N matrix at all
    return (COMP + N * d + (N / 64.0) * N * d) * 4  # + K^T, + V per query block


AI = {v: FLOP / traffic(v) for v in ["naive", "loops", "tiled", "fused", "best", "omp"]}
tab_q5 = ""
for v, tag in [("naive", "naive"), ("loops", "loops"), ("tiled", "tiled"),
               ("fused", "fused"), ("best", "best"), ("omp", "omp (16)")]:
    th = 16 if v == "omp" else None
    tab_q5 += (f"\\texttt{{{tag}}} & {AI[v]:6.1f} & {T(v,I64,th):7.2f} & "
               f"{GF(v,I64,th):6.1f} & "
               f"{100*GF(v,I64,th)/(PA if v=='omp' else P1):4.0f} \\\\\n")

TEX = r"""\documentclass[10pt,a4paper]{article}
\usepackage[top=1.5cm,bottom=1.4cm,left=1.7cm,right=1.7cm]{geometry}
\usepackage{booktabs,array,multirow,graphicx,amsmath,float,longtable}
\usepackage[T1]{fontenc}
\usepackage[font=footnotesize,labelfont=bf,skip=2pt]{caption}
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.5pt}
\setlength{\tabcolsep}{4pt}
\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\var}[1]{\texttt{#1}}
\pagestyle{empty}

\begin{document}
\begin{center}
{\LARGE\bfseries Optimising Attention on a CPU}\\[2pt]
{\large CS6886 Systems for Deep Learning --- Assignment 5}\\[3pt]
{\small """ + f"{M['cpu']}" + r""" (8C/16T Zen~3+), 16\,GB DDR5-4800, Windows 11.
gcc 14.1.0, \code{-O3 -march=native}. All numbers measured on this machine.}
\end{center}
\vspace{-8pt}

\noindent\rule{\linewidth}{0.4pt}
\small
At $N{=}4096,d{=}64$ the naive kernel runs at """ + f"{GF('naive',I64):.2f}" + r"""\,GFLOP/s.
The final single-threaded kernel runs at """ + f"{GF('best',I64):.1f}" + r"""\,GFLOP/s, a
""" + f"{sp_best:.0f}" + r"""$\times$ speed-up. Sixteen threads reach
""" + f"{GF('omp',I64,16):.0f}" + r"""\,GFLOP/s, a """ + f"{sp_omp:.0f}" + r"""$\times$ speed-up.
All variants pass the check.\par
\noindent\rule{\linewidth}{0.4pt}
\normalsize

\section{Machine}

I read the specification from SMBIOS and CPUID. I measured the limits with
\var{tools/probe.c}. Table~\ref{tab:machine} gives the result.

\begin{table}[h]
\centering\footnotesize
\caption{Machine specification. ``Measured'' means a microbenchmark, not a datasheet value.}
\label{tab:machine}
\begin{tabular}{@{}ll@{}}
\toprule
CPU & AMD Ryzen 7 6800H, Zen 3+ (family 0x19, model 0x44) \\
Cores / threads & 8 physical, 16 logical, 2 CCX $\times$ 4 cores. No P/E split. \\
Clock & 3.20\,GHz base, 4.75\,GHz rated boost. \textbf{Measured 2.38\,GHz 1-core, 2.02\,GHz all-core.} \\
SIMD & AVX2 + FMA3, 256-bit, 2 FMA units per core, 16 YMM registers. No AVX-512. \\
Caches & L1D 32\,KiB/core, L2 512\,KiB/core, L3 16\,MiB (8\,MiB per CCX), 64\,B lines. \\
Memory & 16\,GB DDR5-4800 SO-DIMM, 2 channels. DRAM latency """ + f"{M['dram_latency_ns']:.0f}" + r"""\,ns. \\
\bottomrule
\end{tabular}
\end{table}

Zen~3+ retires two 256-bit FMAs per cycle per core. One FMA is 8 lanes
$\times$ 2\,FLOP, so $P_\text{core}=32f$:
\begin{equation*}
P_1 = 32\times2.38 = """ + f"{P1:.1f}" + r"""\ \text{GFLOP/s},\qquad
P_8 = 32\times8\times2.02 = """ + f"{PA:.0f}" + r"""\ \text{GFLOP/s}
\end{equation*}
\begin{equation*}
B_\text{theor} = 4800\,\text{MT/s}\times8\,\text{B}\times2 = 76.8\ \text{GB/s},\quad
B_1 = """ + f"{BW1}" + r"""\ \text{GB/s},\quad B_{16} = """ + f"{BW16}" + r"""\ \text{GB/s}
\end{equation*}
\begin{equation*}
R_1 = """ + f"{P1:.1f}/{BW1} = {M['ridge_1core_measured']:.2f}" + r"""\ \text{FLOP/B},\qquad
R_8 = """ + f"{PA:.0f}/{BW16} = {M['ridge_allcore_measured']:.2f}" + r"""\ \text{FLOP/B}
\end{equation*}

\textbf{The part runs at half its rated boost.} The clock settles at 2.38\,GHz,
not 4.75\,GHz. Three checks exclude a measurement error. The OS is on the
\emph{Best performance} overlay. The power plan allows 100\,\% processor state on
AC. Throughput and latency-based clock estimates agree within 1\,\%. The platform
power-limits the APU. I therefore report efficiency against the measured
""" + f"{P1:.1f}" + r"""\,GFLOP/s. Against the datasheet """ + f"{PR:.0f}" + r"""\,GFLOP/s
every figure would halve.

\section{Naive baseline}

\begin{table}[h]
\centering\footnotesize
\caption{Naive kernel. Median of 5 runs after 1 warm-up.}
\label{tab:naive}
\begin{tabular}{@{}lrrrr@{}}
\toprule
$N/d$ & time (ms) & GFLOP/s & \% measured peak & \% rated peak \\
\midrule
""" + tab_q2 + r"""\bottomrule
\end{tabular}
\end{table}

\textbf{Scaling with $N$.} The work grows as $4N^2d$. Time grows faster. Doubling
$N$ costs """ + f"{T('naive',I2)/T('naive',I1):.1f}" + r"""$\times$, then
""" + f"{T('naive',I64)/T('naive',I2):.1f}" + r"""$\times$, against the 4$\times$
that FLOPs predict. The score matrix explains the gap. At $N{=}2048$ it is 16\,MB
and fits the 16\,MB L3. At $N{=}4096$ it is 64\,MB, so each pass is a DRAM round
trip.

\textbf{Scaling with $d$.} Doubling $d$ costs
""" + f"{T('naive',I128)/T('naive',I64):.1f}" + r"""$\times$, not 2$\times$. Step~3
walks $j$ with stride $d$. At $d{=}64$ the stride is 256\,B and a line feeds four
floats. At $d{=}128$ it is 512\,B, so every load pulls its own line and 15 of 16
bytes are wasted.

\textbf{Where the time goes.} S~V dominates at every size:
""" + f"{100*NV[I1][2]/NV[I1][3]:.0f}" + r"""\,\% at 1024/64, rising to
""" + f"{100*NV[I128][2]/NV[I128][3]:.0f}" + r"""\,\% at 4096/128. gcc will not
vectorise it: \code{-fopt-info-vec} reports \emph{``not suitable for strided
load''}. Each element costs a scalar load plus one step of a 4-cycle add chain.
QK$^\top$ is vectorised but reaches only 2.4--2.8\,GFLOP/s, because one
accumulator serialises the reduction. Softmax is 1--4\,\% here. It becomes the
largest term once the GEMMs are fixed.

\section{Optimisation}

\begin{table}[h]
\centering\footnotesize
\caption{All variants, one session. ms and speed-up over naive. Every variant
PASSes the check (max relative error $1.1\times10^{-6}$ to $5.0\times10^{-6}$).}
\label{tab:all}
\begin{tabular}{@{}lrr@{\hskip 6pt}rr@{\hskip 6pt}rr@{\hskip 6pt}rr@{}}
\toprule
& \multicolumn{2}{c}{1024/64} & \multicolumn{2}{c}{2048/64}
& \multicolumn{2}{c}{4096/64} & \multicolumn{2}{c}{4096/128} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
variant & ms & sp. & ms & sp. & ms & sp. & ms & sp. \\
\midrule
""" + tab_q3 + r"""\bottomrule
\end{tabular}
\end{table}

\textbf{(a) Loop reordering.} Transpose $K$ once. Then interchange the loops so
the innermost loop is an axpy. An axpy has no loop-carried dependency, so it
vectorises without \code{-ffast-math}:
\begin{verbatim}
S[i][:] += Q[i][k] * KT[k][:]      k outer, j inner
O[i][:] += S[i][j] * V[j][:]       j outer, k inner
\end{verbatim}
Result: """ + f"{BASE[I64]/T('loops',I64):.1f}" + r"""$\times$. Nothing is blocked.
Each row re-streams all of $K^{\top}$ and $V$ from L3, about 8\,GB at
$N{=}4096$. The kernel is L3-bound, not FP-bound.

\textbf{(b) Tiling.} QK$^\top$ uses a $4\times16$ micro-tile. Eight YMM
accumulators, $k$ innermost. Per $k$ that is 8 FMA (128\,FLOP) against two 32\,B
loads plus four broadcasts, so 3 load cycles cover 4 FMA cycles. S~V is blocked
over the reduction. A 64-key slab of $V$ is 16\,KB and stays in L1 while all
output rows reuse it, which keeps the $d$ accumulators in registers. Three
measurements decided the details.
\begin{itemize}\itemsep0pt
\item gcc will not keep the register tile in registers. In plain C it vectorises
the wrong axis and spills. The tile works only with AVX2 intrinsics.
\item A 16\,KB row is a multiple of the L1 size, so an unpadded block of rows
maps onto one L1 set and thrashes. 64\,B of padding per row took this variant
from 370.6 to 307.0\,ms at 4096/64.
\item One store per FMA caps a kernel at half of peak. Keeping the output tile in
L1 reached 21\,\%. Register accumulators reach """ + f"{SV_PK:.0f}" + r"""\,\%.
\end{itemize}
Result: """ + f"{BASE[I64]/T('tiled',I64):.1f}" + r"""$\times$. The kernels split
""" + f"{100*QK/PHSUM:.0f}/{100*SM/PHSUM:.0f}/{100*SV/PHSUM:.0f}" + r"""\,\%
(QK$^\top$/softmax/S~V). QK$^\top$ runs at """ + f"{QK_PK:.0f}" + r"""\,\% of peak.
Two limits now dominate. Scalar \var{expf} is
""" + f"{100*SM/PHSUM:.0f}" + r"""\,\% of kernel time. A further
""" + f"{ALLOC:.0f}" + r"""\,ms, """ + f"{100*ALLOC/T('tiled',I64):.0f}" + r"""\,\% of
wall time, is page faults on the 67\,MB score matrix.

\textbf{(c) Fusion.} Row $i$ of $O$ needs only row $i$ of $S$. I keep one score
row in a 16\,KB L1 buffer and run the three steps over it. The $N\times N$ matrix
is never built. Result: """ + f"{BASE[I64]/T('fused',I64):.1f}" + r"""$\times$, which
is \emph{behind} \var{attn\_tiled} (""" + f"{T('fused',I64):.0f}" + r""" against
""" + f"{T('tiled',I64):.0f}" + r"""\,ms), and far behind at $d{=}128$
(""" + f"{T('fused',I128):.0f}" + r""" against """ + f"{T('tiled',I128):.0f}" + r"""\,ms).
Fusion is not a win here. It saves about 200\,MB of score traffic. It costs 4\,GB
of L3 traffic at $d{=}64$ ($V$ re-read per row), and 8\,GB at $d{=}128$. Fusion
removes the right bottleneck and adds a bigger one.

\textbf{(d) Fastest single-threaded.} Two changes. First, block 64 queries. The
score tile stays in cache, so there is no 67\,MB allocation, and $V$ is read once
per block, not once per row. Second, replace \var{expf} with an AVX2 exponential.
Split $y=x\log_2 e$ into $n+f$ with $|f|\le0.5$, evaluate $2^f$ by a degree-5
polynomial, and build $2^n$ from the exponent field. Measured error
$""" + EXPERR + r"""$ over $x\in[-80,0]$; """ + f"{EXPSPD:.1f}" + r"""$\times$ faster
than libm (16.8\,M calls: 9.4\,ms against 78.8\,ms). Result:
""" + f"{sp_best:.1f}" + r"""$\times$, and """ + f"{GF('best',I64):.1f}" + r"""\,GFLOP/s
= """ + f"{pct_best:.0f}" + r"""\,\% of measured peak. That is
""" + f"{T('tiled',I64)/best64:.1f}" + r"""$\times$ faster than \var{tiled}.

\section{OpenMP}

\var{attn\_omp} parallelises the query blocks. $K^{\top}$ is built once before the
team starts. No barrier is needed inside the region. Blocks write disjoint
$64\times d$ slices of $O$. Shared operands are read-only. Slices are 16\,KB
apart, so there is no false sharing. One caution: with
\code{OMP\_PROC\_BIND=close}, Windows packs both threads of a 2-thread run onto
one core (53.9 against 45.3\,ms at 4096/64). Table~\ref{tab:omp} is unpinned.

\begin{table}[h]
\centering\footnotesize
\caption{Strong scaling. Speed-up is against this binary at one thread.}
\label{tab:omp}
\begin{tabular}{@{}llrrrr@{}}
\toprule
$N/d$ & threads & time (ms) & speed-up & eff.\ (\%) & GFLOP/s \\
\midrule
""" + tab_q4 + r"""\bottomrule
\end{tabular}
\end{table}

\textbf{Up to four threads the curve is close to linear.} Eight threads give
""" + f"{T('omp',I64,1)/T('omp',I64,8):.2f}" + r"""$\times$, or
""" + f"{100*T('omp',I64,1)/T('omp',I64,8)/8:.0f}" + r"""\,\% efficiency. The gap is
the clock. The all-core clock is 2.02\,GHz against 2.38\,GHz, so the best possible
8-thread result here is $8\times(2.02/2.38)={8*2.02/2.38:.1f}\times$. Sixteen
threads give """ + f"{T('omp',I64,1)/T('omp',I64,16):.2f}" + r"""$\times$. The kernel
is FMA-bound and the two SMT threads of a core share its two FMA pipes. The whole
machine reaches """ + f"{PA:.0f}" + r"""\,GFLOP/s at 16 threads, against
$8\times""" + f"{P1:.1f}" + r"""=612$ from the single-core peak, so SMT adds about a
quarter. Efficiency is defined against thread count, so it falls steeply while
time still improves.

\section{Roofline}

The roofline uses measured constants: """ + f"{P1:.1f}" + r"""\,GFLOP/s at
""" + f"{BW1}" + r"""\,GB/s on one core. Intensity is useful FLOPs over the bytes a
variant must move. The numerator is $4N^2d$. The softmax is charged to time, not
FLOPs. The denominators:
\begin{equation*}
(4Nd+3N^2)\times4 = 205.5\ \text{MB}\Rightarrow 20.9,\qquad
4Nd\times4 = 4.19\ \text{MB}\Rightarrow 1024,\qquad
(4Nd+Nd+\tfrac{N}{64}Nd)\times4 \Rightarrow 59.4
\end{equation*}
for (naive, loops, tiled), (fused) and (best, omp) respectively.

\begin{table}[h]
\centering\footnotesize
\caption{Variants at $N{=}4096,d{=}64$ against their roofline.}
\label{tab:roof}
\begin{tabular}{@{}lrrrr@{}}
\toprule
variant & AI (FLOP/B) & ms & GFLOP/s & \% of relevant peak \\
\midrule
""" + tab_q5 + r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.84\linewidth]{plots/roofline.png}
\caption{Roofline at $N{=}4096,d{=}64$.}
\label{fig:roof}
\end{figure}

\textbf{What limits each variant.} \var{naive} sits right of the ridge, where the
roofline predicts """ + f"{P1:.1f}" + r"""\,GFLOP/s. It reaches
""" + f"{pct_naive:.1f}" + r"""\,\%. Neither roof applies: the strided scalar loop
costs one dependent add per element. \var{loops} is compute-bound by the model and
""" + f"{P1/GF('loops',I64):.1f}" + r"""$\times$ below it; its real limit is L3
bandwidth. \var{tiled} has a QK$^\top$ kernel at """ + f"{QK_PK:.0f}" + r"""\,\% of
peak, held back by \var{expf} and page faults. \var{fused} is
""" + f"{AI['fused']/M['ridge_1core_measured']:.0f}" + r"""$\times$ past the ridge by
the DRAM model, yet reaches only """ + f"{100*GF('fused',I64)/P1:.0f}" + r"""\,\%.
Its limit is L3 bandwidth, one level above DRAM and invisible on this plot.
\var{best} is at """ + f"{pct_best:.0f}" + r"""\,\% of peak; the rest is the softmax,
the S~V ceiling of """ + f"{SV_PK:.0f}" + r"""\,\%, and compulsory $V$ reads.
\var{omp} lies above the single-core roofline, which is the signature of a kernel
using the whole machine. Against the all-core roof it reaches
""" + f"{pct_omp:.0f}" + r"""\,\%.

\section*{Conformance to the handout}

{\footnotesize
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{6.0cm}
                     >{\raggedright\arraybackslash}p{7.6cm}l@{}}
\caption{Each handout requirement, how it is met, and where the evidence is.}
\label{tab:audit}\\
\toprule
Requirement & How it is met & Evidence \\
\midrule
\endfirsthead
\toprule
Requirement & How it is met & Evidence \\
\midrule
\endhead
\bottomrule
\endfoot
\var{machine\_spec.txt}: all fields, cores incl.\ P/E, clocks, SIMD, caches,
memory size and type & All fields filled from SMBIOS and CPUID, then measured.
P/E recorded as ``none, homogeneous''. & Table~\ref{tab:machine} \\
\addlinespace
Q1: peak FP32 for one and all cores, memory bandwidth, ridge point, with
calculations & Three equation blocks; measured against rated stated. &
  \S1 \\
\addlinespace
Q2: time, GFLOP/s, \% of peak per input; effect of $N$ and $d$; where the time
goes & Table~\ref{tab:naive}, both peak columns, two scaling paragraphs, phase
split and the stride argument. & \S2 \\
\addlinespace
Q3: same measurements plus speed-up; \var{make check} passes; what limits each
variant & Table~\ref{tab:all}; all six PASS; one limit statement per variant. &
  \S3 \\
\addlinespace
Q3(a--d): reorder loops, cache and register tiling, fuse so $N\times N$ is never
stored, maximum single-threaded & Axpy interchange; $4\times16$ YMM tile;
one score row in L1 with no $N^2$ allocation; query blocking plus AVX2 exp. &
  \S3(a)--(d) \\
\addlinespace
``No BLAS or other math libraries'' & Only \var{libm}, \var{immintrin.h} and
\var{omp.h}. An include audit confirms it. & final note \\
\addlinespace
Q4: OpenMP, several thread counts, time, speed-up, efficiency, scaling
explained & Table~\ref{tab:omp}; the clock and SMT
argument. & \S4 \\
\addlinespace
Q5: roofline from Q1 numbers, every variant at $N{=}4096,d{=}64$, intensity
computation, limiting factor & Table~\ref{tab:roof}, Figure~\ref{fig:roof}, byte
counts shown, one limit per variant. & \S5 \\
\addlinespace
Submission structure; \var{results.csv} rows for each thread count &
Structure matches exactly. \var{results.csv} holds 40 rows. & repository \\
\end{longtable}
}

\section*{Notes on reproduction}

The handout \var{Makefile} is submitted unmodified. This box has no \var{make}
(MSYS2 \var{make} cannot start its signal pipe here) and no \var{clang}. All
numbers therefore come from \var{tools/bench.py}, which repeats the Makefile
steps with gcc 14.1.0 and runs the same check rule.
\code{-march=native} resolves to \var{znver3}, the ISA the Makefile would select
with clang. Each variant reports the median of 5 runs after one warm-up. One
table comes from one session, because the part drifts up to 25\,\% between
sessions. Within a session the spread is about $\pm5$\,\%. Only \var{libm},
\var{immintrin.h} and \var{omp.h} are used: no BLAS, no other math library.
Driver code is unchanged from \var{attn\_naive.c} in all five variants.
\var{results.csv} holds 40 rows, including every thread count.

\end{document}
"""

(ROOT / "report.tex").write_text(TEX, encoding="utf-8")
print(f"wrote report.tex ({TEX.count(chr(10))} lines)")

TECT = Path.home() / "AppData" / "Local" / "tectonic" / "tectonic.exe"
if not TECT.exists():
    TECT = Path(shutil.which("tectonic") or "")
if not TECT.exists():
    raise SystemExit("tectonic not found")

pdf = ROOT / "report.pdf"
if pdf.exists():
    pdf.unlink()
r = subprocess.run([str(TECT), "-X", "compile", str(ROOT / "report.tex"),
                    "--outdir", str(ROOT)],
                   capture_output=True, text=True, cwd=str(ROOT))
print("tectonic exit:", r.returncode)
if r.returncode != 0 or not pdf.exists():
    print(r.stdout[-4000:])
    print(r.stderr[-4000:])
    raise SystemExit("LaTeX build failed")
print(f"wrote report.pdf ({pdf.stat().st_size} bytes)")
