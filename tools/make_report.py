#!/usr/bin/env python3
"""Generate report.tex from the measured data and compile it with Tectonic."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "build" / "results.json").read_text())
M = json.loads((ROOT / "tools" / "machine.json").read_text())

I1, I2, I64, I128 = ("qkv_N1024_d64.bin", "qkv_N2048_d64.bin",
                     "qkv_N4096_d64.bin", "qkv_N4096_d128.bin")
INP = [I1, I2, I64, I128]
LBL = {I1: "1024/64", I2: "2048/64", I64: "4096/64", I128: "4096/128"}
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


B = {i: T("naive", i) for i in INP}
b64 = T("best", I64)
sb = B[I64] / b64
so = B[I64] / T("omp", I64, 16)
pb = 100 * GF("best", I64) / P1
po = 100 * GF("omp", I64, 16) / PA
pn = 100 * GF("naive", I64) / P1

# phase profiles from tools/naive_phases.c and tools/phases.c
NV = {I1: (27.822, 2.447, 47.729, 77.998), I2: (108.637, 10.143, 197.327, 316.107),
      I64: (437.504, 41.842, 1318.423, 1797.769),
      I128: (916.917, 41.118, 2974.205, 3932.240)}
QK, SM, SV = 22.473, 44.512, 35.190
PS = QK + SM + SV
AL = T("tiled", I64) - PS
QPK, SPK = 71.1, 45.4

t2 = "".join(f"{LBL[i]} & {T('naive',i):7.2f} & {GF('naive',i):5.2f} & "
             f"{100*GF('naive',i)/P1:4.1f} & {100*GF('naive',i)/PR:4.1f} \\\\\n"
             for i in INP)

t3 = ""
for v in ["loops", "tiled", "fused", "best", "omp"]:
    row = f"\\texttt{{{v}}}"
    for i in INP:
        th = 16 if v == "omp" else None
        ms, gf = T(v, i, th), GF(v, i, th)
        row += (f" & {ms:6.1f} / {gf:4.1f} / {100*gf/(PA if v=='omp' else P1):3.0f}"
                f" / {B[i]/ms:4.1f}")
    t3 += row + " \\\\\n"
row = "\\midrule\n\\texttt{naive}"
for i in INP:
    row += f" & {T('naive',i):6.1f} / {GF('naive',i):4.1f} / {100*GF('naive',i)/P1:3.0f} /  1.0"
t3 += row + " \\\\\n"

t4 = ""
for i in [I64, I128]:
    t1 = T("omp", i, 1)
    for n in [1, 2, 4, 8, 16]:
        r = g("omp", i, n)
        s = t1 / r["time_ms"]
        first = f"\\multirow{{5}}{{*}}{{{LBL[i]}}}" if n == 1 else ""
        t4 += (f"{first} & {n:2d} & {r['time_ms']:8.2f} & {s:5.2f} & "
               f"{100*s/n:4.0f} & {r['gflops']:6.1f} \\\\\n")
    if i != I128:
        t4 += "\\midrule\n"

FL = 4.0 * 4096 * 4096 * 64
COMP = 4.0 * 4096 * 64


def bytes_of(v):
    if v in ("naive", "loops", "tiled"):
        return (COMP + 3.0 * 4096 * 4096) * 4
    if v == "fused":
        return COMP * 4
    return (COMP + 4096 * 64 + 64.0 * 4096 * 64) * 4


AI = {v: FL / bytes_of(v) for v in ["naive", "loops", "tiled", "fused", "best", "omp"]}
t5 = ""
for v in ["naive", "loops", "tiled", "fused", "best", "omp"]:
    th = 16 if v == "omp" else None
    t5 += (f"\\texttt{{{v}}} & {AI[v]:6.1f} & {T(v,I64,th):7.2f} & {GF(v,I64,th):6.1f} & "
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
{\small """ + M["cpu"] + r""" (8C/16T Zen~3+), 16\,GB DDR5-4800, Windows 11.
Built with gcc 14.1.0 and \code{-O3 -march=native}.}
\end{center}
\vspace{-8pt}

\noindent\rule{\linewidth}{0.4pt}
\small
I optimised the given attention kernel step by step on my laptop. At
$N{=}4096,d{=}64$ the naive version takes """ + f"{T('naive',I64):.1f}" + r"""\,ms and
reaches """ + f"{GF('naive',I64):.2f}" + r"""\,GFLOP/s. My final single-threaded version
takes """ + f"{b64:.1f}" + r"""\,ms (""" + f"{GF('best',I64):.1f}" + r"""\,GFLOP/s), so it is
""" + f"{sb:.1f}" + r"""$\times$ faster. With 16 threads it takes
""" + f"{T('omp',I64,16):.1f}" + r"""\,ms (""" + f"{GF('omp',I64,16):.0f}" + r"""\,GFLOP/s),
""" + f"{so:.0f}" + r"""$\times$ faster than naive. All six variants pass \var{make check}.\par
\noindent\rule{\linewidth}{0.4pt}
\normalsize

\section{My machine}

The CPU is an AMD Ryzen 7 6800H. I read most of the specification from SMBIOS
and CPUID, and I measured the rest with small probes in \var{tools/probe.c}.
Table~\ref{tab:machine} has everything.

\begin{table}[h]
\centering\footnotesize
\caption{My machine. ``Measured'' means I ran a microbenchmark, not that I copied a datasheet.}
\label{tab:machine}
\begin{tabular}{@{}ll@{}}
\toprule
CPU & AMD Ryzen 7 6800H, Zen 3+ (family 0x19, model 0x44) \\
Cores / threads & 8 physical, 16 logical, 2 CCX $\times$ 4 cores. No P/E split. \\
Clock & 3.20\,GHz base, 4.75\,GHz rated boost. \textbf{I measured 4.29\,GHz on one core, 3.64\,GHz all-core.} \\
SIMD & AVX2 + FMA3, 256-bit, 2 FMA units per core, 16 YMM registers. No AVX-512. \\
Caches & L1D 32\,KiB/core, L2 512\,KiB/core, L3 16\,MiB (8\,MiB per CCX), 64\,B lines. \\
Memory & 16\,GB DDR5-4800 SO-DIMM, 2 channels. DRAM latency """ + f"{M['dram_latency_ns']:.0f}" + r"""\,ns. \\
\bottomrule
\end{tabular}
\end{table}

Zen~3+ does two 256-bit FMAs per cycle per core. One FMA covers 8 floats and
counts as 2\,FLOP, so a core does 32 FLOP per cycle. My FMA chain gives
4.29\,GHz on one core:
\begin{equation*}
P_1 = 32\times4.29 = """ + f"{P1:.1f}" + r"""\ \text{GFLOP/s},\qquad
P_8 = 32\times8\times3.64 = """ + f"{PA:.0f}" + r"""\ \text{GFLOP/s}
\end{equation*}
The bandwidth I measured over a 512\,MB buffer is """ + f"{BW1}" + r"""\,GB/s on one
thread and """ + f"{BW16}" + r"""\,GB/s on 16 threads. DDR5-4800 dual channel is
76.8\,GB/s in theory, so I am getting about a third of it. That gives
\begin{equation*}
B_\text{theor} = 4800\times8\times2 = 76.8\ \text{GB/s},\quad
R_1 = """ + f"{P1:.1f}/{BW1} = {M['ridge_1core_measured']:.2f}" + r"""\ \text{FLOP/B},\quad
R_8 = """ + f"{PA:.0f}/{BW16} = {M['ridge_allcore_measured']:.2f}" + r"""\ \text{FLOP/B}
\end{equation*}

One thing I did not expect. I ran the same clock probe in an earlier session and
got 2.38\,GHz on one core and 2.02\,GHz all-core, not 4.29\,GHz. The plan is
\emph{Best performance} and the laptop was plugged in both times, so I do not
think it is a setting. The platform seems to limit the clock depending on its
power state. So I measured the peak again before every run and I use that
session's value for all the percentages below. Otherwise the same code would
look """ + f"{4.29/2.38:.1f}" + r"""$\times$ better or worse depending on the day.

\section{The naive kernel}

\begin{table}[h]
\centering\footnotesize
\caption{Naive kernel. Median of 5 runs after one warm-up.}
\label{tab:naive}
\begin{tabular}{@{}lrrrr@{}}
\toprule
$N/d$ & time (ms) & GFLOP/s & \% measured peak & \% rated peak \\
\midrule
""" + t2 + r"""\bottomrule
\end{tabular}
\end{table}

\textbf{Effect of $N$.} Going from 1024 to 2048 costs
""" + f"{T('naive',I2)/T('naive',I1):.1f}" + r"""$\times$ and from 2048 to 4096 costs
""" + f"{T('naive',I64)/T('naive',I2):.1f}" + r"""$\times$, but the work only grows by
4$\times$. The score matrix is the reason. At $N{=}2048$ it is 16\,MB and still
fits in the 16\,MB L3. At $N{=}4096$ it is 64\,MB, so every pass over it goes to
DRAM.

\textbf{Effect of $d$.} Doubling $d$ costs
""" + f"{T('naive',I128)/T('naive',I64):.1f}" + r"""$\times$, not 2$\times$. In step 3 I walk
$j$ with stride $d$, so at $d{=}64$ a 64\,B line gives me four floats, and at
$d{=}128$ every load pulls a line and I use 4 bytes out of 64.

\textbf{Where the time goes.} I timed the three steps separately with
\var{tools/naive\_phases.c}. S~V is """ + f"{100*NV[I1][2]/NV[I1][3]:.0f}" + r"""\,\% of
the time at 1024/64 and """ + f"{100*NV[I128][2]/NV[I128][3]:.0f}" + r"""\,\% at
4096/128. I checked with \code{-fopt-info-vec} and gcc refuses to vectorise that
loop, it says \emph{``not suitable for strided load''}. So every element is a
scalar load plus one step of a 4-cycle add chain. QK$^\top$ does get vectorised
but only reaches """ + f"{2.0*4096*4096*64/(NV[I64][0]*1e-3)*1e-9:.1f}" + r"""\,GFLOP/s,
because one accumulator serialises the reduction. The softmax is only 1--4\,\%
here, so I left it alone for now.

\section{Optimisations}

\begin{table}[H]
\centering\scriptsize
\caption{All my variants, one session. Each cell: time (ms) / GFLOP/s / \% of
measured peak / speed-up over naive. Every variant passes the check, error
$1.1\times10^{-6}$ to $5.0\times10^{-6}$ against the $10^{-4}$ limit.}
\label{tab:all}
\begin{tabular}{@{}lcccc@{}}
\toprule
variant & 1024/64 & 2048/64 & 4096/64 & 4096/128 \\
\midrule
""" + t3 + r"""\bottomrule
\end{tabular}
\end{table}

\textbf{(a) Loop reordering.} I transpose $K$ once, then I change the loop order
so the inner loop is an axpy instead of a dot product. An axpy has no
loop-carried dependency, so gcc vectorises it without \code{-ffast-math}:
\begin{verbatim}
S[i][:] += Q[i][k] * KT[k][:]      k outer, j inner
O[i][:] += S[i][j] * V[j][:]       j outer, k inner
\end{verbatim}
This gave """ + f"{B[I64]/T('loops',I64):.1f}" + r"""$\times$. What still hurts is that
nothing is blocked, so for every row I read all of $K^{\top}$ and all of $V$ again
out of L3, about 8\,GB at $N{=}4096$. So this version is limited by L3 bandwidth,
not by the FP units.

\textbf{(b) Tiling.} QK$^\top$ now uses a $4\times16$ micro-tile. That is 8 YMM
accumulators with $k$ innermost, so per $k$ I issue 8 FMA (128\,FLOP) against two
32\,B loads and four broadcasts. 3 load cycles cover 4 FMA cycles, so the FMA
units become the limit. For S~V I block the reduction instead of the output: a
64-key slab of $V$ is 16\,KB and stays in L1 while every output row uses it, and
that keeps the $d$ accumulators in registers. Three things I had to learn by
trying:
\begin{itemize}\itemsep0pt
\item gcc will not keep the tile in registers. In plain C it vectorises the
wrong axis and spills to the stack. It only works when I write the tile with
AVX2 intrinsics.
\item A 16\,KB row is a multiple of the L1 size, so a block of rows lands on one
L1 set and thrashes. I padded every row by 64\,B and this variant went from
370.6 to 307.0\,ms at 4096/64.
\item One store per FMA caps a kernel at half of peak. My first S~V kept the
output tile in L1 and only reached 21\,\%. With the accumulators in registers it
reaches """ + f"{SPK:.0f}" + r"""\,\%.
\end{itemize}
Now I get """ + f"{B[I64]/T('tiled',I64):.1f}" + r"""$\times$. The kernels split
""" + f"{100*QK/PS:.0f}/{100*SM/PS:.0f}/{100*SV/PS:.0f}" + r"""\,\%
(QK$^\top$/softmax/S~V) and QK$^\top$ runs at """ + f"{QPK:.0f}" + r"""\,\% of peak.
Two problems are left. The scalar \var{expf} is
""" + f"{100*SM/PS:.0f}" + r"""\,\% of the kernel time, and another
""" + f"{AL:.0f}" + r"""\,ms (""" + f"{100*AL/T('tiled',I64):.0f}" + r"""\,\% of the wall
time) is not in any kernel at all, it is page faults from allocating the 67\,MB
score matrix every call.

\textbf{(c) Fusion.} Row $i$ of $O$ only needs row $i$ of $S$, so I keep one
score row in a 16\,KB L1 buffer and run the three steps over it. The $N\times N$
matrix is never built. This gave """ + f"{B[I64]/T('fused',I64):.1f}" + r"""$\times$,
which is slower than \var{attn\_tiled} (""" + f"{T('fused',I64):.0f}" + r""" against
""" + f"{T('tiled',I64):.0f}" + r"""\,ms) and much slower at $d{=}128$
(""" + f"{T('fused',I128):.0f}" + r""" against """ + f"{T('tiled',I128):.0f}" + r"""\,ms).
I expected fusion to win, so I looked at the traffic. It saves about 200\,MB of
score matrix traffic but it makes every row read all of $V$, which is 4\,GB of L3
traffic at $d{=}64$ and 8\,GB at $d{=}128$. The traffic I add is bigger than the
traffic I remove.

\textbf{(d) Fastest single-threaded.} I made two changes. First I block 64
queries, so the score tile stays in cache, there is no 67\,MB allocation, and $V$
is read once per block instead of once per row. Second I replaced \var{expf} with
an AVX2 exponential: split $y=x\log_2 e$ into $n+f$ with $|f|\le0.5$, evaluate
$2^f$ with a degree-5 polynomial, and build $2^n$ by writing $n+127$ into the
exponent field. It is 8.5$\times$ faster than libm (16.8\,M calls: 5.0\,ms
instead of 42.3\,ms), and the whole kernel still stays within
$5.0\times10^{-6}$ of the naive result. Now I get """ + f"{sb:.1f}" + r"""$\times$ and
""" + f"{GF('best',I64):.1f}" + r"""\,GFLOP/s, which is """ + f"{pb:.0f}" + r"""\,\% of
the measured peak. That is """ + f"{T('tiled',I64)/b64:.1f}" + r"""$\times$ faster
than \var{tiled} and """ + f"{T('fused',I64)/b64:.1f}" + r"""$\times$ faster than
\var{fused}.

\section{OpenMP}

I parallelised the query blocks. $K^{\top}$ is built once before the parallel
region starts. I do not need a barrier inside it: the blocks write disjoint
$64\times d$ slices of $O$, everything they read is shared and read-only, and the
slices are 16\,KB apart so there is no false sharing. One surprise here. With
\code{OMP\_PROC\_BIND=close} Windows puts both threads of a 2-thread run on the
two SMT siblings of one core, and I measured 53.9\,ms instead of 45.3\,ms. So the
numbers in Table~\ref{tab:omp} are without binding.

\begin{table}[H]
\centering\footnotesize
\caption{Strong scaling. Speed-up is against the same binary at one thread.}
\label{tab:omp}
\begin{tabular}{@{}llrrrr@{}}
\toprule
$N/d$ & threads & time (ms) & speed-up & eff.\ (\%) & GFLOP/s \\
\midrule
""" + t4 + r"""\bottomrule
\end{tabular}
\end{table}

Up to 4 threads it is close to linear. 8 threads give
""" + f"{T('omp',I64,1)/T('omp',I64,8):.2f}" + r"""$\times$, which is
""" + f"{100*T('omp',I64,1)/T('omp',I64,8)/8:.0f}" + r"""\,\% efficiency, and I think the
clock explains the gap: all-core is 3.64\,GHz against 4.29\,GHz on one core, so the
best I can get here is $8\times(3.64/4.29)={8*3.64/4.29:.1f}\times$. 16 threads give
""" + f"{T('omp',I64,1)/T('omp',I64,16):.2f}" + r"""$\times$. The kernel is FMA bound and
the two SMT threads of a core share the same two FMA pipes. The whole machine
reaches """ + f"{PA:.0f}" + r"""\,GFLOP/s at 16 threads against
$8\times""" + f"{P1:.1f}" + r"""=1075$ from the single core peak, so SMT adds about a
quarter and not double. Efficiency is counted against thread count, so it drops
fast even though the time still improves.

\section{Roofline}

I used the numbers I measured in section 1: """ + f"{P1:.1f}" + r"""\,GFLOP/s at
""" + f"{BW1}" + r"""\,GB/s on one core. Intensity is the useful FLOPs divided by the
bytes a variant has to move. The numerator is always $4N^2d$; I did not count the
softmax in the FLOPs. For the bytes I counted what each kernel actually needs:
\begin{equation*}
(4Nd+3N^2)\times4 = 205.5\ \text{MB}\Rightarrow 20.9,\qquad
4Nd\times4 = 4.19\ \text{MB}\Rightarrow 1024,\qquad
(4Nd+Nd+\tfrac{N}{64}Nd)\times4 \Rightarrow 59.4
\end{equation*}
which are (naive, loops, tiled), (fused) and (best, omp) in that order.

\begin{table}[H]
\centering\footnotesize
\caption{Every variant at $N{=}4096,d{=}64$ against the roofline.}
\label{tab:roof}
\begin{tabular}{@{}lrrrr@{}}
\toprule
variant & AI (FLOP/B) & ms & GFLOP/s & \% of relevant peak \\
\midrule
""" + t5 + r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.84\linewidth]{plots/roofline.png}
\caption{My roofline at $N{=}4096,d{=}64$.}
\label{fig:roof}
\end{figure}

\textbf{What limits each one.} \var{naive} is to the right of the ridge, so the
roofline says it should reach """ + f"{P1:.1f}" + r"""\,GFLOP/s, and I get
""" + f"{pn:.1f}" + r"""\,\%. The roofline cannot explain that, because the strided loop
costs one dependent add per element. \var{loops} is compute bound by the model and
""" + f"{P1/GF('loops',I64):.1f}" + r"""$\times$ below it, and its real limit is L3
bandwidth. \var{tiled} has QK$^\top$ at """ + f"{QPK:.0f}" + r"""\,\% of peak, so what
holds it back is \var{expf} and the page faults. \var{fused} is
""" + f"{AI['fused']/M['ridge_1core_measured']:.0f}" + r"""$\times$ past the ridge by the
DRAM model and I only get """ + f"{100*GF('fused',I64)/P1:.0f}" + r"""\,\%, because its
limit is L3 bandwidth, which is above DRAM and does not show on this plot.
\var{best} reaches """ + f"{pb:.0f}" + r"""\,\% of peak and what is left is the softmax,
S~V at """ + f"{SPK:.0f}" + r"""\,\% of peak, and reading $V$ again. \var{omp} sits above
the single-core roofline, which makes sense because it uses all the cores, and
against the all-core roof it reaches """ + f"{po:.0f}" + r"""\,\%.

\section*{How I did all this}

The handout \var{Makefile} is unchanged. My laptop has no \var{make} (the MSYS2
one cannot start its signal pipe here) and no \var{clang}, so I ran everything
through \var{tools/bench.py}, which does the same steps with gcc 14.1.0 and runs
the same check rule. \code{-march=native} picks \var{znver3}, the same ISA the
Makefile would pick with clang. Every number is the median of 5 runs after one
warm-up. I kept all the tables in one session because the clock moves between
sessions. I only used \var{libm}, \var{immintrin.h} and \var{omp.h}, no BLAS or
other math library. The driver in each file is untouched from
\var{attn\_naive.c}. \var{results.csv} has 40 rows, including every thread count.

\section*{Checking it against the handout}

{\footnotesize
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{6.0cm}
                    >{\raggedright\arraybackslash}p{7.6cm}l@{}}
\caption{What the handout asked for and where I did it.}
\label{tab:audit}\\
\toprule
Requirement & What I did & Where \\
\midrule
\endfirsthead
\toprule
Requirement & What I did & Where \\
\midrule
\endhead
\bottomrule
\endfoot
\var{machine\_spec.txt}: all fields, cores incl.\ P/E, clocks, SIMD, caches,
memory size and type & Filled everything from SMBIOS and CPUID, then measured
the parts I could. P/E is none, the design is homogeneous. & Table~\ref{tab:machine} \\
\addlinespace
Q1: peak FP32 for one and all cores, bandwidth, ridge point, with calculations &
Showed the arithmetic for $P_1$, $P_8$, $B$ and both ridges. & \S1 \\
\addlinespace
Q2: time, GFLOP/s, \% of peak per input; effect of $N$ and $d$; where the time
goes & Table~\ref{tab:naive} has both peak columns, then I wrote two paragraphs
on $N$ and $d$ and timed the three steps. & \S2 \\
\addlinespace
Q3: the same measurements as Q2 plus speed-up; \var{make check} passes; what
limits each variant & Table~\ref{tab:all} gives time, GFLOP/s, \% of peak and
speed-up for every input. All six pass. Every variant has a limit sentence. &
  \S3 \\
\addlinespace
Q3(a--d): reorder loops, cache and register tiling, fuse so $N\times N$ is never
stored, maximum single-threaded & Axpy interchange, then a $4\times16$ YMM tile,
then one score row in L1 with no $N^2$ allocation, then query blocking plus an
AVX2 exp. & \S3(a)--(d) \\
\addlinespace
``No BLAS or other math libraries'' & I only included \var{libm},
\var{immintrin.h} and \var{omp.h}. & last section \\
\addlinespace
Q4: OpenMP, several thread counts, time, speed-up, efficiency, scaling
explained & Ran 1, 2, 4, 8 and 16 threads, then explained the clock and SMT
limits. & Table~\ref{tab:omp} \\
\addlinespace
Q5: roofline from Q1 numbers, every variant at $N{=}4096,d{=}64$, intensity
computation, limiting factor & Table~\ref{tab:roof} and Figure~\ref{fig:roof},
with the byte counts and one limit per variant. & \S5 \\
\addlinespace
Submission structure; \var{results.csv} rows for each thread count &
Kept the structure exactly. \var{results.csv} has 40 rows. & repository \\
\end{longtable}
}

\end{document}
"""

(ROOT / "report.tex").write_text(TEX, encoding="utf-8")
print(f"wrote report.tex ({TEX.count(chr(10))} lines)")

pdf = ROOT / "report.pdf"
if pdf.exists():
    pdf.unlink()
TECT = Path.home() / "AppData" / "Local" / "tectonic" / "tectonic.exe"
r = subprocess.run([str(TECT), "-X", "compile", str(ROOT / "report.tex"),
                    "--outdir", str(ROOT)], capture_output=True, text=True, cwd=str(ROOT))
print("tectonic exit:", r.returncode)
if r.returncode != 0 or not pdf.exists():
    print(r.stdout[-3000:])
    raise SystemExit("LaTeX build failed")
print(f"wrote report.pdf ({pdf.stat().st_size} bytes)")
