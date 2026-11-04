"""Generate src/attn_omp.c from src/attn_best.c.

Same kernels; only attention() differs -- the query-block loop becomes a
parallel loop and each thread owns one score tile.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "src" / "attn_best.c").read_text()

DRV = "/* ------------------------------------------------------------------------- */\n/* Driver: do not change below this line."
head_end = src.index(DRV)
header_end = src.index("#define _POSIX_C_SOURCE")
attn_at = src.index("void attention(const float *Q")
body = src[header_end:attn_at]  # every kernel, but not best's attention()
driver = src[head_end:]

attn = r'''void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int KN = N + PAD, SN = N + PAD;
    const int nblk = (N + BI - 1) / BI;

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    float *restrict KT = malloc((size_t)KN * d * sizeof(float));
    float *restrict Sall = malloc((size_t)nt * BI * SN * sizeof(float));
    if (!KT || !Sall) {
        fprintf(stderr, "attn_omp: out of memory\n");
        exit(1);
    }

    for (int k = 0; k < d; k++)
        for (int i = 0; i < N; i++)
            KT[(size_t)k * KN + i] = K[(size_t)i * d + k];

    /* Blocks are independent: block b writes only O[b*BI ..) and reads Q, KT, V.
     * One score tile per thread, allocated up front so each thread first-touches
     * its own pages and no two threads share a cache line. */
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *S = Sall + (size_t)tid * BI * SN;
#pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < nblk; b++) {
            const int ib = b * BI;
            const int rows = (ib + BI <= N) ? BI : N - ib;
            attn_block(Q + (size_t)ib * d, KT, KN, V, O + (size_t)ib * d, S, SN, rows,
                       N, d, scale);
        }
    }

    free(KT);
    free(Sall);
}
'''

head = '''/*
 * attn_omp.c -- Q4: attn_best with OpenMP.
 *
 * The query-block loop is the parallel loop. Nothing inside the region needs a
 * barrier: blocks write disjoint 64 x d slices of O, every shared operand
 * (KT, V) is read-only, and the slices are 16 KB apart so there is no false
 * sharing. KT is built once before the team starts.
 *
 * Scaling stops at the cores, not at the threads: 8 physical cores at 2.02 GHz
 * all-core is the ceiling, so SMT adds ~25 % rather than 100 %.
 *
 * Generated from src/attn_best.c by tools/gen_omp.py so the kernels cannot drift.
 */
'''

out = head + body + attn + driver
(ROOT / "src" / "attn_omp.c").write_text(out)
print(f"  src/attn_omp.c  {out.count(chr(10)):4d} lines")
