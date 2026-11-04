/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * attn_loops keeps the loops vectorisable but streams a 1 MB operand out of L3
 * once per row. Both GEMMs become blocked GEMMs here.
 *
 * QK^T: 4x16 micro-tile = 8 YMM accumulators, k innermost. Per k that is 8 FMA
 * (128 FLOP) against 2x32 B of KT plus 4 broadcast scalars, so 3 load-port
 * cycles cover 4 FMA cycles and the FMA units are the limit.
 *
 * S V: the output is tiny (N x d), so block the reduction instead of the output.
 * The matching V slab (BJ x d floats = 16 KB) stays in L1 and is reused by every
 * output row, which keeps the d accumulators in registers instead of paying a
 * load and a store per FMA.
 *
 * Rows are padded by 64 B. At N=4096 a row is exactly 16 KB, a multiple of the
 * L1 size, so an unpadded block of rows maps onto one L1 set and thrashes.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define HAVE_AVX2 1
#include <immintrin.h>
#else
#define HAVE_AVX2 0
#endif

#define MI 4   /* rows per micro-tile        */
#define MJ 16  /* columns per micro-tile     */
#define BJ 64  /* keys per KT and V block    */
#define PAD 16 /* row padding, in floats     */

static void transpose(const float *src, float *dst, int KN, int N, int d)
{
    for (int k = 0; k < d; k++)
        for (int i = 0; i < N; i++)
            dst[(size_t)k * KN + i] = src[(size_t)i * d + k];
}

#if HAVE_AVX2
/* one row of the micro-tile: broadcast Q, accumulate both halves of KT */
#define ROW(c, p)                                     \
    do {                                              \
        const __m256 b = _mm256_broadcast_ss((p) + k); \
        c##l = _mm256_fmadd_ps(b, kl, c##l);          \
        c##h = _mm256_fmadd_ps(b, kh, c##h);          \
    } while (0)

#define ST2(r, c)                                         \
    do {                                                  \
        float *s_ = S + (size_t)(ib + (r)) * SN + jb;     \
        _mm256_storeu_ps(s_, _mm256_mul_ps(c##l, sc));    \
        _mm256_storeu_ps(s_ + 8, _mm256_mul_ps(c##h, sc)); \
    } while (0)
#endif

/* S[ib .. ib+4)[jb .. jb+16) = Q[ib..)[:] . KT[:][jb..), scaled */
static void qkt_micro(const float *q, const float *KT, int KN, float *S, int SN,
                      int d, int ib, int jb, float scale)
{
#if HAVE_AVX2
    __m256 a0l = _mm256_setzero_ps(), a0h = a0l, a1l = a0l, a1h = a0l;
    __m256 a2l = a0l, a2h = a0l, a3l = a0l, a3h = a0l;
    const __m256 sc = _mm256_set1_ps(scale);
    const float *q0 = q + (size_t)(ib + 0) * d;
    const float *q1 = q0 + d, *q2 = q1 + d, *q3 = q2 + d;

    for (int k = 0; k < d; k++) {
        const float *kt = KT + (size_t)k * KN + jb;
        const __m256 kl = _mm256_loadu_ps(kt);
        const __m256 kh = _mm256_loadu_ps(kt + 8);
        ROW(a0, q0);
        ROW(a1, q1);
        ROW(a2, q2);
        ROW(a3, q3);
    }
    ST2(0, a0);
    ST2(1, a1);
    ST2(2, a2);
    ST2(3, a3);
#else
    for (int ii = 0; ii < MI; ii++)
        for (int t = 0; t < MJ; t++) {
            float a = 0.0f;
            for (int k = 0; k < d; k++)
                a += q[(size_t)(ib + ii) * d + k] * KT[(size_t)k * KN + jb + t];
            S[(size_t)(ib + ii) * SN + jb + t] = a * scale;
        }
#endif
}

static void qkt(const float *q, const float *KT, int KN, float *S, int SN,
                int N, int d, float scale)
{
    const int iv = (N / MI) * MI;
    for (int jb = 0; jb < N; jb += BJ) {
        const int jn = (jb + BJ <= N) ? BJ : N - jb;
        const int jv = (jn / MJ) * MJ;
        for (int ib = 0; ib < iv; ib += MI)
            for (int jj = 0; jj < jv; jj += MJ)
                qkt_micro(q, KT, KN, S, SN, d, ib, jb + jj, scale);
        /* edge strips, only if N is not a multiple of the tile */
        for (int jj = jv; jj < jn; jj++)
            for (int ii = 0; ii < N; ii++) {
                float a = 0.0f;
                for (int k = 0; k < d; k++)
                    a += q[(size_t)ii * d + k] * KT[(size_t)k * KN + jb + jj];
                S[(size_t)ii * SN + jb + jj] = a * scale;
            }
        for (int jj = 0; jj < jv; jj++)
            for (int ii = iv; ii < N; ii++) {
                float a = 0.0f;
                for (int k = 0; k < d; k++)
                    a += q[(size_t)ii * d + k] * KT[(size_t)k * KN + jb + jj];
                S[(size_t)ii * SN + jb + jj] = a * scale;
            }
    }
}

static void softmax_rows(float *S, int SN, int N)
{
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * SN;
        float mx = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > mx)
                mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }
}

#if HAVE_AVX2
/* oi[0..64) += sum_j si[j] * V[j][0..64) */
static void sv_row(const float *si, const float *vp, int jn, float *oi, int d)
{
    __m256 c0 = _mm256_loadu_ps(oi + 0), c1 = _mm256_loadu_ps(oi + 8);
    __m256 c2 = _mm256_loadu_ps(oi + 16), c3 = _mm256_loadu_ps(oi + 24);
    __m256 c4 = _mm256_loadu_ps(oi + 32), c5 = _mm256_loadu_ps(oi + 40);
    __m256 c6 = _mm256_loadu_ps(oi + 48), c7 = _mm256_loadu_ps(oi + 56);
    for (int j = 0; j < jn; j++) {
        const __m256 sb = _mm256_broadcast_ss(si + j);
        c0 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 0), c0);
        c1 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 8), c1);
        c2 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 16), c2);
        c3 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 24), c3);
        c4 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 32), c4);
        c5 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 40), c5);
        c6 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 48), c6);
        c7 = _mm256_fmadd_ps(sb, _mm256_loadu_ps(vp + 56), c7);
        vp += d;
    }
    _mm256_storeu_ps(oi + 0, c0);
    _mm256_storeu_ps(oi + 8, c1);
    _mm256_storeu_ps(oi + 16, c2);
    _mm256_storeu_ps(oi + 24, c3);
    _mm256_storeu_ps(oi + 32, c4);
    _mm256_storeu_ps(oi + 40, c5);
    _mm256_storeu_ps(oi + 48, c6);
    _mm256_storeu_ps(oi + 56, c7);
}
#endif

static void sv(const float *S, int SN, const float *V, float *o, int N, int d)
{
    memset(o, 0, (size_t)N * d * sizeof(float));
    for (int kb = 0; kb < N; kb += BJ) {
        const int jn = (kb + BJ <= N) ? BJ : N - kb;
        for (int i = 0; i < N; i++) {
            const float *si = S + (size_t)i * SN + kb;
            float *oi = o + (size_t)i * d;
            int k0 = 0;
#if HAVE_AVX2
            for (; k0 + 64 <= d; k0 += 64)
                sv_row(si, V + (size_t)kb * d + k0, jn, oi + k0, d);
#endif
            for (; k0 < d; k0++) {
                float a = oi[k0];
                for (int j = 0; j < jn; j++)
                    a += si[j] * V[(size_t)(kb + j) * d + k0];
                oi[k0] = a;
            }
        }
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int KN = N + PAD, SN = N + PAD;
    float *restrict S = malloc((size_t)SN * N * sizeof(float));
    float *restrict KT = malloc((size_t)KN * d * sizeof(float));
    if (!S || !KT) {
        fprintf(stderr, "attn_tiled: out of memory\n");
        exit(1);
    }
    transpose(K, KT, KN, N, d);
    qkt(Q, KT, KN, S, SN, N, d, scale);
    softmax_rows(S, SN, N);
    sv(S, SN, V, O, N, d);
    free(S);
    free(KT);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
