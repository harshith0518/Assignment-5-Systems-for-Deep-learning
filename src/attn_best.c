/*
 * attn_best.c -- Q3d: fastest single-threaded version.
 *
 * Two measured limits from Q3b drive it.
 *
 * 1. Block over queries (BI=64). The 64-row score tile stays in cache, so there
 *    is no 67 MB allocation and no page-fault storm, and V is streamed once per
 *    block instead of once per row: 64 MB of V traffic instead of 4 GB.
 *
 * 2. Replace expf with an AVX2 exp. 16.8 M expf calls were 45 % of attn_tiled's
 *    kernel time. y = x*log2(e), split y = n + f with |f| <= 0.5, evaluate 2^f
 *    with a degree-5 minimax polynomial, build 2^n by writing n+127 into the
 *    float exponent field. Measured 5.1e-6 max relative error (tools/exp_test.c)
 *    and 9.8x faster than libm.
 *
 * Both GEMMs keep the tiled structure: 4x16 register tile for QK^T, key-blocked
 * outer product for S V so the d accumulators stay in YMM registers.
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

#ifndef MI
#define MI 4 /* rows per micro-tile         */
#endif
#ifndef MJ
#define MJ 16 /* columns per micro-tile     */
#endif
#ifndef BJ
#define BJ 64 /* keys per KT and V block    */
#endif
#ifndef BI
#define BI 64 /* queries per block          */
#endif
#define PAD 16 /* row padding, in floats */

#if HAVE_AVX2
/* exp(x) for x <= 0, which is all softmax needs once the row max is removed */
static inline __m256 exp_ps(__m256 x)
{
    x = _mm256_max_ps(x, _mm256_set1_ps(-80.0f));
    const __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    const __m256i ni = _mm256_cvtps_epi32(y);
    const __m256 f = _mm256_sub_ps(y, _mm256_cvtepi32_ps(ni));

    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(1.33335581e-3f), f,
                               _mm256_set1_ps(9.61812911e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.55036329e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.40226507e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.93147182e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline float hmax_ps(__m256 v)
{
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

static inline float hsum_ps(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif

static void softmax_row(float *row, int N)
{
#if HAVE_AVX2
    int j = 0;
    __m256 vmax = _mm256_loadu_ps(row);
    for (j = 8; j + 8 <= N; j += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + j));
    float mx = hmax_ps(vmax);
    for (; j < N; j++)
        if (row[j] > mx)
            mx = row[j];

    const __m256 mv = _mm256_set1_ps(mx);
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (j = 0; j + 32 <= N; j += 32) {
        __m256 a = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), mv));
        __m256 b = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j + 8), mv));
        __m256 c = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j + 16), mv));
        __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j + 24), mv));
        _mm256_storeu_ps(row + j, a);
        _mm256_storeu_ps(row + j + 8, b);
        _mm256_storeu_ps(row + j + 16, c);
        _mm256_storeu_ps(row + j + 24, e);
        s0 = _mm256_add_ps(s0, a);
        s1 = _mm256_add_ps(s1, b);
        s2 = _mm256_add_ps(s2, c);
        s3 = _mm256_add_ps(s3, e);
    }
    float sum = hsum_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; j < N; j++) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }

    const __m256 iv = _mm256_set1_ps(1.0f / sum);
    for (j = 0; j + 8 <= N; j += 8)
        _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), iv));
    for (; j < N; j++)
        row[j] *= 1.0f / sum;
#else
    float mx = row[0];
    for (int j = 1; j < N; j++)
        if (row[j] > mx)
            mx = row[j];
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    for (int j = 0; j < N; j++)
        row[j] /= sum;
#endif
}

#if HAVE_AVX2
#define ROW(c, p)                                      \
    do {                                               \
        const __m256 b = _mm256_broadcast_ss((p) + k);  \
        c##l = _mm256_fmadd_ps(b, kl, c##l);           \
        c##h = _mm256_fmadd_ps(b, kh, c##h);           \
    } while (0)

#define ST2(r, c)                                          \
    do {                                                   \
        float *s_ = S + (size_t)(ib + (r)) * SN + jb;      \
        _mm256_storeu_ps(s_, _mm256_mul_ps(c##l, sc));     \
        _mm256_storeu_ps(s_ + 8, _mm256_mul_ps(c##h, sc)); \
    } while (0)
#endif

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

#if HAVE_AVX2
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

/* QK^T -> softmax -> S V over one block of `rows` queries; S is rows x N */
static void attn_block(const float *q, const float *KT, int KN, const float *V,
                       float *o, float *S, int SN, int rows, int N, int d, float scale)
{
    const int iv = (rows / MI) * MI;
    for (int jb = 0; jb < N; jb += BJ) {
        const int jn = (jb + BJ <= N) ? BJ : N - jb;
        const int jv = (jn / MJ) * MJ;
        for (int ib = 0; ib < iv; ib += MI)
            for (int jj = 0; jj < jv; jj += MJ)
                qkt_micro(q, KT, KN, S, SN, d, ib, jb + jj, scale);
        for (int jj = jv; jj < jn; jj++) /* edge strips */
            for (int ii = 0; ii < rows; ii++) {
                float a = 0.0f;
                for (int k = 0; k < d; k++)
                    a += q[(size_t)ii * d + k] * KT[(size_t)k * KN + jb + jj];
                S[(size_t)ii * SN + jb + jj] = a * scale;
            }
        for (int jj = 0; jj < jv; jj++)
            for (int ii = iv; ii < rows; ii++) {
                float a = 0.0f;
                for (int k = 0; k < d; k++)
                    a += q[(size_t)ii * d + k] * KT[(size_t)k * KN + jb + jj];
                S[(size_t)ii * SN + jb + jj] = a * scale;
            }
    }

    for (int i = 0; i < rows; i++)
        softmax_row(S + (size_t)i * SN, N);

    memset(o, 0, (size_t)rows * d * sizeof(float));
    for (int kb = 0; kb < N; kb += BJ) {
        const int jn = (kb + BJ <= N) ? BJ : N - kb;
        for (int i = 0; i < rows; i++) {
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
    float *restrict KT = malloc((size_t)KN * d * sizeof(float));
    float *restrict S = malloc((size_t)BI * SN * sizeof(float));
    if (!KT || !S) {
        fprintf(stderr, "attn_best: out of memory\n");
        exit(1);
    }
    for (int k = 0; k < d; k++)
        for (int i = 0; i < N; i++)
            KT[(size_t)k * KN + i] = K[(size_t)i * d + k];

    for (int ib = 0; ib < N; ib += BI) {
        const int rows = (ib + BI <= N) ? BI : N - ib;
        attn_block(Q + (size_t)ib * d, KT, KN, V, O + (size_t)ib * d, S, SN, rows, N,
                   d, scale);
    }

    free(KT);
    free(S);
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
