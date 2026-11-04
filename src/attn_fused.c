/*
 * attn_fused.c -- Q3c: fuse the three steps, so the N x N matrix is never stored.
 *
 * Row i of O only needs row i of S, so keep one score row in a 16 KB L1 buffer
 * and run the three steps over it back to back:
 *
 *     scores -> row max -> exp/sum -> accumulate p*V into 8 YMM registers
 *
 * DRAM traffic drops to the compulsory Q, K, V, O and the 67 MB score matrix
 * disappears. The catch is that row i now reads all of V (1 MB), so at N=4096
 * that is 4 GB of L3 traffic against the ~0.2 GB of score traffic saved. Fusion
 * removes the right bottleneck and adds a bigger one; Q3d blocks queries to fix
 * it. expf is still scalar libm here, and it dominates.
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

#define PAD 16

/* s[jb .. jb+64) = scale * (qi . KT[:, jb..)) : 8 accumulator chains */
static void score_block(const float *qi, const float *KT, int KN, float *s,
                        int jb, int jn, int d, float scale)
{
#if HAVE_AVX2
    if (jn == 64) {
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m256 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        for (int k = 0; k < d; k++) {
            const __m256 qk = _mm256_broadcast_ss(qi + k);
            const float *kt = KT + (size_t)k * KN + jb;
            c0 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 0), c0);
            c1 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 8), c1);
            c2 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 16), c2);
            c3 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 24), c3);
            c4 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 32), c4);
            c5 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 40), c5);
            c6 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 48), c6);
            c7 = _mm256_fmadd_ps(qk, _mm256_loadu_ps(kt + 56), c7);
        }
        const __m256 sc = _mm256_set1_ps(scale);
        _mm256_storeu_ps(s + jb + 0, _mm256_mul_ps(c0, sc));
        _mm256_storeu_ps(s + jb + 8, _mm256_mul_ps(c1, sc));
        _mm256_storeu_ps(s + jb + 16, _mm256_mul_ps(c2, sc));
        _mm256_storeu_ps(s + jb + 24, _mm256_mul_ps(c3, sc));
        _mm256_storeu_ps(s + jb + 32, _mm256_mul_ps(c4, sc));
        _mm256_storeu_ps(s + jb + 40, _mm256_mul_ps(c5, sc));
        _mm256_storeu_ps(s + jb + 48, _mm256_mul_ps(c6, sc));
        _mm256_storeu_ps(s + jb + 56, _mm256_mul_ps(c7, sc));
        return;
    }
#endif
    for (int j = 0; j < jn; j++) {
        float a = 0.0f;
        for (int k = 0; k < d; k++)
            a += qi[k] * KT[(size_t)k * KN + jb + j];
        s[jb + j] = a * scale;
    }
}

/* oi = sum_j p[j]*inv * V[j][:] */
static void accumulate_pv(const float *p, const float *V, float *oi,
                          int N, int d, float inv)
{
    int k0 = 0;
#if HAVE_AVX2
    for (; k0 + 64 <= d; k0 += 64) {
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m256 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        const float *vp = V + k0;
        for (int j = 0; j < N; j++) {
            const float pj = p[j] * inv;
            const __m256 pb = _mm256_broadcast_ss(&pj);
            c0 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 0), c0);
            c1 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 8), c1);
            c2 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 16), c2);
            c3 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 24), c3);
            c4 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 32), c4);
            c5 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 40), c5);
            c6 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 48), c6);
            c7 = _mm256_fmadd_ps(pb, _mm256_loadu_ps(vp + 56), c7);
            vp += d;
        }
        _mm256_storeu_ps(oi + k0 + 0, c0);
        _mm256_storeu_ps(oi + k0 + 8, c1);
        _mm256_storeu_ps(oi + k0 + 16, c2);
        _mm256_storeu_ps(oi + k0 + 24, c3);
        _mm256_storeu_ps(oi + k0 + 32, c4);
        _mm256_storeu_ps(oi + k0 + 40, c5);
        _mm256_storeu_ps(oi + k0 + 48, c6);
        _mm256_storeu_ps(oi + k0 + 56, c7);
    }
#endif
    for (; k0 < d; k0++) {
        float a = 0.0f;
        for (int j = 0; j < N; j++)
            a += p[j] * inv * V[(size_t)j * d + k0];
        oi[k0] = a;
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int KN = N + PAD;
    float *restrict KT = malloc((size_t)KN * d * sizeof(float));
    float *restrict s = malloc((size_t)N * sizeof(float));
    if (!KT || !s) {
        fprintf(stderr, "attn_fused: out of memory\n");
        exit(1);
    }
    /* transpose K once, padded so rows do not alias in L1 */
    for (int k = 0; k < d; k++)
        for (int i = 0; i < N; i++)
            KT[(size_t)k * KN + i] = K[(size_t)i * d + k];

    for (int i = 0; i < N; i++) {
        const float *qi = Q + (size_t)i * d;

        for (int jb = 0; jb < N; jb += 64)
            score_block(qi, KT, KN, s, jb, (jb + 64 <= N) ? 64 : N - jb, d, scale);

        float mx = s[0];
        for (int j = 1; j < N; j++)
            if (s[j] > mx)
                mx = s[j];

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            s[j] = expf(s[j] - mx);
            sum += s[j];
        }

        accumulate_pv(s, V, O + (size_t)i * d, N, d, 1.0f / sum);
    }

    free(KT);
    free(s);
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
