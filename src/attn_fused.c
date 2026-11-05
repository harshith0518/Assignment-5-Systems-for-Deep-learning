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
#include <immintrin.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#error build with -march=native, AVX2 and FMA are required
#endif

#define PAD 16

static void score(const float *q, const float *kt, int kn, float *s, int jb, int jn,
                  int d, float sc)
{
    if (jn == 64) {
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m256 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        for (int k = 0; k < d; k++) {
            const __m256 b = _mm256_broadcast_ss(q + k);
            const float *t = kt + (size_t)k * kn + jb;
            c0 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 0), c0);
            c1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 8), c1);
            c2 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 16), c2);
            c3 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 24), c3);
            c4 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 32), c4);
            c5 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 40), c5);
            c6 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 48), c6);
            c7 = _mm256_fmadd_ps(b, _mm256_loadu_ps(t + 56), c7);
        }
        const __m256 vsc = _mm256_set1_ps(sc);
        _mm256_storeu_ps(s + jb + 0, _mm256_mul_ps(c0, vsc));
        _mm256_storeu_ps(s + jb + 8, _mm256_mul_ps(c1, vsc));
        _mm256_storeu_ps(s + jb + 16, _mm256_mul_ps(c2, vsc));
        _mm256_storeu_ps(s + jb + 24, _mm256_mul_ps(c3, vsc));
        _mm256_storeu_ps(s + jb + 32, _mm256_mul_ps(c4, vsc));
        _mm256_storeu_ps(s + jb + 40, _mm256_mul_ps(c5, vsc));
        _mm256_storeu_ps(s + jb + 48, _mm256_mul_ps(c6, vsc));
        _mm256_storeu_ps(s + jb + 56, _mm256_mul_ps(c7, vsc));
        return;
    }
    for (int j = 0; j < jn; j++) {
        float a = 0;
        for (int k = 0; k < d; k++)
            a += q[k] * kt[(size_t)k * kn + jb + j];
        s[jb + j] = a * sc;
    }
}

static void pv(const float *p, const float *V, float *o, int n, int d, float iv)
{
    int k0 = 0;
    for (; k0 + 64 <= d; k0 += 64) {
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m256 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        const float *v = V + k0;
        for (int j = 0; j < n; j++) {
            float x = p[j] * iv;
            const __m256 b = _mm256_broadcast_ss(&x);
            c0 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 0), c0);
            c1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 8), c1);
            c2 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 16), c2);
            c3 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 24), c3);
            c4 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 32), c4);
            c5 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 40), c5);
            c6 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 48), c6);
            c7 = _mm256_fmadd_ps(b, _mm256_loadu_ps(v + 56), c7);
            v += d;
        }
        _mm256_storeu_ps(o + k0 + 0, c0);
        _mm256_storeu_ps(o + k0 + 8, c1);
        _mm256_storeu_ps(o + k0 + 16, c2);
        _mm256_storeu_ps(o + k0 + 24, c3);
        _mm256_storeu_ps(o + k0 + 32, c4);
        _mm256_storeu_ps(o + k0 + 40, c5);
        _mm256_storeu_ps(o + k0 + 48, c6);
        _mm256_storeu_ps(o + k0 + 56, c7);
    }
    for (; k0 < d; k0++) {
        float a = 0;
        for (int j = 0; j < n; j++)
            a += p[j] * iv * V[(size_t)j * d + k0];
        o[k0] = a;
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int n, int d)
{
    float sc = 1.0f / sqrtf(d);
    int kn = n + PAD;
    float *kt = malloc((size_t)kn * d * sizeof(float));
    float *s = malloc((size_t)n * sizeof(float));
    if (!kt || !s) {
        printf("alloc failed\n");
        exit(1);
    }
    for (int k = 0; k < d; k++)
        for (int i = 0; i < n; i++)
            kt[(size_t)k * kn + i] = K[(size_t)i * d + k];

    for (int i = 0; i < n; i++) {
        const float *q = Q + (size_t)i * d;
        for (int jb = 0; jb < n; jb += 64)
            score(q, kt, kn, s, jb, (jb + 64 <= n) ? 64 : n - jb, d, sc);

        float m = s[0];
        for (int j = 1; j < n; j++)
            if (s[j] > m)
                m = s[j];
        float sum = 0;
        for (int j = 0; j < n; j++) {
            s[j] = expf(s[j] - m);
            sum += s[j];
        }
        pv(s, V, O + (size_t)i * d, n, d, 1.0f / sum);
    }

    free(kt);
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
