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

#define MI 4
#define MJ 16
#define BJ 64
#define PAD 16

#define ROW(c, p)                                     \
    do {                                              \
        const __m256 b = _mm256_broadcast_ss((p) + k); \
        c##l = _mm256_fmadd_ps(b, kl, c##l);          \
        c##h = _mm256_fmadd_ps(b, kh, c##h);          \
    } while (0)

#define PUT(r, c)                                          \
    do {                                                   \
        float *t = s + (size_t)(i0 + (r)) * sn + j0;       \
        _mm256_storeu_ps(t, _mm256_mul_ps(c##l, vsc));     \
        _mm256_storeu_ps(t + 8, _mm256_mul_ps(c##h, vsc)); \
    } while (0)

static void micro(const float *q, const float *kt, int kn, float *s, int sn,
                  int d, int i0, int j0, float sc)
{
    __m256 a0l = _mm256_setzero_ps(), a0h = a0l, a1l = a0l, a1h = a0l;
    __m256 a2l = a0l, a2h = a0l, a3l = a0l, a3h = a0l;
    const __m256 vsc = _mm256_set1_ps(sc);
    const float *q0 = q + (size_t)(i0 + 0) * d;
    const float *q1 = q0 + d, *q2 = q1 + d, *q3 = q2 + d;

    for (int k = 0; k < d; k++) {
        const float *t = kt + (size_t)k * kn + j0;
        const __m256 kl = _mm256_loadu_ps(t);
        const __m256 kh = _mm256_loadu_ps(t + 8);
        ROW(a0, q0);
        ROW(a1, q1);
        ROW(a2, q2);
        ROW(a3, q3);
    }
    PUT(0, a0);
    PUT(1, a1);
    PUT(2, a2);
    PUT(3, a3);
}

static void qkt(const float *q, const float *kt, int kn, float *s, int sn, int n, int d,
                float sc)
{
    int iv = (n / MI) * MI;
    for (int jb = 0; jb < n; jb += BJ) {
        int jn = (jb + BJ <= n) ? BJ : n - jb;
        int jv = (jn / MJ) * MJ;
        for (int ib = 0; ib < iv; ib += MI)
            for (int jj = 0; jj < jv; jj += MJ)
                micro(q, kt, kn, s, sn, d, ib, jb + jj, sc);
        for (int jj = jv; jj < jn; jj++)
            for (int i = 0; i < n; i++) {
                float a = 0;
                for (int k = 0; k < d; k++)
                    a += q[(size_t)i * d + k] * kt[(size_t)k * kn + jb + jj];
                s[(size_t)i * sn + jb + jj] = a * sc;
            }
        for (int jj = 0; jj < jv; jj++)
            for (int i = iv; i < n; i++) {
                float a = 0;
                for (int k = 0; k < d; k++)
                    a += q[(size_t)i * d + k] * kt[(size_t)k * kn + jb + jj];
                s[(size_t)i * sn + jb + jj] = a * sc;
            }
    }
}

static void smax(float *s, int sn, int n)
{
    for (int i = 0; i < n; i++) {
        float *r = s + (size_t)i * sn;
        float m = r[0];
        for (int j = 1; j < n; j++)
            if (r[j] > m)
                m = r[j];
        float sum = 0;
        for (int j = 0; j < n; j++) {
            r[j] = expf(r[j] - m);
            sum += r[j];
        }
        float iv = 1.0f / sum;
        for (int j = 0; j < n; j++)
            r[j] *= iv;
    }
}

static void svrow(const float *a, const float *v, int jn, float *o, int d)
{
    __m256 c0 = _mm256_loadu_ps(o + 0), c1 = _mm256_loadu_ps(o + 8);
    __m256 c2 = _mm256_loadu_ps(o + 16), c3 = _mm256_loadu_ps(o + 24);
    __m256 c4 = _mm256_loadu_ps(o + 32), c5 = _mm256_loadu_ps(o + 40);
    __m256 c6 = _mm256_loadu_ps(o + 48), c7 = _mm256_loadu_ps(o + 56);
    for (int j = 0; j < jn; j++) {
        const __m256 b = _mm256_broadcast_ss(a + j);
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
    _mm256_storeu_ps(o + 0, c0);
    _mm256_storeu_ps(o + 8, c1);
    _mm256_storeu_ps(o + 16, c2);
    _mm256_storeu_ps(o + 24, c3);
    _mm256_storeu_ps(o + 32, c4);
    _mm256_storeu_ps(o + 40, c5);
    _mm256_storeu_ps(o + 48, c6);
    _mm256_storeu_ps(o + 56, c7);
}

static void sv(const float *s, int sn, const float *V, float *o, int n, int d)
{
    memset(o, 0, (size_t)n * d * sizeof(float));
    for (int kb = 0; kb < n; kb += BJ) {
        int jn = (kb + BJ <= n) ? BJ : n - kb;
        for (int i = 0; i < n; i++) {
            const float *a = s + (size_t)i * sn + kb;
            float *oi = o + (size_t)i * d;
            int k0 = 0;
            for (; k0 + 64 <= d; k0 += 64)
                svrow(a, V + (size_t)kb * d + k0, jn, oi + k0, d);
            for (; k0 < d; k0++) {
                float t = oi[k0];
                for (int j = 0; j < jn; j++)
                    t += a[j] * V[(size_t)(kb + j) * d + k0];
                oi[k0] = t;
            }
        }
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int n, int d)
{
    float sc = 1.0f / sqrtf(d);
    int kn = n + PAD, sn = n + PAD;
    float *s = malloc((size_t)sn * n * sizeof(float));
    float *kt = malloc((size_t)kn * d * sizeof(float));
    if (!s || !kt) {
        printf("alloc failed\n");
        exit(1);
    }
    for (int k = 0; k < d; k++)
        for (int i = 0; i < n; i++)
            kt[(size_t)k * kn + i] = K[(size_t)i * d + k];
    qkt(Q, kt, kn, s, sn, n, d, sc);
    smax(s, sn, n);
    sv(s, sn, V, O, n, d);
    free(s);
    free(kt);
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
