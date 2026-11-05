/* tools/phases.c -- split the time inside a kernel.
 *
 *   phases naive <input.bin>   the three steps of the naive kernel
 *   phases tiled <input.bin>   the three kernels of attn_tiled
 *
 * The tiled path #includes the shipped source, so those numbers come from the
 * code that was submitted, not from a copy.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define main variant_main_unused
#define attention attention_unused
#include "../src/attn_tiled.c"
#undef main
#undef attention

static double tnow(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int load(const char *path, int *n, int *d, float **Q, float **K, float **V)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 0;
    }
    int32_t h[2];
    if (fread(h, sizeof h, 1, f) != 1)
        return 0;
    *n = h[0];
    *d = h[1];
    size_t nd = (size_t)(*n) * (*d);
    *Q = malloc(nd * sizeof(float));
    *K = malloc(nd * sizeof(float));
    *V = malloc(nd * sizeof(float));
    if (!*Q || !*K || !*V)
        return 0;
    if (fread(*Q, sizeof(float), nd, f) != nd || fread(*K, sizeof(float), nd, f) != nd ||
        fread(*V, sizeof(float), nd, f) != nd)
        return 0;
    fclose(f);
    return 1;
}

static double median(double *t, int k)
{
    for (int a = 0; a < k; a++)
        for (int b = a + 1; b < k; b++)
            if (t[b] < t[a]) {
                double x = t[a];
                t[a] = t[b];
                t[b] = x;
            }
    return t[k / 2];
}

static void naive_steps(int n, int d, const float *Q, const float *K, const float *V)
{
    float *S = malloc((size_t)n * n * sizeof(float));
    if (!S)
        return;
    const float sc = 1.0f / sqrtf((float)d);
    double tq[5], tm[5], tv[5];

    for (int r = 0; r < 5; r++) {
        double t0 = tnow();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                float a = 0;
                for (int k = 0; k < d; k++)
                    a += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
                S[(size_t)i * n + j] = a * sc;
            }
        tq[r] = tnow() - t0;

        t0 = tnow();
        for (int i = 0; i < n; i++) {
            float *row = S + (size_t)i * n;
            float m = row[0];
            for (int j = 1; j < n; j++)
                if (row[j] > m)
                    m = row[j];
            float sum = 0;
            for (int j = 0; j < n; j++)
                sum += row[j] = expf(row[j] - m);
            for (int j = 0; j < n; j++)
                row[j] /= sum;
        }
        tm[r] = tnow() - t0;

        t0 = tnow();
        float *O = malloc((size_t)n * d * sizeof(float));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < d; k++) {
                float a = 0;
                for (int j = 0; j < n; j++)
                    a += S[(size_t)i * n + j] * V[(size_t)j * d + k];
                O[(size_t)i * d + k] = a;
            }
        tv[r] = tnow() - t0;
        free(O);
    }
    double q = median(tq, 5), m = median(tm, 5), v = median(tv, 5), tot = q + m + v;
    double fl = 2.0 * n * (double)n * d;
    printf("naive  N=%-5d d=%-4d  three steps %8.3f ms  (%.2f GFLOP/s)\n", n, d, tot * 1e3,
           4.0 * fl / tot * 1e-9);
    printf("   QK^T     %8.3f ms  %5.1f%%  %6.2f GFLOP/s\n", q * 1e3, 100 * q / tot, fl / q * 1e-9);
    printf("   softmax  %8.3f ms  %5.1f%%  %6.1f Mexp/s\n", m * 1e3, 100 * m / tot,
           n * (double)n / m * 1e-6);
    printf("   S V      %8.3f ms  %5.1f%%  %6.2f GFLOP/s\n", v * 1e3, 100 * v / tot, fl / v * 1e-9);
    free(S);
}

static void tiled_kernels(int n, int d, const float *Q, const float *K, const float *V)
{
    const float sc = 1.0f / sqrtf((float)d);
    const int kn = n + PAD, sn = n + PAD;
    float *s = malloc((size_t)sn * n * sizeof(float));
    float *kt = malloc((size_t)kn * d * sizeof(float));
    float *O = malloc((size_t)n * d * sizeof(float));
    if (!s || !kt || !O)
        return;
    double tq[5], tm[5], tv[5];

    qkt(Q, kt, kn, s, sn, n, d, sc);
    smax(s, sn, n);
    sv(s, sn, V, O, n, d);

    for (int r = 0; r < 5; r++) {
        double t0 = tnow();
        qkt(Q, kt, kn, s, sn, n, d, sc);
        tq[r] = tnow() - t0;
        t0 = tnow();
        smax(s, sn, n);
        tm[r] = tnow() - t0;
        t0 = tnow();
        sv(s, sn, V, O, n, d);
        tv[r] = tnow() - t0;
    }
    double q = median(tq, 5), m = median(tm, 5), v = median(tv, 5), tot = q + m + v;
    double fl = 2.0 * n * (double)n * d;
    printf("tiled  N=%-5d d=%-4d  kernels %8.3f ms  (%.2f GFLOP/s)\n", n, d, tot * 1e3,
           4.0 * fl / tot * 1e-9);
    printf("   QK^T     %8.3f ms  %5.1f%%  %6.2f GFLOP/s\n", q * 1e3, 100 * q / tot, fl / q * 1e-9);
    printf("   softmax  %8.3f ms  %5.1f%%  %6.1f Mexp/s\n", m * 1e3, 100 * m / tot,
           n * (double)n / m * 1e-6);
    printf("   S V      %8.3f ms  %5.1f%%  %6.2f GFLOP/s\n", v * 1e3, 100 * v / tot, fl / v * 1e-9);
    free(s);
    free(kt);
    free(O);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s naive|tiled <input.bin>\n", argv[0]);
        return 1;
    }
    int n = 0, d = 0;
    float *Q = NULL, *K = NULL, *V = NULL;
    if (!load(argv[2], &n, &d, &Q, &K, &V))
        return 1;
    if (strcmp(argv[1], "naive") == 0)
        naive_steps(n, d, Q, K, V);
    else
        tiled_kernels(n, d, Q, K, V);
    free(Q);
    free(K);
    free(V);
    return 0;
}
