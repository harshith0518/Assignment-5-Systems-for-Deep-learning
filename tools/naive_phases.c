/*
 * tools/naive_phases.c -- where does the naive kernel's time actually go?
 *
 * Same three steps as src/attn_naive.c, but each step is timed separately
 * (median of 5 runs each) so Q2 can attribute time to QK^T, softmax and SV.
 *
 * Build: gcc -O3 -march=native -o build/naive_phases tools/naive_phases.c -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static double median5(double *t)
{
    qsort(t, 5, sizeof t[0], cmp_double);
    return t[2];
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1)
        return 1;
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    float *S = malloc((size_t)N * N * sizeof(float));
    if (!Q || !K || !V || !O || !S)
        return 1;
    if (fread(Q, sizeof(float), n, in) != n || fread(K, sizeof(float), n, in) != n ||
        fread(V, sizeof(float), n, in) != n)
        return 1;
    fclose(in);

    const float scale = 1.0f / sqrtf((float)d);
    double t_qk[5], t_sm[5], t_sv[5];
    double t_all[5];

    for (int r = 0; r < 5; r++) {
        double t0 = now();

        /* 1. S = Q K^T * scale */
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++)
                    acc += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
                S[(size_t)i * N + j] = acc * scale;
            }
        t_qk[r] = now() - t0;
        t0 = now();

        /* 2. softmax per row */
        for (int i = 0; i < N; i++) {
            float *row = S + (size_t)i * N;
            float max = row[0];
            for (int j = 1; j < N; j++)
                if (row[j] > max)
                    max = row[j];
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                row[j] = expf(row[j] - max);
                sum += row[j];
            }
            for (int j = 0; j < N; j++)
                row[j] = row[j] / sum;
        }
        t_sm[r] = now() - t0;
        t0 = now();

        /* 3. O = S V */
        for (int i = 0; i < N; i++)
            for (int k = 0; k < d; k++) {
                float acc = 0.0f;
                for (int j = 0; j < N; j++)
                    acc += S[(size_t)i * N + j] * V[(size_t)j * d + k];
                O[(size_t)i * d + k] = acc;
            }
        t_sv[r] = now() - t0;
    }
    for (int r = 0; r < 5; r++) {
        t_all[r] = t_qk[r] + t_sm[r] + t_sv[r];
    }

    double tq = median5(t_qk), tm = median5(t_sm), tv = median5(t_sv);
    double total = tq + tm + tv;
    double flops = 4.0 * N * (double)N * d;
    printf("%-28s N=%-5d d=%-4d total=%9.3f ms  %6.2f GFLOP/s\n", argv[1], N, d, total * 1e3,
           flops / total * 1e-9);
    printf("    QK^T   %9.3f ms  %5.1f%%   %6.2f GFLOP/s   (2N^2 d = %.2f GFLOP)\n", tq * 1e3,
           100.0 * tq / total, 2.0 * N * (double)N * d / tq * 1e-9, 2.0 * N * (double)N * d * 1e-9);
    printf("    softmax%9.3f ms  %5.1f%%   %6.1f Mexp/s    (N^2 = %.1f Mexp)\n", tm * 1e3,
           100.0 * tm / total, N * (double)N / tm * 1e-6, N * (double)N * 1e-6);
    printf("    S V    %9.3f ms  %5.1f%%   %6.2f GFLOP/s   (2N^2 d = %.2f GFLOP)\n", tv * 1e3,
           100.0 * tv / total, 2.0 * N * (double)N * d / tv * 1e-9, 2.0 * N * (double)N * d * 1e-9);
    printf("    S matrix footprint %.1f MB   (L3 = 16 MB)\n", (double)N * N * 4 / 1048576.0);
    free(Q);
    free(K);
    free(V);
    free(O);
    free(S);
    return 0;
}
