/*
 * tools/exp_test.c -- validates the AVX2 exp used by attn_best / attn_omp.
 *
 *   1. accuracy: max relative error vs libm expf over the range softmax feeds
 *      it (x - max <= 0), plus the extreme -80 end where clamping kicks in.
 *   2. throughput: scalar expf vs the 4-way unrolled vector path inside
 *      softmax_row(), to quantify how much of attn_tiled's softmax time the
 *      vector version removes.
 *
 * Build: gcc -O3 -march=native -o build/exp_test tools/exp_test.c -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define main attn_best_main_unused
#define attention attention_unused
#include "../src/attn_best.c"
#undef main
#undef attention

static double tnow(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void)
{
    /* ---- 1. accuracy over x in [-80, 0] --------------------------------- */
    float xs[8], got[8];
    volatile float sink = 0.0f; /* stops the optimiser deleting the loop */
    double worst = 0.0, worst_at = 0.0;
    for (int i = 0; i < 2000000; i++) {
        double x = -80.0 * (double)i / 2000000.0;
        for (int l = 0; l < 8; l++)
            xs[l] = (float)x;
#if HAVE_AVX2
        _mm256_storeu_ps(got, vexp(_mm256_loadu_ps(xs)));
#else
        for (int l = 0; l < 8; l++)
            got[l] = expf(xs[l]);
#endif
        for (int l = 0; l < 8; l++) {
            float r = expf(xs[l]);
            sink += got[l];
            if (r > 1e-30f) {
                double e = fabs((double)got[l] - (double)r) / (double)r;
                if (e > worst) {
                    worst = e;
                    worst_at = xs[l];
                }
            }
        }
    }
    printf("exp accuracy over x in [-80,0]: max rel error %.3e at x=%.3f (sink %.3e)\n",
           worst, worst_at, (double)sink);

    /* ---- 2. throughput: scalar expf vs softmax_row ---------------------- */
    const int N = 4096, ROWS = 4096; /* 16.8 M exps, same as N=4096 attention */
    float *row = malloc((size_t)N * sizeof(float));
    float *copy = malloc((size_t)N * sizeof(float));
    for (int j = 0; j < N; j++) {
        row[j] = -(float)((j * 2654435761u) % 20000) / 1000.0f; /* in [-20, 0) */
        copy[j] = row[j];
    }

    /* scalar reference softmax */
    double t0 = tnow();
    for (int i = 0; i < ROWS; i++) {
        float mx = copy[0];
        for (int j = 1; j < N; j++)
            if (copy[j] > mx)
                mx = copy[j];
        float s = 0.0f;
        for (int j = 0; j < N; j++)
            s += copy[j] = expf(copy[j] - mx);
        for (int j = 0; j < N; j++)
            copy[j] /= s;
    }
    double t_scalar = tnow() - t0;

    /* the vectorised one, on the same data */
    for (int j = 0; j < N; j++)
        row[j] = -(float)((j * 2654435761u) % 20000) / 1000.0f;
    t0 = tnow();
    for (int i = 0; i < ROWS; i++)
        softmax(row, N);
    double t_vec = tnow() - t0;

    double n = (double)ROWS * N;
    printf("softmax, 16.8 M exps:  scalar expf %8.2f ms (%.0f Mexp/s)   "
           "vector exp %8.2f ms (%.0f Mexp/s)   speedup %.2fx\n",
           t_scalar * 1e3, n / t_scalar * 1e-6, t_vec * 1e3, n / t_vec * 1e-6,
           t_scalar / t_vec);
    free(row);
    free(copy);
    return 0;
}
