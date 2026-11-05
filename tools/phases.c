/* tools/phases.c -- per-kernel timing of attn_tiled.
 * #includes the shipped source so the numbers are the shipped code. */
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
    int n = hdr[0], d = hdr[1];
    size_t nd = (size_t)n * d;
    float *Q = malloc(nd * sizeof(float)), *K = malloc(nd * sizeof(float));
    float *V = malloc(nd * sizeof(float)), *O = malloc(nd * sizeof(float));
    if (!Q || !K || !V || !O)
        return 1;
    if (fread(Q, sizeof(float), nd, in) != nd || fread(K, sizeof(float), nd, in) != nd ||
        fread(V, sizeof(float), nd, in) != nd)
        return 1;
    fclose(in);

    const float sc = 1.0f / sqrtf(d);
    const int kn = n + PAD, sn = n + PAD;
    float *s = malloc((size_t)sn * n * sizeof(float));
    float *kt = malloc((size_t)kn * d * sizeof(float));
    if (!s || !kt)
        return 1;

    double tq[5], tm[5], tv[5];

    /* warm-up */
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
    for (int a = 0; a < 5; a++)
        for (int b = a + 1; b < 5; b++) {
            if (tq[b] < tq[a]) { double x = tq[a]; tq[a] = tq[b]; tq[b] = x; }
            if (tm[b] < tm[a]) { double x = tm[a]; tm[a] = tm[b]; tm[b] = x; }
            if (tv[b] < tv[a]) { double x = tv[a]; tv[a] = tv[b]; tv[b] = x; }
        }
    double q = tq[2], m = tm[2], v = tv[2];
    double tot = q + m + v;
    double fl = 2.0 * n * (double)n * d;
    printf("%-22s N=%-5d d=%-4d  kernels %8.3f ms  (%.2f GFLOP/s end to end)\n", argv[1], n,
           d, tot * 1e3, 4.0 * fl / tot * 1e-9);
    printf("   QK^T     %8.3f ms  %5.1f%%  %6.2f GFLOP/s\n", q * 1e3, 100 * q / tot,
           fl / q * 1e-9);
    printf("   softmax  %8.3f ms  %5.1f%%  %6.1f Mexp/s\n", m * 1e3, 100 * m / tot,
           n * (double)n / m * 1e-6);
    printf("   S V      %8.3f ms  %5.1f%%  %6.2f GFLOP/s\n", v * 1e3, 100 * v / tot,
           fl / v * 1e-9);
    free(Q);
    free(K);
    free(V);
    free(O);
    free(s);
    free(kt);
    return 0;
}
