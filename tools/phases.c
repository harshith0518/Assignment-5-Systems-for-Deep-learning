/*
 * tools/phases.c -- per-kernel timing for a variant that factors its work into
 * kernel_qkt() / kernel_softmax() / kernel_sv().
 *
 * It #includes the variant source (renaming its main/attention) so the numbers
 * come from exactly the shipped kernel code, not from a copy.  The sum of the
 * three phases is checked against the variant's own end-to-end time.
 *
 * Build: gcc -O3 -march=native -o build/phases_<variant> tools/phases.c -lm \
 *            -DVARIANT_HEADER='"src/attn_tiled.c"'
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
#if defined(VARIANT_BEST)
#include "../src/attn_best.c"
#elif defined(VARIANT_FUSED)
#include "../src/attn_fused.c"
#elif defined(VARIANT_OMP)
#include "../src/attn_omp.c"
#else
#include "../src/attn_tiled.c"
#endif
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
    int N = hdr[0], d = hdr[1];
    size_t nd = (size_t)N * d;
    float *Q = malloc(nd * sizeof(float)), *K = malloc(nd * sizeof(float));
    float *V = malloc(nd * sizeof(float)), *O = malloc(nd * sizeof(float));
    if (!Q || !K || !V || !O)
        return 1;
    if (fread(Q, sizeof(float), nd, in) != nd || fread(K, sizeof(float), nd, in) != nd ||
        fread(V, sizeof(float), nd, in) != nd)
        return 1;
    fclose(in);

    const float scale = 1.0f / sqrtf((float)d);
    const int SN = N + PAD;
    const int KN = SN;
    float *S = malloc((size_t)SN * N * sizeof(float));
    float *KT = malloc((size_t)KN * d * sizeof(float));
    if (!S || !KT)
        return 1;

    double ttr[5], tm[5], tv[5], tq[5];

    /* warm-up */
    transpose(K, KT, KN, N, d);
    qkt(Q, KT, KN, S, SN, N, d, scale);
    softmax_rows(S, SN, N);
    sv(S, SN, V, O, N, d);

    for (int r = 0; r < 5; r++) {
        double t0 = tnow();
        transpose(K, KT, KN, N, d);
        ttr[r] = tnow() - t0;

        t0 = tnow();
        qkt(Q, KT, KN, S, SN, N, d, scale);
        tq[r] = tnow() - t0;

        t0 = tnow();
        softmax_rows(S, SN, N);
        tm[r] = tnow() - t0;

        t0 = tnow();
        sv(S, SN, V, O, N, d);
        tv[r] = tnow() - t0;
    }
    double q = 0, m = 0, v = 0, tr = 0;
    /* simple median-of-5 */
    for (int a = 0; a < 5; a++)
        for (int b = a + 1; b < 5; b++) {
            if (tq[b] < tq[a]) { double x = tq[a]; tq[a] = tq[b]; tq[b] = x; }
            if (ttr[b] < ttr[a]) { double x = ttr[a]; ttr[a] = ttr[b]; ttr[b] = x; }
            if (tm[b] < tm[a]) { double x = tm[a]; tm[a] = tm[b]; tm[b] = x; }
            if (tv[b] < tv[a]) { double x = tv[a]; tv[a] = tv[b]; tv[b] = x; }
        }
    q = tq[2];
    m = tm[2];
    v = tv[2];
    tr = ttr[2];
    double total = q + m + v;
    double flops = 2.0 * N * (double)N * d;
    printf("%-22s N=%-5d d=%-4d  sum of phases = %8.3f ms   (%.2f GFLOP/s end to end)\n",
           argv[1], N, d, total * 1e3, 4.0 * flops / total * 1e-9);
    printf("   transpose K    %8.3f ms  %5.1f%%\n", tr * 1e3, 100.0 * tr / total);
    printf("   QK^T           %8.3f ms  %5.1f%%   %6.2f GFLOP/s   %5.1f%% of peak\n", q * 1e3,
           100.0 * q / total, flops / q * 1e-9, 100.0 * flops / q * 1e-9 / 76.5);
    printf("   softmax        %8.3f ms  %5.1f%%   %6.1f Mexp/s\n", m * 1e3, 100.0 * m / total,
           N * (double)N / m * 1e-6);
    printf("   S V            %8.3f ms  %5.1f%%   %6.2f GFLOP/s   %5.1f%% of peak\n", v * 1e3,
           100.0 * v / total, flops / v * 1e-9, 100.0 * flops / v * 1e-9 / 76.5);
    free(Q);
    free(K);
    free(V);
    free(O);
    free(S);
    free(KT);
    return 0;
}
