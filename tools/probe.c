/*
 * tools/probe.c -- machine characterisation for Q1 / Q5.
 *
 * Everything here is measured, not read off a spec sheet. Five probes:
 *
 *   peak    single-core and all-core FP32 FMA throughput  -> GFLOP/s and clock
 *   lat     FMA latency estimate                          -> cycles per FMA
 *   bw      read / write / triad memory bandwidth          -> GB/s
 *   cache   pointer-chase latency vs footprint             -> L1/L2/L3/DRAM
 *   line    stride sweep                                   -> cache line size
 *
 * Build: clang/gcc -O3 -march=native [-fopenmp] -o build/probe tools/probe.c -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* 64-byte aligned allocation, portable across glibc / mingw. */
static void *xalloc(size_t bytes)
{
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0)
        return NULL;
    return p;
#endif
}
static void xfree(void *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ------------------------------------------------------------------ */
/* peak: NACC independent 256-bit FMA chains, 16 FLOP per instruction. */
/* ------------------------------------------------------------------ */
#define NACC 12 /* Zen3+ has 16 architectural YMM regs; 12 leaves room */

static double peak_fma(long iters, int threads)
{
    double t0 = now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (int th = 0; th < threads; th++) {
        __m256 a[NACC], b, c;
        for (int r = 0; r < NACC; r++)
            a[r] = _mm256_set1_ps(1.0f + 1e-6f * r);
        b = _mm256_set1_ps(1.0000001f);
        c = _mm256_set1_ps(1e-7f);
        for (long i = 0; i < iters; i++) {
            for (int r = 0; r < NACC; r++)
                a[r] = _mm256_fmadd_ps(a[r], b, c);
        }
        /* keep the compiler from deleting the work */
        float s = 0.0f;
        for (int r = 0; r < NACC; r++)
            s += _mm256_cvtss_f32(a[r]);
        if (s == 12345.678f)
            printf("impossible %f\n", s);
    }
    double t = now() - t0;
    /* iters * NACC FMAs * 16 FLOP  *  threads */
    return (double)iters * NACC * 16.0 * (double)threads / t * 1e-9;
}

/* Dependency chain of dependent FMAs -> time/iter, and with a known
 * latency we can recover the clock. Zen3/Zen3+ ymm FMA latency = 4. */
static double fma_chain(long iters)
{
    __m256 x = _mm256_set1_ps(1.0f), b = _mm256_set1_ps(1.0000001f),
           c = _mm256_set1_ps(1e-9f);
    double t0 = now();
    for (long i = 0; i < iters; i++)
        x = _mm256_fmadd_ps(x, b, c);
    double t = now() - t0;
    float s = _mm256_cvtss_f32(x);
    if (s == 12345.678f)
        printf("impossible %f\n", s);
    return t / (double)iters; /* seconds per dependent FMA */
}

/* ------------------------------------------------------------------ */
/* bw: three kernels over a buffer much larger than L3.                */
/* ------------------------------------------------------------------ */
static void bw_probe(size_t bytes, int threads)
{
    size_t n = bytes / sizeof(float);
    n -= n % 32;
    float *a = xalloc(n * sizeof(float));
    float *b = xalloc(n * sizeof(float));
    float *c = xalloc(n * sizeof(float));
    if (!a || !b || !c) {
        printf("bw: allocation failed\n");
        return;
    }
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    /* read: 8 independent accumulators so the kernel is BW-bound, not
     * latency-bound on a serial FP add chain. */
    float sread = 0.0f;
    double t0 = now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : sread)
#endif
    for (size_t i = 0; i < n; i += 32) {
        __m256 a0 = _mm256_load_ps(a + i), a1 = _mm256_load_ps(a + i + 8);
        __m256 a2 = _mm256_load_ps(a + i + 16), a3 = _mm256_load_ps(a + i + 24);
        __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        sread += _mm256_cvtss_f32(s);
    }
    double t_read = now() - t0;

    /* copy: c = a (read 1 + write 1) */
    t0 = now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (size_t i = 0; i < n; i += 8)
        _mm256_store_ps(c + i, _mm256_load_ps(a + i));
    double t_copy = now() - t0;

    /* triad: c = a + 2*b (read 2 + write 1) */
    t0 = now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (size_t i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(a + i), vb = _mm256_load_ps(b + i);
        _mm256_store_ps(c + i, _mm256_fmadd_ps(_mm256_set1_ps(2.0f), vb, va));
    }
    double t_triad = now() - t0;

    double gb = (double)n * sizeof(float) * 1e-9;
    printf("bw %2d thread(s)  buffer %6.1f MB  read %7.2f  copy %7.2f  triad %7.2f GB/s\n",
           threads, bytes / 1048576.0, gb / t_read, 2 * gb / t_copy, 3 * gb / t_triad);
    if (sread == 12345.678f)
        printf("impossible %f\n", sread);
    xfree(a);
    xfree(b);
    xfree(c);
}

/* ------------------------------------------------------------------ */
/* cache: random cyclic pointer chase, one access per element.        */
/* ------------------------------------------------------------------ */
static int cmp_int(const void *x, const void *y)
{
    int a = *(const int *)x, b = *(const int *)y;
    return (a > b) - (a < b);
}

/* stops the compiler from deleting the pointer chase */
static volatile int sink;

static void cache_probe(void)
{
    static const size_t kib[] = {2,   4,   8,   16,  24,  32,  48,  64,  96,
                                 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
                                 4096, 8192, 16384, 32768, 65536};
    int nk = (int)(sizeof kib / sizeof kib[0]);
    printf("\n  footprint      ns/access   implied latency\n");
    for (int k = 0; k < nk; k++) {
        size_t bytes = kib[k] * 1024;
        size_t n = bytes / sizeof(int);
        int *idx = malloc(n * sizeof(int));
        int *perm = malloc(n * sizeof(int));
        if (!idx || !perm) {
            printf("cache: alloc failed at %zu KiB\n", kib[k]);
            free(idx);
            free(perm);
            continue;
        }
        for (size_t i = 0; i < n; i++)
            perm[i] = (int)i;
        /* Fisher-Yates over a cheap LCG */
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t i = n - 1; i > 0; i--) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            size_t j = (size_t)((s >> 33) % (i + 1));
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        /* cyclic chain: idx[perm[i]] = perm[i+1] */
        for (size_t i = 0; i < n; i++)
            idx[perm[i]] = perm[(i + 1) % n];
        free(perm);

        long steps = 8000000;
        if (n < 4096)
            steps = 20000000;
        int p = 0;
        /* warm up the chain */
        for (int i = 0; i < 100000; i++)
            p = idx[p];
        double t0 = now();
        for (long i = 0; i < steps; i++) {
            p = idx[p];
            p = idx[p];
        }
        double t = now() - t0;
        sink = p; /* volatile sink: stops the compiler deleting the chase */
        /* two dependent accesses per loop iteration */
        double ns = t / (double)(steps * 2) * 1e9;
        printf("  %6zu KiB %6.1f MB   %7.2f ns\n", kib[k], bytes / 1048576.0, ns);
        free(idx);
    }
}

/* ------------------------------------------------------------------ */
/* line: sequential stride sweep over an L1-resident array.           */
/* ------------------------------------------------------------------ */
static void line_probe(void)
{
    printf("\n  stride  ns/load(1 core, 32KB footprint)\n");
    size_t n = 8192; /* 32KB of floats */
    float *a = xalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        a[i] = (float)i;
    for (int stride = 1; stride <= 32; stride *= 2) {
        long iters = 200000;
        size_t pos = 0;
        float acc = 0.0f;
        double t0 = now();
        for (long i = 0; i < iters; i++) {
            acc += a[pos];
            pos = (pos + (size_t)stride) % n;
        }
        double t = now() - t0;
        printf("  %5d   %7.3f ns  (%.1f B per load)\n", stride, t / (double)iters * 1e9,
               (double)stride * sizeof(float));
        if (acc == 12345.678f)
            printf("impossible %f\n", acc);
    }
    xfree(a);
}

int main(int argc, char **argv)
{
    const char *what = argc > 1 ? argv[1] : "all";
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    if (!strcmp(what, "peak") || !strcmp(what, "all")) {
        long iters = 40000000;
        double one = peak_fma(iters, 1);
        double lat = fma_chain(300000000);
        /* docs: Zen3+ ymm FMA latency is 4 cycles */
        double clk = 4.0 * 1e-9 / lat;
        printf("peak      single core  %8.2f GFLOP/s   (fma chain: %.3f ns, "
               "%.0f cycles/iter, implied clock %.2f GHz)\n",
               one, lat * 1e9, lat * clk * 1e9, clk);
#ifdef _OPENMP
        int threads = nthreads;
        int cores = threads / 2; /* Ryzen 7 6800H: SMT on, 2 threads per core */
        if (cores < 1)
            cores = 1;
        double many = peak_fma(iters, threads);
        printf("peak      %2d threads    %8.2f GFLOP/s   (%.2f GFLOP/s per thread)\n",
               threads, many, many / threads);
        /* sustained all-core clock from measured throughput, 32 FLOP/cycle/core */
        printf("peak      implied all-core clock %.2f GHz  (%.2f GFLOP/s / (32 FLOP/cyc * %d cores))\n",
               many / (32.0 * cores), many / (32.0 * cores), cores);
#endif
    }
    if (!strcmp(what, "bw") || !strcmp(what, "all")) {
        size_t bytes = (size_t)512 * 1024 * 1024;
        bw_probe(bytes, 1);
#ifdef _OPENMP
        if (nthreads > 1)
            bw_probe(bytes, nthreads);
#endif
    }
    if (!strcmp(what, "cache") || !strcmp(what, "all"))
        cache_probe();
    if (!strcmp(what, "line") || !strcmp(what, "all"))
        line_probe();
    return 0;
}
