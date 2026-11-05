"""Generate src/attn_omp.c from src/attn_best.c: same kernels, parallel attention()."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "src" / "attn_best.c").read_text()

DRV = "/* ------------------------------------------------------------------------- */\n/* Driver: do not change below this line."
head_end = src.index(DRV)
body = src[src.index("#define _POSIX_C_SOURCE"):src.index("void attention(const float *Q")]
driver = src[head_end:]

attn = r'''void attention(const float *Q, const float *K, const float *V, float *O, int n, int d)
{
    float sc = 1.0f / sqrtf(d);
    int kn = n + PAD, sn = n + PAD;
    int nb = (n + BI - 1) / BI;

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    float *kt = malloc((size_t)kn * d * sizeof(float));
    float *sall = malloc((size_t)nt * BI * sn * sizeof(float));
    if (!kt || !sall) {
        printf("alloc failed\n");
        exit(1);
    }

    for (int k = 0; k < d; k++)
        for (int i = 0; i < n; i++)
            kt[(size_t)k * kn + i] = K[(size_t)i * d + k];

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *s = sall + (size_t)tid * BI * sn;
#pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < nb; b++) {
            int i = b * BI;
            int rows = (i + BI <= n) ? BI : n - i;
            block(Q + (size_t)i * d, kt, kn, V, O + (size_t)i * d, s, sn, rows, n, d, sc);
        }
    }

    free(kt);
    free(sall);
}
'''

(ROOT / "src" / "attn_omp.c").write_text(body + attn + driver)
print(f"  attn_omp.c  {(body + attn + driver).count(chr(10)):4d} lines")
