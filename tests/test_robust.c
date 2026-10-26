/* Robust statistics (MAD, trimmed mean, SEM, rankdata, winsorize), checked against SciPy
 * (scipy.stats.median_abs_deviation/trim_mean/sem/rankdata, scipy.stats.mstats.winsorize). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "core/metrics.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_mad_trim_sem(void) {
    float d[] = {1, 2, 3, 4, 100};
    Tensor* D = cml_tensor_1d(d, 5);
    int ok    = fabsf(cml_median_abs_deviation(D) - 1.0f) < 1e-5f &&
             fabsf(cml_trim_mean(D, 0.2f) - 3.0f) < 1e-5f && fabsf(cml_sem(D) - 19.506409f) < 1e-3f;
    tensor_free(D);
    return ok;
}

static int test_rankdata(void) {
    float v[] = {3, 1, 4, 1, 5};
    float e[] = {3, 1.5f, 4, 1.5f, 5};
    float out[5];
    cml_rankdata(v, 5, out);
    int ok = 1;
    for (int i = 0; i < 5; i++)
        ok &= fabsf(out[i] - e[i]) < 1e-6f;
    return ok;
}

static int test_winsorize(void) {
    float d[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float e[] = {2, 2, 3, 4, 5, 6, 7, 8, 9, 9};
    Tensor* D = cml_tensor_1d(d, 10);
    Tensor* W = cml_winsorize(D, 0.1f, 0.1f);
    int ok    = W && (int)W->numel == 10;
    for (int i = 0; ok && i < 10; i++)
        ok &= fabsf(at(W, i) - e[i]) < 1e-6f;
    tensor_free(D);
    tensor_free(W);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== robust statistics ===\n");
    TEST(mad_trim_sem);
    TEST(rankdata);
    TEST(winsorize);
    cml_cleanup();
    return TEST_SUMMARY();
}
