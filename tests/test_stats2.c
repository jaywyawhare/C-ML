/* Higher-order statistics (skewness, kurtosis, z-score, nanmean, nansum), checked against
 * SciPy (scipy.stats.skew/kurtosis/zscore) and NumPy (np.nanmean/nansum). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_skew_kurt(void) {
    float d[] = {2, 4, 4, 4, 5, 5, 7, 9};
    Tensor* D = cml_tensor_1d(d, 8);
    int ok    = fabsf(cml_skewness(D) - 0.65625f) < 1e-4f && /* scipy.stats.skew */
             fabsf(cml_kurtosis(D) + 0.21875f) < 1e-4f;      /* scipy.stats.kurtosis */
    tensor_free(D);
    return ok;
}

static int test_zscore(void) {
    float d[] = {2, 4, 4, 4, 5, 5, 7, 9};
    float e[] = {-1.5f, -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 2.0f};
    Tensor* D = cml_tensor_1d(d, 8);
    Tensor* Z = cml_zscore(D);
    int ok    = Z && (int)Z->numel == 8;
    for (int i = 0; ok && i < 8; i++)
        ok &= fabsf(at(Z, i) - e[i]) < 1e-5f;
    tensor_free(D);
    tensor_free(Z);
    return ok;
}

static int test_nan_reductions(void) {
    float d[] = {1.0f, NAN, 3.0f, NAN, 5.0f};
    Tensor* D = cml_tensor_1d(d, 5);
    int ok    = fabsf(cml_nanmean(D) - 3.0f) < 1e-6f && fabsf(cml_nansum(D) - 9.0f) < 1e-6f;
    tensor_free(D);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== higher-order statistics ===\n");
    TEST(skew_kurt);
    TEST(zscore);
    TEST(nan_reductions);
    cml_cleanup();
    return TEST_SUMMARY();
}
