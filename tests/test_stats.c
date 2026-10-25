/* Descriptive statistics (median, quantile, percentile, covariance, correlation), checked
 * against NumPy reference values (np.median/quantile/cov/corrcoef). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_median_quantile(void) {
    float x[] = {3, 1, 4, 1, 5, 9, 2, 6};
    Tensor* X = cml_tensor_1d(x, 8);
    int ok    = fabsf(cml_median(X) - 3.5f) < 1e-5f &&        /* np.median */
             fabsf(cml_quantile(X, 0.25f) - 1.75f) < 1e-5f && /* np.quantile 0.25 */
             fabsf(cml_quantile(X, 0.90f) - 6.9f) < 1e-4f &&  /* np.quantile 0.90 */
             fabsf(cml_percentile(X, 50.0f) - 3.5f) < 1e-5f;
    tensor_free(X);
    return ok;
}

static int test_quantile_edges(void) {
    float x[] = {2, 8, 5};
    Tensor* X = cml_tensor_1d(x, 3);
    int ok    = fabsf(cml_quantile(X, 0.0f) - 2.0f) < 1e-6f && /* min */
             fabsf(cml_quantile(X, 1.0f) - 8.0f) < 1e-6f;      /* max */
    tensor_free(X);
    return ok;
}

static int test_cov(void) {
    /* rows are variables; np.cov -> [[5/3, 1],[1, 5/3]]. */
    float m[] = {1, 2, 3, 4, 2, 1, 4, 3};
    Tensor* M = cml_tensor_2d(m, 2, 4);
    Tensor* C = cml_cov(M, true);
    int ok    = C && fabsf(at(C, 0) - 5.0f / 3.0f) < 1e-5f && fabsf(at(C, 1) - 1.0f) < 1e-5f &&
             fabsf(at(C, 2) - 1.0f) < 1e-5f && fabsf(at(C, 3) - 5.0f / 3.0f) < 1e-5f;
    tensor_free(M);
    tensor_free(C);
    return ok;
}

static int test_corrcoef(void) {
    /* np.corrcoef -> [[1, 0.6],[0.6, 1]]. */
    float m[] = {1, 2, 3, 4, 2, 1, 4, 3};
    Tensor* M = cml_tensor_2d(m, 2, 4);
    Tensor* R = cml_corrcoef(M, true);
    int ok    = R && fabsf(at(R, 0) - 1.0f) < 1e-5f && fabsf(at(R, 1) - 0.6f) < 1e-5f &&
             fabsf(at(R, 2) - 0.6f) < 1e-5f && fabsf(at(R, 3) - 1.0f) < 1e-5f;
    tensor_free(M);
    tensor_free(R);
    return ok;
}

static int test_rejects_bad(void) {
    float m[] = {1, 2};
    Tensor* M = cml_tensor_2d(m, 2, 1); /* one observation per variable -> undefined cov */
    Tensor* C = cml_cov(M, true);
    int ok    = C == NULL;
    tensor_free(M);
    tensor_free(C);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== descriptive statistics ===\n");
    TEST(median_quantile);
    TEST(quantile_edges);
    TEST(cov);
    TEST(corrcoef);
    TEST(rejects_bad);
    cml_cleanup();
    return TEST_SUMMARY();
}
