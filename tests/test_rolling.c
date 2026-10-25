/* Rolling and cumulative ops (rolling sum/mean, EWMA, cummax, cummin), checked against
 * NumPy (convolve/accumulate) and the EWMA recurrence. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int vec_eq(Tensor* t, const float* e, int n) {
    int ok = t && (int)t->numel == n;
    for (int i = 0; ok && i < n; i++)
        ok &= fabsf(at(t, i) - e[i]) < 1e-5f;
    return ok;
}

static int test_rolling(void) {
    float a[]    = {1, 2, 3, 4, 5};
    float rsum[] = {6, 9, 12};
    float rmen[] = {2, 3, 4};
    Tensor* A    = cml_tensor_1d(a, 5);
    Tensor* S    = cml_rolling_sum(A, 3);
    Tensor* M    = cml_rolling_mean(A, 3);
    int ok       = vec_eq(S, rsum, 3) && vec_eq(M, rmen, 3);
    tensor_free(A);
    tensor_free(S);
    tensor_free(M);
    return ok;
}

static int test_ewma(void) {
    float a[] = {1, 2, 3, 4, 5};
    float e[] = {1.0f, 1.5f, 2.25f, 3.125f, 4.0625f}; /* alpha=0.5, adjust=False */
    Tensor* A = cml_tensor_1d(a, 5);
    Tensor* E = cml_ewma(A, 0.5f);
    int ok    = vec_eq(E, e, 5);
    tensor_free(A);
    tensor_free(E);
    return ok;
}

static int test_cum(void) {
    float a[]  = {3, 1, 4, 1, 5};
    float mx[] = {3, 3, 4, 4, 5};
    float mn[] = {3, 1, 1, 1, 1};
    Tensor* A  = cml_tensor_1d(a, 5);
    Tensor* MX = cml_cummax(A);
    Tensor* MN = cml_cummin(A);
    int ok     = vec_eq(MX, mx, 5) && vec_eq(MN, mn, 5);
    tensor_free(A);
    tensor_free(MX);
    tensor_free(MN);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== rolling & cumulative ops ===\n");
    TEST(rolling);
    TEST(ewma);
    TEST(cum);
    cml_cleanup();
    return TEST_SUMMARY();
}
