/* Preprocessing scalers (standard, min-max, robust, L2 normalize), checked against
 * sklearn.preprocessing (StandardScaler/MinMaxScaler/RobustScaler/normalize). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int mat_eq(Tensor* t, const float* e, int n) {
    int ok = t && (int)t->numel == n;
    for (int i = 0; ok && i < n; i++)
        ok &= fabsf(at(t, i) - e[i]) < 1e-5f;
    return ok;
}

static int test_standard(void) {
    /* Each column standardizes to the same pattern (population std). */
    float x[] = {1, 10, 2, 20, 3, 30, 4, 40};
    float e[] = {-1.341641f, -1.341641f, -0.447214f, -0.447214f,
                 0.447214f,  0.447214f,  1.341641f,  1.341641f};
    Tensor* X = cml_tensor_2d(x, 4, 2);
    Tensor* S = cml_standard_scaler(X);
    int ok    = mat_eq(S, e, 8);
    tensor_free(X);
    tensor_free(S);
    return ok;
}

static int test_minmax(void) {
    float x[] = {1, 10, 2, 20, 3, 30, 4, 40};
    float e[] = {0, 0, 1.0f / 3, 1.0f / 3, 2.0f / 3, 2.0f / 3, 1, 1};
    Tensor* X = cml_tensor_2d(x, 4, 2);
    Tensor* M = cml_minmax_scaler(X);
    int ok    = mat_eq(M, e, 8);
    tensor_free(X);
    tensor_free(M);
    return ok;
}

static int test_robust(void) {
    /* median 3, IQR = Q3-Q1 = 4-2 = 2 -> (x-3)/2. */
    float x[] = {1, 2, 3, 4, 5};
    float e[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    Tensor* X = cml_tensor_2d(x, 5, 1);
    Tensor* R = cml_robust_scaler(X);
    int ok    = mat_eq(R, e, 5);
    tensor_free(X);
    tensor_free(R);
    return ok;
}

static int test_l2(void) {
    float x[] = {3, 4, 1, 0};
    float e[] = {0.6f, 0.8f, 1.0f, 0.0f};
    Tensor* X = cml_tensor_2d(x, 2, 2);
    Tensor* L = cml_normalize_l2(X);
    int ok    = mat_eq(L, e, 4);
    tensor_free(X);
    tensor_free(L);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== preprocessing scalers ===\n");
    TEST(standard);
    TEST(minmax);
    TEST(robust);
    TEST(l2);
    cml_cleanup();
    return TEST_SUMMARY();
}
