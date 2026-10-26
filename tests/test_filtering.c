/* Digital filtering (lfilter, medfilt, Savitzky-Golay), checked against scipy.signal. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}
static int vec_eq(Tensor* t, const float* e, int n, float tol) {
    int ok = t && (int)t->numel == n;
    for (int i = 0; ok && i < n; i++)
        ok &= fabsf(at(t, i) - e[i]) < tol;
    return ok;
}

static int test_lfilter(void) {
    float x[]   = {1, 2, 3, 4, 5};
    float bma[] = {0.5f, 0.5f}, a1[] = {1.0f};
    float b1[] = {1.0f}, aiir[] = {1.0f, -0.5f};
    Tensor *X = cml_tensor_1d(x, 5), *B = cml_tensor_1d(bma, 2), *A = cml_tensor_1d(a1, 1);
    Tensor *B2 = cml_tensor_1d(b1, 1), *A2 = cml_tensor_1d(aiir, 2);
    float ema[]  = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
    float eiir[] = {1.0f, 2.5f, 4.25f, 6.125f, 8.0625f};
    Tensor* M    = cml_lfilter(B, A, X);
    Tensor* I    = cml_lfilter(B2, A2, X);
    int ok       = vec_eq(M, ema, 5, 1e-5f) && vec_eq(I, eiir, 5, 1e-4f);
    tensor_free(X);
    tensor_free(B);
    tensor_free(A);
    tensor_free(B2);
    tensor_free(A2);
    tensor_free(M);
    tensor_free(I);
    return ok;
}

static int test_medfilt(void) {
    float x[] = {2, 80, 6, 3, 1};
    float e[] = {2, 6, 6, 3, 1};
    Tensor* X = cml_tensor_1d(x, 5);
    Tensor* Y = cml_medfilt(X, 3);
    int ok    = vec_eq(Y, e, 5, 1e-6f);
    tensor_free(X);
    tensor_free(Y);
    return ok;
}

static int test_savgol(void) {
    float x[] = {2, 8, 3, 7, 4, 6, 5};
    float e[] = {2.914286f, 5.142857f, 6.085714f, 4.6f, 5.714286f, 5.657143f, 4.885714f};
    Tensor* X = cml_tensor_1d(x, 7);
    Tensor* Y = cml_savgol_filter(X, 5, 2);
    int ok    = vec_eq(Y, e, 7, 1e-4f);
    tensor_free(X);
    tensor_free(Y);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== digital filtering ===\n");
    TEST(lfilter);
    TEST(medfilt);
    TEST(savgol);
    cml_cleanup();
    return TEST_SUMMARY();
}
