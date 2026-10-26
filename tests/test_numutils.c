/* Numerical utilities (nan_to_num, clip_by_norm), checked against NumPy and hand values. */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_nan_to_num(void) {
    float x[] = {1.0f, NAN, INFINITY, -INFINITY};
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* Y = cml_nan_to_num(X);
    int ok    = Y && fabsf(at(Y, 0) - 1.0f) < 1e-6f && at(Y, 1) == 0.0f && at(Y, 2) == FLT_MAX &&
             at(Y, 3) == -FLT_MAX;
    tensor_free(X);
    tensor_free(Y);
    return ok;
}

static int test_clip_by_norm(void) {
    float x[] = {3, 4}; /* norm 5 */
    Tensor* X = cml_tensor_1d(x, 2);
    Tensor* C = cml_clip_by_norm(X, 1.0f);  /* scale 0.2 -> [0.6,0.8] */
    Tensor* U = cml_clip_by_norm(X, 10.0f); /* unchanged */
    int ok    = C && U && fabsf(at(C, 0) - 0.6f) < 1e-6f && fabsf(at(C, 1) - 0.8f) < 1e-6f &&
             fabsf(at(U, 0) - 3.0f) < 1e-6f && fabsf(at(U, 1) - 4.0f) < 1e-6f;
    tensor_free(X);
    tensor_free(C);
    tensor_free(U);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== numerical utilities ===\n");
    TEST(nan_to_num);
    TEST(clip_by_norm);
    cml_cleanup();
    return TEST_SUMMARY();
}
