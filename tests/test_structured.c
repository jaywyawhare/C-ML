/* Structured matrices (Toeplitz, circulant, Hankel), checked against scipy.linalg. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int mat_eq(Tensor* t, const float* e, int rows, int cols) {
    int ok = t && t->shape[0] == rows && t->shape[1] == cols;
    for (int i = 0; ok && i < rows * cols; i++)
        ok &= fabsf(at(t, i) - e[i]) < 1e-6f;
    return ok;
}

static int test_toeplitz(void) {
    float c[] = {1, 2, 3}, r[] = {1, 4, 5, 6};
    float e[] = {1, 4, 5, 6, 2, 1, 4, 5, 3, 2, 1, 4};
    Tensor* C = cml_tensor_1d(c, 3);
    Tensor* R = cml_tensor_1d(r, 4);
    Tensor* T = cml_toeplitz(C, R);
    int ok    = mat_eq(T, e, 3, 4);
    tensor_free(C);
    tensor_free(R);
    tensor_free(T);
    return ok;
}

static int test_circulant(void) {
    float c[] = {1, 2, 3};
    float e[] = {1, 3, 2, 2, 1, 3, 3, 2, 1};
    Tensor* C = cml_tensor_1d(c, 3);
    Tensor* M = cml_circulant(C);
    int ok    = mat_eq(M, e, 3, 3);
    tensor_free(C);
    tensor_free(M);
    return ok;
}

static int test_hankel(void) {
    float c[] = {1, 2, 3}, r[] = {3, 4, 5};
    float e[] = {1, 2, 3, 2, 3, 4, 3, 4, 5};
    Tensor* C = cml_tensor_1d(c, 3);
    Tensor* R = cml_tensor_1d(r, 3);
    Tensor* H = cml_hankel(C, R);
    int ok    = mat_eq(H, e, 3, 3);
    tensor_free(C);
    tensor_free(R);
    tensor_free(H);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== structured matrices ===\n");
    TEST(toeplitz);
    TEST(circulant);
    TEST(hankel);
    cml_cleanup();
    return TEST_SUMMARY();
}
