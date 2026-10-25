/* Array utilities (diff, outer, tile, unique, trapz), checked against NumPy
 * (np.diff/outer/tile/unique/trapezoid). */
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

static int test_diff(void) {
    float a[]  = {1, 2, 4, 7, 0};
    float d1[] = {1, 2, 3, -7};
    float d2[] = {1, 1, -10};
    Tensor* A  = cml_tensor_1d(a, 5);
    Tensor* D1 = cml_diff(A, 1);
    Tensor* D2 = cml_diff(A, 2);
    int ok     = vec_eq(D1, d1, 4) && vec_eq(D2, d2, 3);
    tensor_free(A);
    tensor_free(D1);
    tensor_free(D2);
    return ok;
}

static int test_outer(void) {
    float a[] = {1, 2, 3}, b[] = {4, 5};
    float e[] = {4, 5, 8, 10, 12, 15};
    Tensor* A = cml_tensor_1d(a, 3);
    Tensor* B = cml_tensor_1d(b, 2);
    Tensor* O = cml_outer(A, B);
    int ok    = O && O->shape[0] == 3 && O->shape[1] == 2 && vec_eq(O, e, 6);
    tensor_free(A);
    tensor_free(B);
    tensor_free(O);
    return ok;
}

static int test_tile(void) {
    float a[] = {1, 2, 3};
    float e[] = {1, 2, 3, 1, 2, 3};
    Tensor* A = cml_tensor_1d(a, 3);
    Tensor* T = cml_tile(A, 2);
    int ok    = vec_eq(T, e, 6);
    tensor_free(A);
    tensor_free(T);
    return ok;
}

static int test_unique(void) {
    float a[] = {3, 1, 2, 1, 3, 5};
    float e[] = {1, 2, 3, 5};
    Tensor* A = cml_tensor_1d(a, 6);
    Tensor* U = cml_unique(A);
    int ok    = vec_eq(U, e, 4);
    tensor_free(A);
    tensor_free(U);
    return ok;
}

static int test_trapz(void) {
    float a[] = {1, 2, 3, 4};
    Tensor* A = cml_tensor_1d(a, 4);
    int ok    = fabsf(cml_trapz(A, 0.5f) - 3.75f) < 1e-5f; /* np.trapezoid(dx=0.5) */
    tensor_free(A);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== array utilities: diff, outer, tile, unique, trapz ===\n");
    TEST(diff);
    TEST(outer);
    TEST(tile);
    TEST(unique);
    TEST(trapz);
    cml_cleanup();
    return TEST_SUMMARY();
}
