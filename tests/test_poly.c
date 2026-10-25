/* Polynomials (Vandermonde, polyfit, polyval), checked against NumPy
 * (np.vander/polyfit/polyval) and the fit->eval round-trip. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_vander(void) {
    /* np.vander([1,2,3],3) decreasing -> rows [x^2, x^1, 1]. */
    float x[] = {1, 2, 3};
    float e[] = {1, 1, 1, 4, 2, 1, 9, 3, 1};
    Tensor* X = cml_tensor_1d(x, 3);
    Tensor* V = cml_vander(X, 3, false);
    int ok    = V && V->shape[0] == 3 && V->shape[1] == 3;
    for (int i = 0; ok && i < 9; i++)
        ok &= fabsf(at(V, i) - e[i]) < 1e-5f;
    tensor_free(X);
    tensor_free(V);
    return ok;
}

static int test_polyfit(void) {
    /* y = x^2 + x + 1 -> coefficients [1,1,1]. */
    float x[] = {0, 1, 2, 3};
    float y[] = {1, 3, 7, 13};
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* Y = cml_tensor_1d(y, 4);
    Tensor* C = cml_polyfit(X, Y, 2);
    int ok    = C && C->shape[0] == 3 && fabsf(at(C, 0) - 1.0f) < 1e-3f &&
             fabsf(at(C, 1) - 1.0f) < 1e-3f && fabsf(at(C, 2) - 1.0f) < 1e-3f;
    tensor_free(X);
    tensor_free(Y);
    tensor_free(C);
    return ok;
}

static int test_polyval(void) {
    /* np.polyval([1,-2,1],[0,1,2,3]) = (x-1)^2 = [1,0,1,4]. */
    float c[] = {1, -2, 1};
    float x[] = {0, 1, 2, 3};
    float e[] = {1, 0, 1, 4};
    Tensor* C = cml_tensor_1d(c, 3);
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* Y = cml_polyval(C, X);
    int ok    = Y && (int)Y->numel == 4;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(at(Y, i) - e[i]) < 1e-5f;
    tensor_free(C);
    tensor_free(X);
    tensor_free(Y);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== polynomials: vander, polyfit, polyval ===\n");
    TEST(vander);
    TEST(polyfit);
    TEST(polyval);
    cml_cleanup();
    return TEST_SUMMARY();
}
