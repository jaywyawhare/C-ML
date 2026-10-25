/* Linear-algebra extras (matrix power, condition number, triangular solve, matrix exponential),
 * checked against numpy.linalg and scipy.linalg. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_matrix_power(void) {
    float a[]   = {1, 2, 3, 4};
    float p3[]  = {37, 54, 81, 118};
    float p0[]  = {1, 0, 0, 1};
    float pm1[] = {-2, 1, 1.5f, -0.5f};
    Tensor* A   = cml_tensor_2d(a, 2, 2);
    Tensor* P3  = cml_matrix_power(A, 3);
    Tensor* P0  = cml_matrix_power(A, 0);
    Tensor* PM1 = cml_matrix_power(A, -1);
    int ok      = P3 && P0 && PM1;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(at(P3, i) - p3[i]) < 1e-3f && fabsf(at(P0, i) - p0[i]) < 1e-5f &&
              fabsf(at(PM1, i) - pm1[i]) < 1e-4f;
    tensor_free(A);
    tensor_free(P3);
    tensor_free(P0);
    tensor_free(PM1);
    return ok;
}

static int test_cond(void) {
    float a[] = {1, 2, 3, 4};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    int ok    = fabsf(cml_cond(A) - 14.933034f) < 1e-2f; /* np.linalg.cond */
    tensor_free(A);
    return ok;
}

static int test_solve_triangular(void) {
    /* lower [[2,0],[1,3]] x = [4,5] -> x=[2,1]. */
    float a[] = {2, 0, 1, 3};
    float b[] = {4, 5};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    Tensor* B = cml_tensor_1d(b, 2);
    Tensor* X = cml_solve_triangular(A, B, true);
    int ok    = X && fabsf(at(X, 0) - 2.0f) < 1e-5f && fabsf(at(X, 1) - 1.0f) < 1e-5f;
    tensor_free(A);
    tensor_free(B);
    tensor_free(X);
    return ok;
}

static int test_expm(void) {
    /* expm([[0,1],[0,0]]) = [[1,1],[0,1]]; expm(diag(1,2)) = diag(e, e^2). */
    float nil[] = {0, 1, 0, 0};
    float dia[] = {1, 0, 0, 2};
    Tensor* N   = cml_tensor_2d(nil, 2, 2);
    Tensor* D   = cml_tensor_2d(dia, 2, 2);
    Tensor* EN  = cml_expm(N);
    Tensor* ED  = cml_expm(D);
    int ok      = EN && ED && fabsf(at(EN, 0) - 1.0f) < 1e-4f && fabsf(at(EN, 1) - 1.0f) < 1e-4f &&
             fabsf(at(EN, 2)) < 1e-4f && fabsf(at(EN, 3) - 1.0f) < 1e-4f &&
             fabsf(at(ED, 0) - 2.718282f) < 1e-3f && fabsf(at(ED, 1)) < 1e-4f &&
             fabsf(at(ED, 2)) < 1e-4f && fabsf(at(ED, 3) - 7.389056f) < 1e-3f;
    tensor_free(N);
    tensor_free(D);
    tensor_free(EN);
    tensor_free(ED);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== linalg extras: matrix power, cond, triangular solve, expm ===\n");
    TEST(matrix_power);
    TEST(cond);
    TEST(solve_triangular);
    TEST(expm);
    cml_cleanup();
    return TEST_SUMMARY();
}
