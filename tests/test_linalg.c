/* Dense linear algebra: solve, inverse, determinant, Cholesky and least squares, each
 * checked against hand-computed results and solver identities. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static int close_at(Tensor* t, int i, float expect) {
    float* d = (float*)tensor_data_ptr(t);
    return d && fabsf(d[i] - expect) < 1e-4f;
}

static int test_solve(void) {
    float a[] = {3, 2, 1, 2};
    float b[] = {5, 5};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    Tensor* B = cml_tensor_1d(b, 2);
    Tensor* x = cml_solve(A, B); /* x=0, y=2.5 */
    int ok    = x && close_at(x, 0, 0.0f) && close_at(x, 1, 2.5f);
    tensor_free(A);
    tensor_free(B);
    tensor_free(x);
    return ok;
}

static int test_inverse_and_det(void) {
    float a[]  = {4, 7, 2, 6};
    Tensor* A  = cml_tensor_2d(a, 2, 2);
    Tensor* iv = cml_inverse(A); /* [[0.6,-0.7],[-0.2,0.4]] */
    float det  = cml_det(A);     /* 10 */
    int ok     = iv && close_at(iv, 0, 0.6f) && close_at(iv, 1, -0.7f) && close_at(iv, 2, -0.2f) &&
             close_at(iv, 3, 0.4f) && fabsf(det - 10.0f) < 1e-3f;
    tensor_free(A);
    tensor_free(iv);
    return ok;
}

static int test_cholesky(void) {
    float a[] = {4, 2, 2, 3};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    Tensor* L = cml_cholesky(A); /* [[2,0],[1,sqrt2]] */
    int ok    = L && close_at(L, 0, 2.0f) && close_at(L, 1, 0.0f) && close_at(L, 2, 1.0f) &&
             close_at(L, 3, sqrtf(2.0f));
    tensor_free(A);
    tensor_free(L);
    return ok;
}

static int test_lstsq(void) {
    float a[] = {1, 0, 1, 1, 1, 2}; /* columns: [1, t], t=0,1,2 */
    float b[] = {1, 2, 3};          /* exact line y = 1 + t */
    Tensor* A = cml_tensor_2d(a, 3, 2);
    Tensor* B = cml_tensor_1d(b, 3);
    Tensor* x = cml_lstsq(A, B); /* intercept 1, slope 1 */
    int ok    = x && close_at(x, 0, 1.0f) && close_at(x, 1, 1.0f);
    tensor_free(A);
    tensor_free(B);
    tensor_free(x);
    return ok;
}

static int test_rejects_singular(void) {
    float a[] = {1, 2, 2, 4}; /* singular */
    float b[] = {1, 1};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    Tensor* B = cml_tensor_1d(b, 2);
    int ok    = cml_solve(A, B) == NULL && cml_inverse(A) == NULL && fabsf(cml_det(A)) < 1e-3f;
    tensor_free(A);
    tensor_free(B);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== dense linear algebra ===\n");
    TEST(solve);
    TEST(inverse_and_det);
    TEST(cholesky);
    TEST(lstsq);
    TEST(rejects_singular);
    cml_cleanup();
    return TEST_SUMMARY();
}
