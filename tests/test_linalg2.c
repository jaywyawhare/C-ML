/* Frobenius norm, matrix rank, pseudo-inverse and symmetric eigendecomposition, checked
 * against hand-computed values and the defining identities (A x = lambda x, A pinv A = A). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_norm(void) {
    float a[] = {1, 2, 3, 4};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    int ok    = fabsf(cml_matrix_norm(A) - sqrtf(30.0f)) < 1e-4f;
    tensor_free(A);
    return ok;
}

static int test_rank(void) {
    float sing[] = {1, 2, 2, 4}; /* rank 1 */
    float full[] = {1, 0, 0, 1}; /* rank 2 */
    Tensor* S    = cml_tensor_2d(sing, 2, 2);
    Tensor* F    = cml_tensor_2d(full, 2, 2);
    int ok       = cml_matrix_rank(S, -1.0f) == 1 && cml_matrix_rank(F, -1.0f) == 2;
    tensor_free(S);
    tensor_free(F);
    return ok;
}

static int test_pinv_square(void) {
    float a[]  = {4, 7, 2, 6};
    Tensor* A  = cml_tensor_2d(a, 2, 2);
    Tensor* pi = cml_pinv(A); /* full-rank square -> equals inverse */
    int ok     = pi && fabsf(at(pi, 0) - 0.6f) < 1e-3f && fabsf(at(pi, 1) + 0.7f) < 1e-3f &&
             fabsf(at(pi, 2) + 0.2f) < 1e-3f && fabsf(at(pi, 3) - 0.4f) < 1e-3f;
    tensor_free(A);
    tensor_free(pi);
    return ok;
}

static int test_pinv_tall(void) {
    float a[]  = {1, 0, 1, 1, 1, 2}; /* [3,2], full column rank */
    Tensor* A  = cml_tensor_2d(a, 3, 2);
    Tensor* pi = cml_pinv(A); /* [2,3], left inverse: pi @ A = I2 */
    Tensor* I  = (pi) ? cml_matmul(pi, A) : NULL;
    int ok     = I && fabsf(at(I, 0) - 1.0f) < 1e-3f && fabsf(at(I, 1)) < 1e-3f &&
             fabsf(at(I, 2)) < 1e-3f && fabsf(at(I, 3) - 1.0f) < 1e-3f;
    tensor_free(A);
    tensor_free(pi);
    tensor_free(I);
    return ok;
}

static int test_eigh(void) {
    float a[]   = {2, 1, 1, 2}; /* eigenvalues 1 (ascending) and 3 */
    Tensor* A   = cml_tensor_2d(a, 2, 2);
    EigResult e = cml_eigh(A);
    int ok      = e.values && e.vectors && fabsf(at(e.values, 0) - 1.0f) < 1e-4f &&
             fabsf(at(e.values, 1) - 3.0f) < 1e-4f;
    /* verify A x = lambda x for each eigenvector column */
    for (int c = 0; ok && c < 2; c++) {
        float v0 = at(e.vectors, 0 * 2 + c), v1 = at(e.vectors, 1 * 2 + c);
        float lam = at(e.values, c);
        ok &= fabsf((2 * v0 + 1 * v1) - lam * v0) < 1e-4f;
        ok &= fabsf((1 * v0 + 2 * v1) - lam * v1) < 1e-4f;
    }
    tensor_free(A);
    tensor_free(e.values);
    tensor_free(e.vectors);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== linalg: norm, rank, pinv, eigh ===\n");
    TEST(norm);
    TEST(rank);
    TEST(pinv_square);
    TEST(pinv_tall);
    TEST(eigh);
    cml_cleanup();
    return TEST_SUMMARY();
}
