/* Distance metrics round 2 (Minkowski, Chebyshev, Hamming, Mahalanobis), checked against
 * scipy.spatial.distance. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static int test_minkowski_chebyshev(void) {
    float a[] = {1, 2, 3}, b[] = {4, 6, 8};
    Tensor* A = cml_tensor_1d(a, 3);
    Tensor* B = cml_tensor_1d(b, 3);
    int ok    = fabsf(cml_minkowski_distance(A, B, 3.0f) - 6.0f) < 1e-4f && /* (27+64+125)^(1/3) */
             fabsf(cml_chebyshev_distance(A, B) - 5.0f) < 1e-6f;
    tensor_free(A);
    tensor_free(B);
    return ok;
}

static int test_hamming(void) {
    float u[] = {1, 0, 1, 1}, v[] = {1, 1, 0, 1};
    Tensor* U = cml_tensor_1d(u, 4);
    Tensor* V = cml_tensor_1d(v, 4);
    int ok    = fabsf(cml_hamming_distance(U, V) - 0.5f) < 1e-6f;
    tensor_free(U);
    tensor_free(V);
    return ok;
}

static int test_mahalanobis(void) {
    /* x=[2,2], mu=0, VI=diag(1/2,1) -> sqrt(4/2 + 4) = sqrt(6). */
    float x[] = {2, 2}, mu[] = {0, 0};
    float vi[] = {0.5f, 0, 0, 1};
    Tensor* X  = cml_tensor_1d(x, 2);
    Tensor* M  = cml_tensor_1d(mu, 2);
    Tensor* VI = cml_tensor_2d(vi, 2, 2);
    int ok     = fabsf(cml_mahalanobis_distance(X, M, VI) - sqrtf(6.0f)) < 1e-5f;
    tensor_free(X);
    tensor_free(M);
    tensor_free(VI);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== distance metrics (round 2) ===\n");
    TEST(minkowski_chebyshev);
    TEST(hamming);
    TEST(mahalanobis);
    cml_cleanup();
    return TEST_SUMMARY();
}
