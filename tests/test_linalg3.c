/* Linear-algebra extras (trace, Kronecker product, cross product, slogdet), checked against
 * NumPy reference values (np.trace/kron/cross/linalg.slogdet). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_trace(void) {
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    Tensor* A = cml_tensor_2d(a, 3, 3);
    int ok    = fabsf(cml_trace(A) - 16.0f) < 1e-5f; /* 1 + 5 + 10 */
    tensor_free(A);
    return ok;
}

static int test_kron(void) {
    /* np.kron([[1,2],[3,4]],[[0,5],[6,7]]). */
    float a[]      = {1, 2, 3, 4};
    float b[]      = {0, 5, 6, 7};
    float expect[] = {0, 5, 0, 10, 6, 7, 12, 14, 0, 15, 0, 20, 18, 21, 24, 28};
    Tensor* A      = cml_tensor_2d(a, 2, 2);
    Tensor* B      = cml_tensor_2d(b, 2, 2);
    Tensor* K      = cml_kron(A, B);
    int ok         = K && K->shape[0] == 4 && K->shape[1] == 4;
    for (int i = 0; ok && i < 16; i++)
        ok &= fabsf(at(K, i) - expect[i]) < 1e-5f;
    tensor_free(A);
    tensor_free(B);
    tensor_free(K);
    return ok;
}

static int test_cross(void) {
    /* np.cross([1,2,3],[4,5,6]) = [-3, 6, -3]. */
    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};
    Tensor* A = cml_tensor_1d(a, 3);
    Tensor* B = cml_tensor_1d(b, 3);
    Tensor* C = cml_cross(A, B);
    int ok    = C && fabsf(at(C, 0) + 3.0f) < 1e-5f && fabsf(at(C, 1) - 6.0f) < 1e-5f &&
             fabsf(at(C, 2) + 3.0f) < 1e-5f;
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    return ok;
}

static int test_slogdet(void) {
    /* det = -3, so sign = -1 and logabsdet = ln 3. */
    float a[]       = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    Tensor* A       = cml_tensor_2d(a, 3, 3);
    SlogdetResult s = cml_slogdet(A);
    int ok          = fabsf(s.sign + 1.0f) < 1e-5f && fabsf(s.logabsdet - logf(3.0f)) < 1e-4f;
    tensor_free(A);
    return ok;
}

static int test_slogdet_singular(void) {
    float a[]       = {1, 2, 2, 4}; /* singular */
    Tensor* A       = cml_tensor_2d(a, 2, 2);
    SlogdetResult s = cml_slogdet(A);
    int ok          = s.sign == 0.0f && isinf(s.logabsdet) && s.logabsdet < 0.0f;
    tensor_free(A);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== linalg extras: trace, kron, cross, slogdet ===\n");
    TEST(trace);
    TEST(kron);
    TEST(cross);
    TEST(slogdet);
    TEST(slogdet_singular);
    cml_cleanup();
    return TEST_SUMMARY();
}
