/* Histogram, bincount, searchsorted and 1-D interpolation, checked against NumPy
 * (np.histogram, np.bincount, np.searchsorted, np.interp). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_histogram(void) {
    /* np.histogram(a, bins=3, range=(0,3)) -> [2,1,3]. */
    float a[] = {0.5f, 0.5f, 1.5f, 2.5f, 2.5f, 2.9f};
    Tensor* A = cml_tensor_1d(a, 6);
    Tensor* H = cml_histogram(A, 3, 0.0f, 3.0f);
    int ok    = H && H->shape[0] == 3 && fabsf(at(H, 0) - 2.0f) < 1e-6f &&
             fabsf(at(H, 1) - 1.0f) < 1e-6f && fabsf(at(H, 2) - 3.0f) < 1e-6f;
    tensor_free(A);
    tensor_free(H);
    return ok;
}

static int test_bincount(void) {
    /* np.bincount([0,1,1,3,3,3], minlength=5) -> [1,2,0,3,0]. */
    float a[]      = {0, 1, 1, 3, 3, 3};
    float expect[] = {1, 2, 0, 3, 0};
    Tensor* A      = cml_tensor_1d(a, 6);
    Tensor* B      = cml_bincount(A, 5);
    int ok         = B && B->shape[0] == 5;
    for (int i = 0; ok && i < 5; i++)
        ok &= fabsf(at(B, i) - expect[i]) < 1e-6f;
    tensor_free(A);
    tensor_free(B);
    return ok;
}

static int test_searchsorted(void) {
    float sp[] = {1, 3, 5, 7};
    Tensor* S  = cml_tensor_1d(sp, 4);
    int ok     = cml_searchsorted(S, 4.0f, 0) == 2 && /* left */
             cml_searchsorted(S, 5.0f, 0) == 2 &&     /* left, exact */
             cml_searchsorted(S, 5.0f, 1) == 3 &&     /* right, exact */
             cml_searchsorted(S, 0.0f, 0) == 0 && cml_searchsorted(S, 9.0f, 0) == 4;
    tensor_free(S);
    return ok;
}

static int test_interp(void) {
    /* np.interp([0,1.5,2.5,4], [1,2,3], [10,20,10]) -> [10,15,15,10]. */
    float xq[]     = {0, 1.5f, 2.5f, 4};
    float xp[]     = {1, 2, 3};
    float fp[]     = {10, 20, 10};
    float expect[] = {10, 15, 15, 10};
    Tensor* X      = cml_tensor_1d(xq, 4);
    Tensor* XP     = cml_tensor_1d(xp, 3);
    Tensor* FP     = cml_tensor_1d(fp, 3);
    Tensor* Y      = cml_interp1d(X, XP, FP);
    int ok         = Y && Y->shape[0] == 4;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(at(Y, i) - expect[i]) < 1e-5f;
    tensor_free(X);
    tensor_free(XP);
    tensor_free(FP);
    tensor_free(Y);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== binning: histogram, bincount, searchsorted, interp ===\n");
    TEST(histogram);
    TEST(bincount);
    TEST(searchsorted);
    TEST(interp);
    cml_cleanup();
    return TEST_SUMMARY();
}
