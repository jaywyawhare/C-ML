/* Set operations (intersect1d, union1d, setdiff1d, isin), checked against NumPy. */
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
        ok &= fabsf(at(t, i) - e[i]) < 1e-6f;
    return ok;
}

static int test_setops(void) {
    float x[] = {1, 2, 3, 4}, y[] = {3, 4, 5, 6};
    Tensor* X     = cml_tensor_1d(x, 4);
    Tensor* Y     = cml_tensor_1d(y, 4);
    float inter[] = {3, 4};
    float uni[]   = {1, 2, 3, 4, 5, 6};
    float diff[]  = {1, 2};
    float isin[]  = {0, 0, 1, 1};
    Tensor* I     = cml_intersect1d(X, Y);
    Tensor* U     = cml_union1d(X, Y);
    Tensor* D     = cml_setdiff1d(X, Y);
    Tensor* S     = cml_isin(X, Y);
    int ok = vec_eq(I, inter, 2) && vec_eq(U, uni, 6) && vec_eq(D, diff, 2) && vec_eq(S, isin, 4);
    tensor_free(X);
    tensor_free(Y);
    tensor_free(I);
    tensor_free(U);
    tensor_free(D);
    tensor_free(S);
    return ok;
}

static int test_intersect_dedup(void) {
    /* duplicates collapse; result is unique and sorted. */
    float x[] = {2, 2, 1, 3, 3}, y[] = {3, 2, 2};
    Tensor* X = cml_tensor_1d(x, 5);
    Tensor* Y = cml_tensor_1d(y, 3);
    float e[] = {2, 3};
    Tensor* I = cml_intersect1d(X, Y);
    int ok    = vec_eq(I, e, 2);
    tensor_free(X);
    tensor_free(Y);
    tensor_free(I);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== set operations ===\n");
    TEST(setops);
    TEST(intersect_dedup);
    cml_cleanup();
    return TEST_SUMMARY();
}
