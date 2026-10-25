/* Special functions (erfc, lgamma, digamma, logit), checked against scipy.special. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int close4(Tensor* t, const float* e) {
    int ok = t != NULL;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(at(t, i) - e[i]) < 1e-5f;
    return ok;
}

static int test_erfc(void) {
    float x[] = {0.0f, 0.5f, 1.0f, 2.0f};
    float e[] = {1.0f, 0.479500f, 0.157299f, 0.004678f};
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* R = cml_erfc(X);
    int ok    = close4(R, e);
    tensor_free(X);
    tensor_free(R);
    return ok;
}

static int test_lgamma(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.5f};
    float e[] = {0.0f, 0.0f, 0.693147f, 2.453737f};
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* R = cml_lgamma(X);
    int ok    = close4(R, e);
    tensor_free(X);
    tensor_free(R);
    return ok;
}

static int test_digamma(void) {
    float x[] = {1.0f, 2.0f, 5.0f, 10.0f};
    float e[] = {-0.577216f, 0.422784f, 1.506118f, 2.251753f};
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* R = cml_digamma(X);
    int ok    = R && fabsf(at(R, 0) - e[0]) < 1e-4f && fabsf(at(R, 1) - e[1]) < 1e-4f &&
             fabsf(at(R, 2) - e[2]) < 1e-4f && fabsf(at(R, 3) - e[3]) < 1e-4f;
    tensor_free(X);
    tensor_free(R);
    return ok;
}

static int test_logit(void) {
    float x[] = {0.5f, 0.25f, 0.75f};
    float e[] = {0.0f, -1.098612f, 1.098612f};
    Tensor* X = cml_tensor_1d(x, 3);
    Tensor* R = cml_logit(X);
    int ok    = R && fabsf(at(R, 0) - e[0]) < 1e-5f && fabsf(at(R, 1) - e[1]) < 1e-5f &&
             fabsf(at(R, 2) - e[2]) < 1e-5f;
    tensor_free(X);
    tensor_free(R);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== special functions ===\n");
    TEST(erfc);
    TEST(lgamma);
    TEST(digamma);
    TEST(logit);
    cml_cleanup();
    return TEST_SUMMARY();
}
