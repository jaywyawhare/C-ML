/* Signal round 3 (periodogram, FFT resample), checked against scipy.signal.periodogram/resample. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_periodogram(void) {
    /* period-4 signal -> one-sided PSD concentrates at bin 2 (f=0.25): scipy gives [0,0,4,0,0]. */
    float x[] = {1, 2, 1, 0, 1, 2, 1, 0};
    float e[] = {0, 0, 4, 0, 0};
    Tensor* X = cml_tensor_1d(x, 8);
    Tensor* P = cml_periodogram(X, 1.0f);
    int ok    = P && (int)P->numel == 5;
    for (int i = 0; ok && i < 5; i++)
        ok &= fabsf(at(P, i) - e[i]) < 1e-4f;
    tensor_free(X);
    tensor_free(P);
    return ok;
}

static int test_resample(void) {
    /* scipy.signal.resample([1,2,3,4], 8). */
    float x[] = {1, 2, 3, 4};
    float e[] = {1.0f, 1.085786f, 2.0f, 2.5f, 3.0f, 3.914214f, 4.0f, 2.5f};
    Tensor* X = cml_tensor_1d(x, 4);
    Tensor* R = cml_resample(X, 8);
    int ok    = R && (int)R->numel == 8;
    for (int i = 0; ok && i < 8; i++)
        ok &= fabsf(at(R, i) - e[i]) < 1e-3f;
    tensor_free(X);
    tensor_free(R);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== signal round 3: periodogram, resample ===\n");
    TEST(periodogram);
    TEST(resample);
    cml_cleanup();
    return TEST_SUMMARY();
}
