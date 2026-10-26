/* Signal features (zero-crossing rate, RMS energy, spectral centroid), checked against
 * NumPy references. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static int test_zcr(void) {
    /* sign flips at (2,-1) and (-2,3): 2 of 4 adjacent pairs -> 0.5. */
    float x[] = {1, 2, -1, -2, 3};
    Tensor* X = cml_tensor_1d(x, 5);
    int ok    = fabsf(cml_zero_crossing_rate(X) - 0.5f) < 1e-6f;
    tensor_free(X);
    return ok;
}

static int test_rms(void) {
    float x[] = {0, 1, 2, 3};
    Tensor* X = cml_tensor_1d(x, 4);
    int ok    = fabsf(cml_rms_energy(X) - 1.8708287f) < 1e-5f; /* sqrt(14/4) */
    tensor_free(X);
    return ok;
}

static int test_spectral_centroid(void) {
    float y[] = {1, 2, 1, 0, 1, 2, 1, 0};
    Tensor* Y = cml_tensor_1d(y, 8);
    int ok    = fabsf(cml_spectral_centroid(Y, 8.0f) - 0.6666667f) < 1e-4f;
    tensor_free(Y);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== signal features ===\n");
    TEST(zcr);
    TEST(rms);
    TEST(spectral_centroid);
    cml_cleanup();
    return TEST_SUMMARY();
}
