/* Real FFT and 1-D convolution/correlation, checked against NumPy
 * (np.fft.rfft/irfft, np.convolve, np.correlate) and the rfft->irfft round-trip. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_rfft(void) {
    /* np.fft.rfft([1,2,3,4]) = [10, -2+2j, -2]. */
    float s[] = {1, 2, 3, 4};
    Tensor* S = cml_tensor_1d(s, 4);
    Tensor* R = cml_rfft(S);
    int ok    = R && R->shape[0] == 3 && R->shape[1] == 2;
    ok        = ok && fabsf(at(R, 0) - 10.0f) < 1e-4f && fabsf(at(R, 1)) < 1e-4f &&
         fabsf(at(R, 2) + 2.0f) < 1e-4f && fabsf(at(R, 3) - 2.0f) < 1e-4f &&
         fabsf(at(R, 4) + 2.0f) < 1e-4f && fabsf(at(R, 5)) < 1e-4f;
    tensor_free(S);
    tensor_free(R);
    return ok;
}

static int test_rfft_irfft_roundtrip(void) {
    float s[] = {3, 1, 4, 1, 5, 9, 2, 6};
    Tensor* S = cml_tensor_1d(s, 8);
    Tensor* R = cml_rfft(S);
    Tensor* B = R ? cml_irfft(R, 8) : NULL;
    int ok    = B && B->shape[0] == 8;
    for (int i = 0; ok && i < 8; i++)
        ok &= fabsf(at(B, i) - s[i]) < 1e-3f;
    tensor_free(S);
    tensor_free(R);
    tensor_free(B);
    return ok;
}

static int conv_eq(Tensor* out, const float* expect, int n) {
    int ok = out && out->shape[0] == n;
    for (int i = 0; ok && i < n; i++)
        ok &= fabsf(at(out, i) - expect[i]) < 1e-3f;
    return ok;
}

static int test_convolve(void) {
    float a[] = {1, 2, 3}, b[] = {0, 1, 0.5f};
    Tensor* A     = cml_tensor_1d(a, 3);
    Tensor* B     = cml_tensor_1d(b, 3);
    float full[]  = {0, 1, 2.5f, 4, 1.5f};
    float same[]  = {1, 2.5f, 4};
    float valid[] = {2.5f};
    Tensor* f     = cml_convolve1d(A, B, 0);
    Tensor* s     = cml_convolve1d(A, B, 1);
    Tensor* v     = cml_convolve1d(A, B, 2);
    int ok        = conv_eq(f, full, 5) && conv_eq(s, same, 3) && conv_eq(v, valid, 1);
    tensor_free(A);
    tensor_free(B);
    tensor_free(f);
    tensor_free(s);
    tensor_free(v);
    return ok;
}

static int test_correlate(void) {
    /* np.correlate([1,2,3],[0,1,0.5],'full') = [0.5, 2, 3.5, 3, 0]. */
    float a[] = {1, 2, 3}, b[] = {0, 1, 0.5f};
    Tensor* A     = cml_tensor_1d(a, 3);
    Tensor* B     = cml_tensor_1d(b, 3);
    float full[]  = {0.5f, 2, 3.5f, 3, 0};
    float valid[] = {3.5f};
    Tensor* f     = cml_correlate1d(A, B, 0);
    Tensor* v     = cml_correlate1d(A, B, 2);
    int ok        = conv_eq(f, full, 5) && conv_eq(v, valid, 1);
    tensor_free(A);
    tensor_free(B);
    tensor_free(f);
    tensor_free(v);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== signal: rfft, irfft, convolve, correlate ===\n");
    TEST(rfft);
    TEST(rfft_irfft_roundtrip);
    TEST(convolve);
    TEST(correlate);
    cml_cleanup();
    return TEST_SUMMARY();
}
