/* FFT/signal helpers (fftfreq, rfftfreq, detrend, hilbert), checked against NumPy and SciPy
 * (np.fft.fftfreq/rfftfreq, scipy.signal.detrend/hilbert). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_fftfreq(void) {
    float even[] = {0.0f, 0.125f, 0.25f, 0.375f, -0.5f, -0.375f, -0.25f, -0.125f};
    float odd[]  = {0.0f, 0.2f, 0.4f, -0.4f, -0.2f};
    float rf[]   = {0.0f, 0.125f, 0.25f, 0.375f, 0.5f};
    Tensor* E    = cml_fftfreq(8, 1.0f, &CFG);
    Tensor* O    = cml_fftfreq(5, 1.0f, &CFG);
    Tensor* R    = cml_rfftfreq(8, 1.0f, &CFG);
    int ok       = E && O && R && E->shape[0] == 8 && O->shape[0] == 5 && R->shape[0] == 5;
    for (int i = 0; ok && i < 8; i++)
        ok &= fabsf(at(E, i) - even[i]) < 1e-6f;
    for (int i = 0; ok && i < 5; i++)
        ok &= fabsf(at(O, i) - odd[i]) < 1e-6f && fabsf(at(R, i) - rf[i]) < 1e-6f;
    tensor_free(E);
    tensor_free(O);
    tensor_free(R);
    return ok;
}

static int test_detrend(void) {
    float y[]    = {1.5f, 1.7f, 3.1f, 3.8f, 5.4f};
    float lin[]  = {0.38f, -0.41f, 0.0f, -0.29f, 0.32f};
    float cons[] = {-1.6f, -1.4f, 0.0f, 0.7f, 2.3f};
    Tensor* Y    = cml_tensor_1d(y, 5);
    Tensor* L    = cml_detrend(Y, 1);
    Tensor* C    = cml_detrend(Y, 0);
    int ok       = L && C;
    for (int i = 0; ok && i < 5; i++)
        ok &= fabsf(at(L, i) - lin[i]) < 1e-4f && fabsf(at(C, i) - cons[i]) < 1e-4f;
    tensor_free(Y);
    tensor_free(L);
    tensor_free(C);
    return ok;
}

static int test_hilbert(void) {
    /* analytic signal: real part reconstructs x, imag matches scipy.signal.hilbert. */
    float x[]  = {1, 2, 3, 4, 3, 2, 1, 0};
    float im[] = {-1.414214f, -1.414214f, -1.414214f, 0.0f, 1.414214f, 1.414214f, 1.414214f, 0.0f};
    Tensor* X  = cml_tensor_1d(x, 8);
    Tensor* H  = cml_hilbert(X);
    int ok     = H && H->shape[0] == 8 && H->shape[1] == 2;
    for (int i = 0; ok && i < 8; i++)
        ok &= fabsf(at(H, 2 * i) - x[i]) < 1e-3f && fabsf(at(H, 2 * i + 1) - im[i]) < 1e-3f;
    tensor_free(X);
    tensor_free(H);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== fft/signal helpers ===\n");
    TEST(fftfreq);
    TEST(detrend);
    TEST(hilbert);
    cml_cleanup();
    return TEST_SUMMARY();
}
