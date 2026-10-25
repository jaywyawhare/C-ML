/* Signal window generators, checked against NumPy (np.hanning/hamming/blackman/bartlett) and
 * that cml_spectrogram still reuses the Hann window (de-duplicated). */
#include <math.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"
#include "cml.h"
#include "core/audio_features.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int win_eq(Tensor* w, const float* e, int n) {
    int ok = w && w->shape[0] == n;
    for (int i = 0; ok && i < n; i++)
        ok &= fabsf(at(w, i) - e[i]) < 1e-5f;
    return ok;
}

static int test_windows(void) {
    float hann[]  = {0.0f, 0.5f, 1.0f, 0.5f, 0.0f};
    float hamm[]  = {0.08f, 0.54f, 1.0f, 0.54f, 0.08f};
    float black[] = {0.0f, 0.34f, 1.0f, 0.34f, 0.0f};
    float bart[]  = {0.0f, 0.5f, 1.0f, 0.5f, 0.0f};
    Tensor* H     = cml_hann_window(5, &CFG);
    Tensor* M     = cml_hamming_window(5, &CFG);
    Tensor* B     = cml_blackman_window(5, &CFG);
    Tensor* T     = cml_bartlett_window(5, &CFG);
    int ok = win_eq(H, hann, 5) && win_eq(M, hamm, 5) && win_eq(B, black, 5) && win_eq(T, bart, 5);
    tensor_free(H);
    tensor_free(M);
    tensor_free(B);
    tensor_free(T);
    return ok;
}

static int test_spectrogram_still_works(void) {
    /* A pure tone still concentrates energy at its bin after the Hann-window refactor. */
    const int SR = 64, N_FFT = 16, N = 64;
    float sig[64];
    int k0 = 4;
    for (int i = 0; i < N; i++)
        sig[i] = sinf(2.0f * (float)M_PI * k0 * i / N_FFT);
    (void)SR;
    int frames = 0, bins = 0;
    float* spec = cml_spectrogram(sig, N, N_FFT, 4, 1, &frames, &bins);
    int ok      = spec && bins == N_FFT / 2 + 1;
    if (ok) {
        float peak = spec[k0];
        for (int b = 0; b < bins; b++)
            if (b != k0 && spec[b] > peak)
                ok = 0;
    }
    cml_free(spec);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== signal windows ===\n");
    TEST(windows);
    TEST(spectrogram_still_works);
    cml_cleanup();
    return TEST_SUMMARY();
}
