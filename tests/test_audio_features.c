/* Audio front-end: a pure tone concentrates STFT energy in its expected bin, the mel
 * filterbank is non-negative with strictly increasing band centers, and the mel-spectrogram
 * and MFCC pipelines produce the expected shapes and finite values. */
#include <math.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"
#include "core/audio_features.h"
#include "test_harness.h"

#define N_FFT 64
#define SR 64

static int argmax(const float* row, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (row[i] > row[best])
            best = i;
    return best;
}

static int test_spectrogram_peak(void) {
    float sig[128];
    int k0 = 8; /* bin at f0 = k0 * SR / N_FFT = 8 Hz */
    for (int i = 0; i < 128; i++)
        sig[i] = cosf(2.0f * (float)M_PI * k0 * i / N_FFT);
    int frames = 0, bins = 0;
    float* spec = cml_spectrogram(sig, 128, N_FFT, N_FFT, 2, &frames, &bins);
    int ok      = spec && frames == 2 && bins == N_FFT / 2 + 1 && argmax(spec, bins) == k0;
    cml_free(spec);
    return ok;
}

static int test_filterbank(void) {
    int n_mels = 4, nb = N_FFT / 2 + 1;
    float* fb     = cml_mel_filterbank(n_mels, N_FFT, SR, 0.0f, SR / 2.0f);
    int ok        = fb != NULL;
    int prev_peak = -1;
    for (int m = 0; ok && m < n_mels; m++) {
        for (int b = 0; b < nb; b++)
            ok &= fb[m * nb + b] >= 0.0f;
        int peak = argmax(fb + m * nb, nb);
        ok &= peak > prev_peak; /* band centers increase with mel index */
        prev_peak = peak;
    }
    cml_free(fb);
    return ok;
}

static int test_melspectrogram(void) {
    float sig[128];
    for (int i = 0; i < 128; i++)
        sig[i] = cosf(2.0f * (float)M_PI * 8 * i / N_FFT);
    int frames = 0, mels = 0;
    float* mel = cml_melspectrogram(sig, 128, SR, N_FFT, N_FFT, 8, &frames, &mels);
    int ok     = mel && frames == 2 && mels == 8;
    for (int i = 0; ok && i < frames * mels; i++)
        ok &= mel[i] >= 0.0f && isfinite(mel[i]);
    cml_free(mel);
    return ok;
}

static int test_mfcc(void) {
    float sig[128];
    for (int i = 0; i < 128; i++)
        sig[i] = cosf(2.0f * (float)M_PI * 8 * i / N_FFT) + 0.3f * sinf(0.7f * i);
    int frames = 0, nc = 0;
    float* a = cml_mfcc(sig, 128, SR, N_FFT, N_FFT, 16, 13, &frames, &nc);
    int ok   = a && frames == 2 && nc == 13;
    for (int i = 0; ok && i < frames * nc; i++)
        ok &= isfinite(a[i]);
    /* determinism */
    float* b = cml_mfcc(sig, 128, SR, N_FFT, N_FFT, 16, 13, &frames, &nc);
    for (int i = 0; ok && b && i < frames * nc; i++)
        ok &= fabsf(a[i] - b[i]) < 1e-6f;
    cml_free(a);
    cml_free(b);
    return ok;
}

static int test_rejects_bad(void) {
    int f = 0, bq = 0;
    float sig[8] = {0};
    return cml_spectrogram(sig, 8, 63, 4, 2, &f, &bq) == NULL &&      /* n_fft not power of two */
           cml_mfcc(sig, 8, SR, N_FFT, N_FFT, 4, 8, &f, &bq) == NULL; /* n_mfcc>n_mels */
}

int main(void) {
    printf("=== audio features: STFT / mel / MFCC ===\n");
    TEST(spectrogram_peak);
    TEST(filterbank);
    TEST(melspectrogram);
    TEST(mfcc);
    TEST(rejects_bad);
    return TEST_SUMMARY();
}
