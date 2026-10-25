/* Audio feature extraction on top of cml_fft_1d: STFT spectrogram, mel filterbank,
 * mel-spectrogram and MFCC. Each stage reuses the previous one. */
#include "core/audio_features.h"
#include <math.h>
#include <stdlib.h>
#include "alloc/cml_allocator.h"
#include "cml.h"
#include "core/logging.h"

/** Whether a positive integer is a power of two.
 * @param x (int)
 * @return (int)
 */
static int is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

/** Convert a frequency in Hz to the mel scale (HTK formula).
 * @param hz (float)
 * @return (float)
 */
static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }

/** Convert a mel value back to Hz (inverse HTK formula).
 * @param mel (float)
 * @return (float)
 */
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

float* cml_spectrogram(const float* signal, int n, int n_fft, int hop, int power, int* frames,
                       int* bins) {
    if (!signal || !is_pow2(n_fft) || hop <= 0 || n < n_fft || (power != 1 && power != 2)) {
        LOG_ERROR("cml_spectrogram: bad arguments (n_fft must be a power of two, n>=n_fft)");
        return NULL;
    }
    int nf     = 1 + (n - n_fft) / hop;
    int nb     = n_fft / 2 + 1;
    float* w   = cml_malloc((size_t)n_fft * sizeof(float));
    float* re  = cml_malloc((size_t)n_fft * sizeof(float));
    float* im  = cml_malloc((size_t)n_fft * sizeof(float));
    float* out = cml_malloc((size_t)nf * nb * sizeof(float));
    if (!w || !re || !im || !out) {
        cml_free(w);
        cml_free(re);
        cml_free(im);
        cml_free(out);
        return NULL;
    }
    for (int i = 0; i < n_fft; i++)
        w[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (n_fft - 1));
    for (int f = 0; f < nf; f++) {
        const float* seg = signal + (size_t)f * hop;
        for (int i = 0; i < n_fft; i++) {
            re[i] = seg[i] * w[i];
            im[i] = 0.0f;
        }
        cml_fft_1d(re, im, n_fft, 0);
        for (int b = 0; b < nb; b++) {
            float p                 = re[b] * re[b] + im[b] * im[b];
            out[(size_t)f * nb + b] = power == 2 ? p : sqrtf(p);
        }
    }
    cml_free(w);
    cml_free(re);
    cml_free(im);
    if (frames)
        *frames = nf;
    if (bins)
        *bins = nb;
    return out;
}

float* cml_mel_filterbank(int n_mels, int n_fft, int sample_rate, float fmin, float fmax) {
    if (n_mels <= 0 || !is_pow2(n_fft) || sample_rate <= 0 || fmax <= fmin) {
        LOG_ERROR("cml_mel_filterbank: bad arguments");
        return NULL;
    }
    int nb     = n_fft / 2 + 1;
    float* fb  = cml_calloc((size_t)n_mels * nb, sizeof(float));
    float* pts = cml_malloc((size_t)(n_mels + 2) * sizeof(float)); /* band edges in Hz */
    if (!fb || !pts) {
        cml_free(fb);
        cml_free(pts);
        return NULL;
    }
    float mmin = hz_to_mel(fmin), mmax = hz_to_mel(fmax);
    for (int i = 0; i < n_mels + 2; i++)
        pts[i] = mel_to_hz(mmin + (mmax - mmin) * i / (n_mels + 1));
    float bin_hz = (float)sample_rate / n_fft;
    for (int m = 0; m < n_mels; m++) {
        float left = pts[m], center = pts[m + 1], right = pts[m + 2];
        for (int b = 0; b < nb; b++) {
            float hz  = b * bin_hz;
            float wgt = 0.0f;
            if (hz >= left && hz <= center && center > left)
                wgt = (hz - left) / (center - left);
            else if (hz > center && hz <= right && right > center)
                wgt = (right - hz) / (right - center);
            fb[(size_t)m * nb + b] = wgt;
        }
    }
    cml_free(pts);
    return fb;
}

float* cml_melspectrogram(const float* signal, int n, int sample_rate, int n_fft, int hop,
                          int n_mels, int* frames, int* out_mels) {
    int nf = 0, nb = 0;
    float* spec = cml_spectrogram(signal, n, n_fft, hop, 2, &nf, &nb);
    if (!spec)
        return NULL;
    float* fb  = cml_mel_filterbank(n_mels, n_fft, sample_rate, 0.0f, sample_rate / 2.0f);
    float* out = cml_malloc((size_t)nf * n_mels * sizeof(float));
    if (!fb || !out) {
        cml_free(spec);
        cml_free(fb);
        cml_free(out);
        return NULL;
    }
    for (int f = 0; f < nf; f++)
        for (int m = 0; m < n_mels; m++) {
            float s = 0.0f;
            for (int b = 0; b < nb; b++)
                s += spec[(size_t)f * nb + b] * fb[(size_t)m * nb + b];
            out[(size_t)f * n_mels + m] = s;
        }
    cml_free(spec);
    cml_free(fb);
    if (frames)
        *frames = nf;
    if (out_mels)
        *out_mels = n_mels;
    return out;
}

float* cml_mfcc(const float* signal, int n, int sample_rate, int n_fft, int hop, int n_mels,
                int n_mfcc, int* frames, int* out_coeffs) {
    if (n_mfcc <= 0 || n_mfcc > n_mels) {
        LOG_ERROR("cml_mfcc: need 0 < n_mfcc <= n_mels");
        return NULL;
    }
    int nf = 0, nm = 0;
    float* mel = cml_melspectrogram(signal, n, sample_rate, n_fft, hop, n_mels, &nf, &nm);
    if (!mel)
        return NULL;
    float* out = cml_malloc((size_t)nf * n_mfcc * sizeof(float));
    if (!out) {
        cml_free(mel);
        return NULL;
    }
    for (int f = 0; f < nf; f++) {
        float* row = mel + (size_t)f * n_mels;
        for (int m = 0; m < n_mels; m++)
            row[m] = logf(row[m] + 1e-10f);
        for (int k = 0; k < n_mfcc; k++) {
            float s = 0.0f;
            for (int m = 0; m < n_mels; m++)
                s += row[m] * cosf((float)M_PI / n_mels * (m + 0.5f) * k);
            out[(size_t)f * n_mfcc + k] = s;
        }
    }
    cml_free(mel);
    if (frames)
        *frames = nf;
    if (out_coeffs)
        *out_coeffs = n_mfcc;
    return out;
}
