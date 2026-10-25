#ifndef CML_CORE_AUDIO_FEATURES_H
#define CML_CORE_AUDIO_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Audio feature front-end built on cml_fft_1d: Hann-windowed STFT power/magnitude
 * spectrogram, mel filterbank, mel-spectrogram and MFCC. n_fft must be a power of two.
 * Each function returns a freshly allocated row-major buffer (caller frees with cml_free). */

/**
 * Hann-windowed STFT spectrogram, [frames, n_fft/2+1] row-major. power=1 gives magnitude,
 * power=2 gives power. Sets frames and bins. NULL on bad arguments.
 * @param signal (const float*)
 * @param n (int)
 * @param n_fft (int)
 * @param hop (int)
 * @param power (int)
 * @param frames (int*)
 * @param bins (int*)
 * @return (float*)
 */
float* cml_spectrogram(const float* signal, int n, int n_fft, int hop, int power, int* frames,
                       int* bins);

/**
 * Triangular mel filterbank weights, [n_mels, n_fft/2+1] row-major. NULL on bad arguments.
 * @param n_mels (int)
 * @param n_fft (int)
 * @param sample_rate (int)
 * @param fmin (float)
 * @param fmax (float)
 * @return (float*)
 */
float* cml_mel_filterbank(int n_mels, int n_fft, int sample_rate, float fmin, float fmax);

/**
 * Mel-spectrogram, [frames, n_mels] row-major (power spectrogram projected through the mel
 * filterbank). Sets frames and out_mels. NULL on bad arguments.
 * @param signal (const float*)
 * @param n (int)
 * @param sample_rate (int)
 * @param n_fft (int)
 * @param hop (int)
 * @param n_mels (int)
 * @param frames (int*)
 * @param out_mels (int*)
 * @return (float*)
 */
float* cml_melspectrogram(const float* signal, int n, int sample_rate, int n_fft, int hop,
                          int n_mels, int* frames, int* out_mels);

/**
 * MFCCs, [frames, n_mfcc] row-major (DCT-II of the log mel-spectrogram). Sets frames and
 * out_coeffs. NULL on bad arguments.
 * @param signal (const float*)
 * @param n (int)
 * @param sample_rate (int)
 * @param n_fft (int)
 * @param hop (int)
 * @param n_mels (int)
 * @param n_mfcc (int)
 * @param frames (int*)
 * @param out_coeffs (int*)
 * @return (float*)
 */
float* cml_mfcc(const float* signal, int n, int sample_rate, int n_fft, int hop, int n_mels,
                int n_mfcc, int* frames, int* out_coeffs);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_AUDIO_FEATURES_H */
