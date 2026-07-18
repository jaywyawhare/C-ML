#ifndef CML_ZOO_WHISPER_H
#define CML_ZOO_WHISPER_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WHISPER_TINY,
    WHISPER_BASE,
    WHISPER_SMALL,
    WHISPER_MEDIUM,
    WHISPER_LARGE
} WhisperSize;

typedef struct {
    WhisperSize size;
    int n_mels;
    int n_audio_ctx;
    int n_audio_state;
    int n_audio_head;
    int n_audio_layer;
    int n_text_ctx;
    int n_text_state;
    int n_text_head;
    int n_text_layer;
    int n_vocab;
    DType dtype;
    DeviceType device;
} WhisperConfig;

WhisperConfig whisper_tiny_config(void);
WhisperConfig whisper_base_config(void);
WhisperConfig whisper_small_config(void);
WhisperConfig whisper_medium_config(void);
WhisperConfig whisper_large_config(void);

Module* cml_zoo_whisper(const WhisperConfig* config);

/* Log-mel spectrogram frontend: 16 kHz mono audio -> [n_mels, frames] tensor
 * (Whisper convention: n_fft=400, hop=160, Hann window, log10 clamped to
 * max-8, scaled (x+4)/4). Feed the result to the audio encoder. */
Tensor* cml_whisper_log_mel(const float* audio, int num_samples, int n_mels);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_WHISPER_H */
