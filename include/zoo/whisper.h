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
    WHISPER_SMALL
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

Module* cml_zoo_whisper(const WhisperConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_WHISPER_H */
