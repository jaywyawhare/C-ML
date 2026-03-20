#include "zoo/whisper.h"
#include "zoo/zoo.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/model_io.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

WhisperConfig whisper_tiny_config(void) {
    WhisperConfig cfg = {
        .size = WHISPER_TINY,
        .n_mels = 80,
        .n_audio_ctx = 1500,
        .n_audio_state = 384,
        .n_audio_head = 6,
        .n_audio_layer = 4,
        .n_text_ctx = 448,
        .n_text_state = 384,
        .n_text_head = 6,
        .n_text_layer = 4,
        .n_vocab = 51865,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

WhisperConfig whisper_base_config(void) {
    WhisperConfig cfg = {
        .size = WHISPER_BASE,
        .n_mels = 80,
        .n_audio_ctx = 1500,
        .n_audio_state = 512,
        .n_audio_head = 8,
        .n_audio_layer = 6,
        .n_text_ctx = 448,
        .n_text_state = 512,
        .n_text_head = 8,
        .n_text_layer = 6,
        .n_vocab = 51865,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

WhisperConfig whisper_small_config(void) {
    WhisperConfig cfg = {
        .size = WHISPER_SMALL,
        .n_mels = 80,
        .n_audio_ctx = 1500,
        .n_audio_state = 768,
        .n_audio_head = 12,
        .n_audio_layer = 12,
        .n_text_ctx = 448,
        .n_text_state = 768,
        .n_text_head = 12,
        .n_text_layer = 12,
        .n_vocab = 51865,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

WhisperConfig whisper_medium_config(void) {
    WhisperConfig cfg = {
        .size = WHISPER_MEDIUM,
        .n_mels = 80,
        .n_audio_ctx = 1500,
        .n_audio_state = 1024,
        .n_audio_head = 16,
        .n_audio_layer = 24,
        .n_text_ctx = 448,
        .n_text_state = 1024,
        .n_text_head = 16,
        .n_text_layer = 24,
        .n_vocab = 51865,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

WhisperConfig whisper_large_config(void) {
    WhisperConfig cfg = {
        .size = WHISPER_LARGE,
        .n_mels = 128,
        .n_audio_ctx = 1500,
        .n_audio_state = 1280,
        .n_audio_head = 20,
        .n_audio_layer = 32,
        .n_text_ctx = 448,
        .n_text_state = 1280,
        .n_text_head = 20,
        .n_text_layer = 32,
        .n_vocab = 51865,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

static void build_audio_encoder(Sequential* model, const WhisperConfig* cfg) {
    DType dt = cfg->dtype;
    DeviceType dev = cfg->device;
    int d = cfg->n_audio_state;
    int ff = d * 4;

    sequential_add(model, (Module*)nn_conv1d(cfg->n_mels, d, 3, 1, 1, 1, true, dt, dev));
    sequential_add(model, (Module*)nn_gelu(false));
    sequential_add(model, (Module*)nn_conv1d(d, d, 3, 2, 1, 1, true, dt, dev));
    sequential_add(model, (Module*)nn_gelu(false));

    for (int i = 0; i < cfg->n_audio_layer; i++) {
        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_multihead_attention(d, cfg->n_audio_head, 0.0f, dt, dev));
        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_linear(d, ff, dt, dev, true));
        sequential_add(model, (Module*)nn_gelu(false));
        sequential_add(model, (Module*)nn_linear(ff, d, dt, dev, true));
    }

    sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
}

static void build_text_decoder(Sequential* model, const WhisperConfig* cfg) {
    DType dt = cfg->dtype;
    DeviceType dev = cfg->device;
    int d = cfg->n_text_state;
    int ff = d * 4;

    sequential_add(model, (Module*)nn_embedding(cfg->n_vocab, d, -1, dt, dev));

    for (int i = 0; i < cfg->n_text_layer; i++) {
        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_multihead_attention(d, cfg->n_text_head, 0.0f, dt, dev));

        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_multihead_attention(d, cfg->n_text_head, 0.0f, dt, dev));

        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_linear(d, ff, dt, dev, true));
        sequential_add(model, (Module*)nn_gelu(false));
        sequential_add(model, (Module*)nn_linear(ff, d, dt, dev, true));
    }

    sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
    sequential_add(model, (Module*)nn_linear(d, cfg->n_vocab, dt, dev, false));
}

Module* cml_zoo_whisper(const WhisperConfig* config) {
    WhisperConfig cfg = config ? *config : whisper_base_config();

    Sequential* model = nn_sequential();

    build_audio_encoder(model, &cfg);
    build_text_decoder(model, &cfg);

    const char* size_str = "base";
    if (cfg.size == WHISPER_TINY) size_str = "tiny";
    else if (cfg.size == WHISPER_SMALL) size_str = "small";
    else if (cfg.size == WHISPER_MEDIUM) size_str = "medium";
    else if (cfg.size == WHISPER_LARGE) size_str = "large";

    LOG_INFO("Created Whisper-%s: %d encoder layers, %d decoder layers, dim=%d",
             size_str, cfg.n_audio_layer, cfg.n_text_layer, cfg.n_audio_state);
    return (Module*)model;
}
