#include "zoo/whisper.h"
#include "zoo/zoo.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/model_io.h"
#include "cml.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Log-mel spectrogram frontend --------------------------------------
 * Whisper convention: 16 kHz audio, n_fft=400, hop=160, Hann window, mel
 * filters over 0-8 kHz, log10 clamped to (max - 8), scaled to (x+4)/4. */

#define WHISPER_N_FFT       400
#define WHISPER_HOP         160
#define WHISPER_SAMPLE_RATE 16000

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

Tensor* cml_whisper_log_mel(const float* audio, int num_samples, int n_mels) {
    if (!audio || num_samples < WHISPER_N_FFT || n_mels <= 0) {
        LOG_ERROR("cml_whisper_log_mel: need at least %d samples", WHISPER_N_FFT);
        return NULL;
    }

    int n_fft = WHISPER_N_FFT;
    int n_bins = n_fft / 2 + 1;
    int frames = 1 + (num_samples - n_fft) / WHISPER_HOP;

    /* Hann window */
    float* window = (float*)cml_malloc((size_t)n_fft * sizeof(float));
    float* re = (float*)cml_malloc((size_t)n_fft * sizeof(float));
    float* im = (float*)cml_malloc((size_t)n_fft * sizeof(float));
    float* power = (float*)cml_malloc((size_t)frames * (size_t)n_bins * sizeof(float));
    if (!window || !re || !im || !power) {
        cml_free(window); cml_free(re); cml_free(im); cml_free(power);
        return NULL;
    }
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)n_fft));

    for (int t = 0; t < frames; t++) {
        const float* seg = audio + (size_t)t * WHISPER_HOP;
        for (int i = 0; i < n_fft; i++) {
            re[i] = seg[i] * window[i];
            im[i] = 0.0f;
        }
        if (cml_fft_1d(re, im, n_fft, 0) != 0) {
            cml_free(window); cml_free(re); cml_free(im); cml_free(power);
            return NULL;
        }
        for (int b = 0; b < n_bins; b++)
            power[(size_t)t * n_bins + b] = re[b] * re[b] + im[b] * im[b];
    }
    cml_free(window); cml_free(re); cml_free(im);

    /* Triangular mel filterbank over 0..8 kHz */
    float mel_lo = hz_to_mel(0.0f);
    float mel_hi = hz_to_mel((float)WHISPER_SAMPLE_RATE / 2.0f);
    float* centers = (float*)cml_malloc((size_t)(n_mels + 2) * sizeof(float));
    if (!centers) { cml_free(power); return NULL; }
    for (int m = 0; m < n_mels + 2; m++) {
        float mel = mel_lo + (mel_hi - mel_lo) * (float)m / (float)(n_mels + 1);
        centers[m] = mel_to_hz(mel) * (float)n_fft / (float)WHISPER_SAMPLE_RATE;
    }

    int mel_shape[2] = {n_mels, frames};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* mel = tensor_empty(mel_shape, 2, &cfg);
    if (!mel) { cml_free(power); cml_free(centers); return NULL; }
    float* mdata = (float*)tensor_data_ptr(mel);

    float log_max = -1e30f;
    for (int m = 0; m < n_mels; m++) {
        float f0 = centers[m], f1 = centers[m + 1], f2 = centers[m + 2];
        for (int t = 0; t < frames; t++) {
            float acc = 0.0f;
            int b0 = (int)ceilf(f0), b2 = (int)floorf(f2);
            if (b0 < 0) b0 = 0;
            if (b2 > n_bins - 1) b2 = n_bins - 1;
            for (int b = b0; b <= b2; b++) {
                float w = 0.0f;
                if ((float)b <= f1 && f1 > f0)
                    w = ((float)b - f0) / (f1 - f0);
                else if (f2 > f1)
                    w = (f2 - (float)b) / (f2 - f1);
                if (w > 0.0f)
                    acc += w * power[(size_t)t * n_bins + b];
            }
            float v = log10f(acc > 1e-10f ? acc : 1e-10f);
            mdata[(size_t)m * frames + t] = v;
            if (v > log_max) log_max = v;
        }
    }
    cml_free(power);
    cml_free(centers);

    /* Whisper normalization: clamp to (max - 8), scale to roughly [-1, 1]. */
    float floor_v = log_max - 8.0f;
    for (size_t i = 0; i < (size_t)n_mels * (size_t)frames; i++) {
        float v = mdata[i] < floor_v ? floor_v : mdata[i];
        mdata[i] = (v + 4.0f) / 4.0f;
    }

    mel->is_executed = true;
    return mel;
}

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
