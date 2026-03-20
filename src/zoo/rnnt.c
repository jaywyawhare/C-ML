#include "zoo/rnnt.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

CMLRNNTConfig cml_zoo_rnnt_config_default(void) {
    return (CMLRNNTConfig){
        .input_features    = 80,
        .encoder_layers    = 5,
        .encoder_dim       = 1024,
        .pred_layers       = 2,
        .pred_dim          = 320,
        .joint_dim         = 512,
        .vocab_size        = 29,
        .subsample_stride  = 2,
        .subsample_kernel  = 3,
    };
}

static Tensor* rnnt_forward(Module* module, Tensor* input) {
    (void)module;
    (void)input;
    return NULL;
}

static void rnnt_free(Module* module) {
    CMLRNNT* net = (CMLRNNT*)module;
    if (!net) return;

    if (net->subsample_conv) module_free(net->subsample_conv);

    for (int i = 0; i < net->num_enc_layers; i++) {
        if (net->encoder_lstms[i]) module_free(net->encoder_lstms[i]);
        if (net->encoder_lnorms[i]) module_free(net->encoder_lnorms[i]);
    }
    free(net->encoder_lstms);
    free(net->encoder_lnorms);

    if (net->pred_embedding) module_free(net->pred_embedding);
    for (int i = 0; i < net->num_pred_layers; i++) {
        if (net->pred_lstms[i]) module_free(net->pred_lstms[i]);
        if (net->pred_lnorms[i]) module_free(net->pred_lnorms[i]);
    }
    free(net->pred_lstms);
    free(net->pred_lnorms);

    if (net->joint_linear1) module_free(net->joint_linear1);
    if (net->joint_relu) module_free(net->joint_relu);
    if (net->joint_linear2) module_free(net->joint_linear2);

    free(net);
}

Tensor* cml_rnnt_encode(Module* module, Tensor* audio_features) {
    CMLRNNT* net = (CMLRNNT*)module;
    if (!net || !audio_features) return NULL;

    Tensor* x = module_forward(net->subsample_conv, audio_features);
    if (!x) return NULL;

    for (int i = 0; i < net->num_enc_layers; i++) {
        LSTM* lstm = (LSTM*)net->encoder_lstms[i];
        Tensor* out = NULL;
        Tensor* h_n = NULL;
        Tensor* c_n = NULL;
        lstm_forward(lstm, x, NULL, NULL, &out, &h_n, &c_n);
        if (!out) return NULL;
        x = module_forward(net->encoder_lnorms[i], out);
        if (!x) return NULL;
    }

    return x;
}

Tensor* cml_rnnt_predict(Module* module, Tensor* prev_tokens) {
    CMLRNNT* net = (CMLRNNT*)module;
    if (!net || !prev_tokens) return NULL;

    Tensor* x = module_forward(net->pred_embedding, prev_tokens);
    if (!x) return NULL;

    for (int i = 0; i < net->num_pred_layers; i++) {
        LSTM* lstm = (LSTM*)net->pred_lstms[i];
        Tensor* out = NULL;
        Tensor* h_n = NULL;
        Tensor* c_n = NULL;
        lstm_forward(lstm, x, NULL, NULL, &out, &h_n, &c_n);
        if (!out) return NULL;
        x = module_forward(net->pred_lnorms[i], out);
        if (!x) return NULL;
    }

    return x;
}

Tensor* cml_rnnt_joint(Module* module, Tensor* enc_out, Tensor* pred_out) {
    CMLRNNT* net = (CMLRNNT*)module;
    if (!net || !enc_out || !pred_out) return NULL;

    Tensor* cat_tensors[] = {enc_out, pred_out};
    Tensor* combined = tensor_concat(cat_tensors, 2, -1);
    if (!combined) return NULL;

    Tensor* x = module_forward(net->joint_linear1, combined);
    if (!x) return NULL;
    x = module_forward(net->joint_relu, x);
    if (!x) return NULL;
    return module_forward(net->joint_linear2, x);
}

Module* cml_zoo_rnnt_create(const CMLRNNTConfig* config, DType dtype, DeviceType device) {
    if (!config) return NULL;

    CMLRNNT* net = calloc(1, sizeof(CMLRNNT));
    if (!net) return NULL;

    if (module_init((Module*)net, "RNN-T", rnnt_forward, rnnt_free) != 0) {
        free(net);
        return NULL;
    }

    net->config = *config;
    net->dtype = dtype;
    net->device = device;

    int pad = (config->subsample_kernel - 1) / 2;
    net->subsample_conv = (Module*)nn_conv1d(
        config->input_features, config->encoder_dim,
        config->subsample_kernel, config->subsample_stride,
        pad, 1, true, dtype, device);

    net->num_enc_layers = config->encoder_layers;
    net->encoder_lstms = calloc(config->encoder_layers, sizeof(Module*));
    net->encoder_lnorms = calloc(config->encoder_layers, sizeof(Module*));

    for (int i = 0; i < config->encoder_layers; i++) {
        int in_sz = config->encoder_dim;
        net->encoder_lstms[i] = (Module*)nn_lstm(
            in_sz, config->encoder_dim, 1,
            false, true, 0.0f, true, dtype, device);
        net->encoder_lnorms[i] = (Module*)nn_layernorm(
            config->encoder_dim, 1e-5f, true, dtype, device);
    }

    net->pred_embedding = (Module*)nn_embedding(
        config->vocab_size, config->pred_dim, -1, dtype, device);

    net->num_pred_layers = config->pred_layers;
    net->pred_lstms = calloc(config->pred_layers, sizeof(Module*));
    net->pred_lnorms = calloc(config->pred_layers, sizeof(Module*));

    for (int i = 0; i < config->pred_layers; i++) {
        int in_sz = config->pred_dim;
        net->pred_lstms[i] = (Module*)nn_lstm(
            in_sz, config->pred_dim, 1,
            false, true, 0.0f, true, dtype, device);
        net->pred_lnorms[i] = (Module*)nn_layernorm(
            config->pred_dim, 1e-5f, true, dtype, device);
    }

    net->joint_linear1 = (Module*)nn_linear(
        config->encoder_dim + config->pred_dim,
        config->joint_dim, dtype, device, true);
    net->joint_relu = (Module*)nn_relu(false);
    net->joint_linear2 = (Module*)nn_linear(
        config->joint_dim, config->vocab_size, dtype, device, true);

    LOG_INFO("Created RNN-T (enc=%dx%d, pred=%dx%d, joint=%d, vocab=%d)",
             config->encoder_layers, config->encoder_dim,
             config->pred_layers, config->pred_dim,
             config->joint_dim, config->vocab_size);
    return (Module*)net;
}
