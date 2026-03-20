#include "zoo/unet3d.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>

CMLUNet3DConfig cml_zoo_unet3d_config_default(void) {
    return (CMLUNet3DConfig){
        .in_channels  = 1,
        .num_classes  = 3,
        .depth        = 4,
        .base_filters = 32
    };
}

static Sequential* conv3d_instnorm_relu(int in_ch, int out_ch, DType dtype, DeviceType device) {
    Sequential* block = nn_sequential();
    sequential_add(block, (Module*)nn_conv3d(in_ch, out_ch, 3, 1, 1, 1, false, dtype, device));
    sequential_add(block, (Module*)nn_batchnorm3d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block, (Module*)nn_relu(false));
    return block;
}

static Sequential* double_conv3d_block(int in_ch, int out_ch, DType dtype, DeviceType device) {
    Sequential* block = nn_sequential();
    sequential_add(block, (Module*)conv3d_instnorm_relu(in_ch, out_ch, dtype, device));
    sequential_add(block, (Module*)conv3d_instnorm_relu(out_ch, out_ch, dtype, device));
    return block;
}

#define UNET3D_MAX_DEPTH 8

typedef struct {
    Module base;
    Sequential* enc_blocks[UNET3D_MAX_DEPTH];
    MaxPool3d*  pools[UNET3D_MAX_DEPTH];
    Sequential* bottleneck;
    Upsample*   ups[UNET3D_MAX_DEPTH];
    Sequential* dec_blocks[UNET3D_MAX_DEPTH];
    Conv3d*     final_conv;
    int depth;
} UNet3DModel;

static Tensor* unet3d_forward(Module* module, Tensor* input) {
    UNet3DModel* net = (UNet3DModel*)module;
    if (!net || !input) return NULL;

    Tensor* skips[UNET3D_MAX_DEPTH];
    Tensor* x = input;

    for (int i = 0; i < net->depth; i++) {
        x = module_forward((Module*)net->enc_blocks[i], x);
        if (!x) return NULL;
        skips[i] = x;
        x = module_forward((Module*)net->pools[i], x);
        if (!x) return NULL;
    }

    x = module_forward((Module*)net->bottleneck, x);
    if (!x) return NULL;

    for (int i = net->depth - 1; i >= 0; i--) {
        x = module_forward((Module*)net->ups[i], x);
        if (!x) return NULL;

        Tensor* cat_tensors[] = {x, skips[i]};
        x = tensor_concat(cat_tensors, 2, 1);
        if (!x) return NULL;

        x = module_forward((Module*)net->dec_blocks[i], x);
        if (!x) return NULL;
    }

    return module_forward((Module*)net->final_conv, x);
}

static void unet3d_free(Module* module) {
    UNet3DModel* net = (UNet3DModel*)module;
    if (!net) return;

    for (int i = 0; i < net->depth; i++) {
        if (net->enc_blocks[i]) module_free((Module*)net->enc_blocks[i]);
        if (net->pools[i]) module_free((Module*)net->pools[i]);
        if (net->ups[i]) module_free((Module*)net->ups[i]);
        if (net->dec_blocks[i]) module_free((Module*)net->dec_blocks[i]);
    }
    if (net->bottleneck) module_free((Module*)net->bottleneck);
    if (net->final_conv) module_free((Module*)net->final_conv);
    free(net);
}

Module* cml_zoo_unet3d_create(const CMLUNet3DConfig* config, DType dtype, DeviceType device) {
    if (!config) return NULL;

    int depth = config->depth;
    if (depth > UNET3D_MAX_DEPTH)
        depth = UNET3D_MAX_DEPTH;

    UNet3DModel* net = calloc(1, sizeof(UNet3DModel));
    if (!net) return NULL;

    if (module_init((Module*)net, "UNet3D", unet3d_forward, unet3d_free) != 0) {
        free(net);
        return NULL;
    }

    net->depth = depth;
    int filters = config->base_filters;
    int in_ch = config->in_channels;

    for (int i = 0; i < depth; i++) {
        net->enc_blocks[i] = double_conv3d_block(in_ch, filters, dtype, device);
        net->pools[i] = nn_maxpool3d(2, 2, 0, 1, false);
        in_ch = filters;
        filters *= 2;
    }

    net->bottleneck = double_conv3d_block(in_ch, filters, dtype, device);

    for (int i = depth - 1; i >= 0; i--) {
        int dec_out = filters / 2;
        int out_sz[] = {0, 0, 0};
        net->ups[i] = nn_upsample(2.0f, out_sz, 3, UPSAMPLE_NEAREST, false);
        net->dec_blocks[i] = double_conv3d_block(filters + dec_out, dec_out, dtype, device);
        filters = dec_out;
    }

    net->final_conv = nn_conv3d(config->base_filters, config->num_classes,
                                1, 1, 0, 1, true, dtype, device);

    LOG_INFO("Created UNet3D (depth=%d, base_filters=%d, in=%d, classes=%d)",
             depth, config->base_filters, config->in_channels, config->num_classes);
    return (Module*)net;
}
