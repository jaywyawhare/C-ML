#include "zoo/unet.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>

CMLUNetConfig cml_zoo_unet_config_default(void) {
    return (CMLUNetConfig){
        .in_channels  = 1,
        .num_classes  = 2,
        .depth        = 4,
        .base_filters = 64
    };
}

/* Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU */
static Sequential* create_conv_block(int in_ch, int out_ch, DType dtype, DeviceType device) {
    Sequential* block = nn_sequential();
    sequential_add(block, (Module*)nn_conv2d(in_ch, out_ch, 3, 1, 1, 1, false, dtype, device));
    sequential_add(block, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block, (Module*)nn_relu(false));
    sequential_add(block, (Module*)nn_conv2d(out_ch, out_ch, 3, 1, 1, 1, false, dtype, device));
    sequential_add(block, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block, (Module*)nn_relu(false));
    return block;
}

#define UNET_MAX_DEPTH 8

typedef struct {
    Module base;
    Sequential* enc_blocks[UNET_MAX_DEPTH];
    MaxPool2d* pools[UNET_MAX_DEPTH];
    Sequential* bottleneck;
    Upsample* ups[UNET_MAX_DEPTH];
    Sequential* dec_blocks[UNET_MAX_DEPTH];
    Conv2d* final_conv;
    int depth;
} UNetModel;

static Tensor* unet_forward(Module* module, Tensor* input) {
    UNetModel* unet = (UNetModel*)module;
    if (!unet || !input) return NULL;

    Tensor* skips[UNET_MAX_DEPTH];
    Tensor* x = input;

    for (int i = 0; i < unet->depth; i++) {
        x = module_forward((Module*)unet->enc_blocks[i], x);
        if (!x) return NULL;
        skips[i] = x;
        x = module_forward((Module*)unet->pools[i], x);
        if (!x) return NULL;
    }

    x = module_forward((Module*)unet->bottleneck, x);
    if (!x) return NULL;

    for (int i = unet->depth - 1; i >= 0; i--) {
        x = module_forward((Module*)unet->ups[i], x);
        if (!x) return NULL;

        Tensor* cat_tensors[] = {x, skips[i]};
        x = tensor_concat(cat_tensors, 2, 1);
        if (!x) return NULL;

        x = module_forward((Module*)unet->dec_blocks[i], x);
        if (!x) return NULL;
    }

    return module_forward((Module*)unet->final_conv, x);
}

static void unet_free(Module* module) {
    UNetModel* unet = (UNetModel*)module;
    if (!unet) return;

    for (int i = 0; i < unet->depth; i++) {
        if (unet->enc_blocks[i]) module_free((Module*)unet->enc_blocks[i]);
        if (unet->pools[i]) module_free((Module*)unet->pools[i]);
        if (unet->ups[i]) module_free((Module*)unet->ups[i]);
        if (unet->dec_blocks[i]) module_free((Module*)unet->dec_blocks[i]);
    }
    if (unet->bottleneck) module_free((Module*)unet->bottleneck);
    if (unet->final_conv) module_free((Module*)unet->final_conv);
    free(unet);
}

Module* cml_zoo_unet_create(CMLUNetConfig* config, DType dtype, DeviceType device) {
    if (!config)
        return NULL;

    int depth = config->depth;
    if (depth > UNET_MAX_DEPTH)
        depth = UNET_MAX_DEPTH;

    UNetModel* unet = malloc(sizeof(UNetModel));
    if (!unet)
        return NULL;

    if (module_init((Module*)unet, "UNet", unet_forward, unet_free) != 0) {
        free(unet);
        return NULL;
    }

    unet->depth = depth;

    int filters = config->base_filters;
    int in_ch = config->in_channels;

    for (int i = 0; i < depth; i++) {
        unet->enc_blocks[i] = create_conv_block(in_ch, filters, dtype, device);
        unet->pools[i] = nn_maxpool2d(2, 2, 0, 1, false);
        in_ch = filters;
        filters *= 2;
    }

    unet->bottleneck = create_conv_block(in_ch, filters, dtype, device);

    for (int i = depth - 1; i >= 0; i--) {
        int dec_out = filters / 2;
        unet->ups[i] = nn_upsample(2.0f, NULL, 0, UPSAMPLE_BILINEAR, false);
        unet->dec_blocks[i] = create_conv_block(filters + dec_out, dec_out, dtype, device);
        filters = dec_out;
    }

    unet->final_conv = nn_conv2d(config->base_filters, config->num_classes,
                                  1, 1, 0, 1, true, dtype, device);

    LOG_INFO("Created U-Net (depth=%d, base_filters=%d, in=%d, classes=%d)",
             depth, config->base_filters, config->in_channels, config->num_classes);
    return (Module*)unet;
}
