#include "zoo/convnext.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>

ConvNeXtConfig cml_zoo_convnext_config_tiny(void) {
    ConvNeXtConfig cfg = {
        .dims = {96, 192, 384, 768},
        .depths = {3, 3, 9, 3}
    };
    return cfg;
}

ConvNeXtConfig cml_zoo_convnext_config_small(void) {
    ConvNeXtConfig cfg = {
        .dims = {96, 192, 384, 768},
        .depths = {3, 3, 27, 3}
    };
    return cfg;
}

ConvNeXtConfig cml_zoo_convnext_config_base(void) {
    ConvNeXtConfig cfg = {
        .dims = {128, 256, 512, 1024},
        .depths = {3, 3, 27, 3}
    };
    return cfg;
}

ConvNeXtConfig cml_zoo_convnext_config_large(void) {
    ConvNeXtConfig cfg = {
        .dims = {192, 384, 768, 1536},
        .depths = {3, 3, 27, 3}
    };
    return cfg;
}

static Conv2d* depthwise_conv7x7(int channels, DType dtype, DeviceType device) {
    Conv2d* conv = nn_conv2d(channels, channels, 7, 1, 3, 1, true, dtype, device);
    if (conv)
        conv->groups = channels;
    return conv;
}

typedef struct {
    Module base;
    Sequential* path;
} ConvNeXtBlock;

static Tensor* convnext_block_forward(Module* module, Tensor* input) {
    ConvNeXtBlock* block = (ConvNeXtBlock*)module;
    if (!block || !input)
        return NULL;

    Tensor* out = module_forward((Module*)block->path, input);
    if (!out)
        return NULL;

    return tensor_add(out, input);
}

static void convnext_block_free(Module* module) {
    ConvNeXtBlock* block = (ConvNeXtBlock*)module;
    if (!block)
        return;
    if (block->path)
        module_free((Module*)block->path);
    free(block);
}

static Module* create_convnext_block(int dim, DType dtype, DeviceType device) {
    ConvNeXtBlock* block = malloc(sizeof(ConvNeXtBlock));
    if (!block)
        return NULL;

    if (module_init((Module*)block, "ConvNeXtBlock", convnext_block_forward, convnext_block_free) != 0) {
        free(block);
        return NULL;
    }

    int hidden = dim * 4;

    block->path = nn_sequential();
    if (!block->path) {
        free(block);
        return NULL;
    }

    sequential_add(block->path, (Module*)depthwise_conv7x7(dim, dtype, device));
    sequential_add(block->path, (Module*)nn_layernorm(dim, 1e-6f, true, dtype, device));
    sequential_add(block->path, (Module*)nn_linear(dim, hidden, dtype, device, true));
    sequential_add(block->path, (Module*)nn_gelu(false));
    sequential_add(block->path, (Module*)nn_linear(hidden, dim, dtype, device, true));

    return (Module*)block;
}

static Module* create_downsample(int in_dim, int out_dim, DType dtype, DeviceType device) {
    Sequential* ds = nn_sequential();
    if (!ds)
        return NULL;

    sequential_add(ds, (Module*)nn_layernorm2d(in_dim, 1e-6f, true, dtype, device));
    sequential_add(ds, (Module*)nn_conv2d(in_dim, out_dim, 2, 2, 0, 1, true, dtype, device));

    return (Module*)ds;
}

Module* cml_zoo_convnext_create(const ConvNeXtConfig* cfg, int num_classes,
                                 DType dtype, DeviceType device) {
    if (!cfg)
        return NULL;
    if (num_classes <= 0)
        num_classes = 1000;

    Sequential* model = nn_sequential();
    if (!model)
        return NULL;

    sequential_add(model, (Module*)nn_conv2d(3, cfg->dims[0], 4, 4, 0, 1, true, dtype, device));
    sequential_add(model, (Module*)nn_layernorm2d(cfg->dims[0], 1e-6f, true, dtype, device));

    for (int stage = 0; stage < 4; stage++) {
        if (stage > 0)
            sequential_add(model, create_downsample(cfg->dims[stage - 1], cfg->dims[stage], dtype, device));

        for (int b = 0; b < cfg->depths[stage]; b++)
            sequential_add(model, create_convnext_block(cfg->dims[stage], dtype, device));
    }

    sequential_add(model, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_layernorm(cfg->dims[3], 1e-6f, true, dtype, device));
    sequential_add(model, (Module*)nn_linear(cfg->dims[3], num_classes, dtype, device, true));

    LOG_INFO("Created ConvNeXt (%d-%d-%d-%d, %d classes)",
             cfg->dims[0], cfg->dims[1], cfg->dims[2], cfg->dims[3], num_classes);
    return (Module*)model;
}
