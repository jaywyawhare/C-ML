#include "zoo/efficientnet.h"
#include "zoo/zoo.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/model_io.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

EfficientNetConfig efficientnet_b0_config(int num_classes) {
    EfficientNetConfig cfg = {
        .num_classes = num_classes > 0 ? num_classes : 1000,
        .dropout_rate = 0.2f,
        .width_mult = 1.0f,
        .depth_mult = 1.0f,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

static const EfficientNetBlockConfig b0_blocks[] = {
    {1, 32,  16,  3, 1, 1, 4},
    {6, 16,  24,  3, 2, 2, 4},
    {6, 24,  40,  5, 2, 2, 4},
    {6, 40,  80,  3, 2, 3, 4},
    {6, 80,  112, 5, 1, 3, 4},
    {6, 112, 192, 5, 2, 4, 4},
    {6, 192, 320, 3, 1, 1, 4},
};

static void add_se_block(Sequential* seq, int channels, int se_channels,
                          DType dtype, DeviceType device) {
    sequential_add(seq, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(seq, (Module*)nn_conv2d(channels, se_channels, 1, 1, 0, 1, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_conv2d(se_channels, channels, 1, 1, 0, 1, true, dtype, device));
    sequential_add(seq, (Module*)nn_sigmoid());
}

static void add_mbconv(Sequential* model, int in_ch, int out_ch, int expand_ratio,
                        int kernel_size, int stride, int se_ratio,
                        DType dtype, DeviceType device) {
    int expanded = in_ch * expand_ratio;
    int padding = kernel_size / 2;

    if (expand_ratio != 1) {
        sequential_add(model, (Module*)nn_conv2d(in_ch, expanded, 1, 1, 0, 1, false, dtype, device));
        sequential_add(model, (Module*)nn_batchnorm2d(expanded, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(model, (Module*)nn_silu());
    }

    sequential_add(model, (Module*)nn_conv2d(expanded, expanded, kernel_size, stride, padding, 1, false, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(expanded, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(model, (Module*)nn_silu());

    int se_channels = in_ch / se_ratio;
    if (se_channels < 1) se_channels = 1;
    add_se_block(model, expanded, se_channels, dtype, device);

    sequential_add(model, (Module*)nn_conv2d(expanded, out_ch, 1, 1, 0, 1, false, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
}

Module* cml_zoo_efficientnet_b0(const EfficientNetConfig* config) {
    EfficientNetConfig cfg = config ? *config : efficientnet_b0_config(1000);

    Sequential* model = nn_sequential();

    sequential_add(model, (Module*)nn_conv2d(3, 32, 3, 2, 1, 1, false, cfg.dtype, cfg.device));
    sequential_add(model, (Module*)nn_batchnorm2d(32, 1e-5f, 0.1f, true, true, cfg.dtype, cfg.device));
    sequential_add(model, (Module*)nn_silu());

    int num_stages = sizeof(b0_blocks) / sizeof(b0_blocks[0]);
    for (int s = 0; s < num_stages; s++) {
        const EfficientNetBlockConfig* blk = &b0_blocks[s];
        for (int b = 0; b < blk->num_blocks; b++) {
            int in_ch = (b == 0) ? blk->in_channels : blk->out_channels;
            int stride = (b == 0) ? blk->stride : 1;
            add_mbconv(model, in_ch, blk->out_channels, blk->expand_ratio,
                       blk->kernel_size, stride, blk->se_ratio, cfg.dtype, cfg.device);
        }
    }

    sequential_add(model, (Module*)nn_conv2d(320, 1280, 1, 1, 0, 1, false, cfg.dtype, cfg.device));
    sequential_add(model, (Module*)nn_batchnorm2d(1280, 1e-5f, 0.1f, true, true, cfg.dtype, cfg.device));
    sequential_add(model, (Module*)nn_silu());
    sequential_add(model, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_dropout(cfg.dropout_rate, false));
    sequential_add(model, (Module*)nn_linear(1280, cfg.num_classes, cfg.dtype, cfg.device, true));

    LOG_INFO("Created EfficientNet-B0: %d classes", cfg.num_classes);
    return (Module*)model;
}
