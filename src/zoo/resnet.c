#include "zoo/resnet.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include "alloc/cml_allocator.h"

/* Residual block: y = relu(conv(x) + (downsample ? downsample(x) : x)).
 * Both the basic and bottleneck variants are this shape; only the conv stack
 * inside differs. */
typedef struct {
    Module base;
    Sequential* conv;
    Sequential* downsample;
} ResidualBlock;

/** Forward for a residual block: relu(conv(x) + shortcut), with optional downsample. */
static Tensor* residual_block_forward(Module* module, Tensor* input) {
    ResidualBlock* block = (ResidualBlock*)module;
    if (!block || !input)
        return NULL;

    Tensor* out = module_forward((Module*)block->conv, input);
    if (!out)
        return NULL;

    Tensor* skip = input;
    if (block->downsample)
        skip = module_forward((Module*)block->downsample, input);

    Tensor* result = tensor_add(out, skip);
    if (!result)
        return NULL;

    return f_relu(result);
}

/** Free a residual block's conv stack and optional downsample shortcut. */
static void residual_block_free(Module* module) {
    ResidualBlock* block = (ResidualBlock*)module;
    if (!block)
        return;
    if (block->conv)
        module_free((Module*)block->conv);
    if (block->downsample)
        module_free((Module*)block->downsample);
    cml_free(block);
}

/** Allocate a residual block named @p name with an empty conv stack, plus the
 * 1x1 projection shortcut when the shape changes. */
static ResidualBlock* residual_block_new(const char* name, int in_channels, int out_channels,
                                         int stride, DType dtype, DeviceType device) {
    ResidualBlock* block = cml_malloc(sizeof(ResidualBlock));
    if (!block)
        return NULL;

    if (module_init((Module*)block, name, residual_block_forward, residual_block_free) != 0) {
        cml_free(block);
        return NULL;
    }

    block->conv = nn_sequential();
    if (!block->conv) {
        cml_free(block);
        return NULL;
    }

    block->downsample = NULL;
    if (in_channels != out_channels || stride != 1) {
        block->downsample = nn_sequential();
        sequential_add(block->downsample, (Module*)nn_conv2d(in_channels, out_channels, 1, stride,
                                                             0, 1, false, dtype, device));
        sequential_add(block->downsample, (Module*)nn_batchnorm2d(out_channels, 1e-5f, 0.1f, true,
                                                                  true, dtype, device));
    }

    return block;
}

/** Build a bottleneck residual block: 1x1 reduce -> 3x3 -> 1x1 expand with BN/ReLU. */
static Module* create_bottleneck(int in_channels, int mid_channels, int out_channels, int stride,
                                 DType dtype, DeviceType device) {
    ResidualBlock* block =
        residual_block_new("Bottleneck", in_channels, out_channels, stride, dtype, device);
    if (!block)
        return NULL;

    sequential_add(block->conv,
                   (Module*)nn_conv2d(in_channels, mid_channels, 1, 1, 0, 1, false, dtype, device));
    sequential_add(block->conv,
                   (Module*)nn_batchnorm2d(mid_channels, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block->conv, (Module*)nn_relu(false));

    sequential_add(block->conv, (Module*)nn_conv2d(mid_channels, mid_channels, 3, stride, 1, 1,
                                                   false, dtype, device));
    sequential_add(block->conv,
                   (Module*)nn_batchnorm2d(mid_channels, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block->conv, (Module*)nn_relu(false));

    sequential_add(block->conv, (Module*)nn_conv2d(mid_channels, out_channels, 1, 1, 0, 1, false,
                                                   dtype, device));
    sequential_add(block->conv,
                   (Module*)nn_batchnorm2d(out_channels, 1e-5f, 0.1f, true, true, dtype, device));

    return (Module*)block;
}

/** Build a basic residual block: two 3x3 convs with BN, ReLU between them. */
static Module* create_basic_block(int in_channels, int out_channels, int stride, DType dtype,
                                  DeviceType device) {
    ResidualBlock* block =
        residual_block_new("BasicBlock", in_channels, out_channels, stride, dtype, device);
    if (!block)
        return NULL;

    sequential_add(block->conv, (Module*)nn_conv2d(in_channels, out_channels, 3, stride, 1, 1,
                                                   false, dtype, device));
    sequential_add(block->conv,
                   (Module*)nn_batchnorm2d(out_channels, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block->conv, (Module*)nn_relu(false));
    sequential_add(block->conv, (Module*)nn_conv2d(out_channels, out_channels, 3, 1, 1, 1, false,
                                                   dtype, device));
    sequential_add(block->conv,
                   (Module*)nn_batchnorm2d(out_channels, 1e-5f, 0.1f, true, true, dtype, device));

    return (Module*)block;
}

/** Stack @p num_blocks bottleneck blocks; only the first applies the stride/projection. */
static Module* build_bottleneck_stage(int num_blocks, int in_channels, int mid_channels,
                                      int out_channels, int stride, DType dtype,
                                      DeviceType device) {
    Sequential* stage = nn_sequential();
    if (!stage)
        return NULL;

    sequential_add(
        stage, create_bottleneck(in_channels, mid_channels, out_channels, stride, dtype, device));
    for (int i = 1; i < num_blocks; i++)
        sequential_add(
            stage, create_bottleneck(out_channels, mid_channels, out_channels, 1, dtype, device));

    return (Module*)stage;
}

/** Stack @p num_blocks basic blocks; only the first applies the stride/projection. */
static Module* build_basic_stage(int num_blocks, int in_channels, int out_channels, int stride,
                                 DType dtype, DeviceType device) {
    Sequential* stage = nn_sequential();
    if (!stage)
        return NULL;

    sequential_add(stage, create_basic_block(in_channels, out_channels, stride, dtype, device));
    for (int i = 1; i < num_blocks; i++)
        sequential_add(stage, create_basic_block(out_channels, out_channels, 1, dtype, device));

    return (Module*)stage;
}

/** Build ResNet-50: 7x7 stem then bottleneck stages (3,4,6,3), global pool and linear head. */
Module* cml_zoo_resnet50_create(int num_classes, DType dtype, DeviceType device) {
    if (num_classes <= 0)
        num_classes = 1000;

    Sequential* model = nn_sequential();
    if (!model)
        return NULL;

    sequential_add(model, (Module*)nn_conv2d(3, 64, 7, 2, 3, 1, false, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(3, 2, 1, 1, false));

    sequential_add(model, build_bottleneck_stage(3, 64, 64, 256, 1, dtype, device));
    sequential_add(model, build_bottleneck_stage(4, 256, 128, 512, 2, dtype, device));
    sequential_add(model, build_bottleneck_stage(6, 512, 256, 1024, 2, dtype, device));
    sequential_add(model, build_bottleneck_stage(3, 1024, 512, 2048, 2, dtype, device));

    sequential_add(model, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_linear(2048, num_classes, dtype, device, true));

    LOG_INFO("Created ResNet-50 (%d classes)", num_classes);
    return (Module*)model;
}

/** Build ResNet-34: 7x7 stem then basic-block stages (3,4,6,3), global pool and linear head. */
Module* cml_zoo_resnet34_create(int num_classes, DType dtype, DeviceType device) {
    if (num_classes <= 0)
        num_classes = 1000;

    Sequential* model = nn_sequential();
    if (!model)
        return NULL;

    sequential_add(model, (Module*)nn_conv2d(3, 64, 7, 2, 3, 1, false, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(3, 2, 1, 1, false));

    sequential_add(model, build_basic_stage(3, 64, 64, 1, dtype, device));
    sequential_add(model, build_basic_stage(4, 64, 128, 2, dtype, device));
    sequential_add(model, build_basic_stage(6, 128, 256, 2, dtype, device));
    sequential_add(model, build_basic_stage(3, 256, 512, 2, dtype, device));

    sequential_add(model, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_linear(512, num_classes, dtype, device, true));

    LOG_INFO("Created ResNet-34 (%d classes)", num_classes);
    return (Module*)model;
}

/** Build ResNet-18: 7x7 stem then basic-block stages (2,2,2,2), global pool and linear head. */
Module* cml_zoo_resnet18_create(int num_classes, DType dtype, DeviceType device) {
    if (num_classes <= 0)
        num_classes = 1000;

    Sequential* model = nn_sequential();
    if (!model)
        return NULL;

    sequential_add(model, (Module*)nn_conv2d(3, 64, 7, 2, 3, 1, false, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(3, 2, 1, 1, false));

    sequential_add(model, build_basic_stage(2, 64, 64, 1, dtype, device));
    sequential_add(model, build_basic_stage(2, 64, 128, 2, dtype, device));
    sequential_add(model, build_basic_stage(2, 128, 256, 2, dtype, device));
    sequential_add(model, build_basic_stage(2, 256, 512, 2, dtype, device));

    sequential_add(model, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_linear(512, num_classes, dtype, device, true));

    LOG_INFO("Created ResNet-18 (%d classes)", num_classes);
    return (Module*)model;
}
