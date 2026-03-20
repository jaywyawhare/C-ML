#include "zoo/inception.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>

static Module* conv_bn_relu(int in_ch, int out_ch, int kernel, int stride, int padding,
                             DType dtype, DeviceType device) {
    Sequential* block = nn_sequential();
    if (!block)
        return NULL;

    sequential_add(block, (Module*)nn_conv2d(in_ch, out_ch, kernel, stride, padding, 1, false, dtype, device));
    sequential_add(block, (Module*)nn_batchnorm2d(out_ch, 1e-3f, 0.1f, true, true, dtype, device));
    sequential_add(block, (Module*)nn_relu(false));

    return (Module*)block;
}

typedef struct {
    Module base;
    Module** branches;
    int num_branches;
} ConcatBlock;

static Tensor* concat_forward(Module* module, Tensor* input) {
    ConcatBlock* blk = (ConcatBlock*)module;
    if (!blk || !input)
        return NULL;

    Tensor** outputs = malloc(sizeof(Tensor*) * blk->num_branches);
    if (!outputs)
        return NULL;

    for (int i = 0; i < blk->num_branches; i++) {
        outputs[i] = module_forward(blk->branches[i], input);
        if (!outputs[i]) {
            free(outputs);
            return NULL;
        }
    }

    Tensor* result = tensor_concat(outputs, blk->num_branches, 1);
    free(outputs);
    return result;
}

static void concat_free(Module* module) {
    ConcatBlock* blk = (ConcatBlock*)module;
    if (!blk)
        return;
    for (int i = 0; i < blk->num_branches; i++)
        module_free(blk->branches[i]);
    free(blk->branches);
    free(blk);
}

static ConcatBlock* create_concat_block(const char* name, int num_branches) {
    ConcatBlock* blk = malloc(sizeof(ConcatBlock));
    if (!blk)
        return NULL;

    if (module_init((Module*)blk, name, concat_forward, concat_free) != 0) {
        free(blk);
        return NULL;
    }

    blk->branches = calloc(num_branches, sizeof(Module*));
    if (!blk->branches) {
        free(blk);
        return NULL;
    }
    blk->num_branches = num_branches;

    return blk;
}

static Module* create_inception_a(int in_ch, int pool_proj, DType dtype, DeviceType device) {
    ConcatBlock* blk = create_concat_block("InceptionA", 4);
    if (!blk)
        return NULL;

    Sequential* b0 = nn_sequential();
    sequential_add(b0, conv_bn_relu(in_ch, 64, 1, 1, 0, dtype, device));
    blk->branches[0] = (Module*)b0;

    Sequential* b1 = nn_sequential();
    sequential_add(b1, conv_bn_relu(in_ch, 48, 1, 1, 0, dtype, device));
    sequential_add(b1, conv_bn_relu(48, 64, 5, 1, 2, dtype, device));
    blk->branches[1] = (Module*)b1;

    Sequential* b2 = nn_sequential();
    sequential_add(b2, conv_bn_relu(in_ch, 64, 1, 1, 0, dtype, device));
    sequential_add(b2, conv_bn_relu(64, 96, 3, 1, 1, dtype, device));
    sequential_add(b2, conv_bn_relu(96, 96, 3, 1, 1, dtype, device));
    blk->branches[2] = (Module*)b2;

    Sequential* b3 = nn_sequential();
    sequential_add(b3, (Module*)nn_avgpool2d(3, 1, 1, false, true));
    sequential_add(b3, conv_bn_relu(in_ch, pool_proj, 1, 1, 0, dtype, device));
    blk->branches[3] = (Module*)b3;

    return (Module*)blk;
}

static Module* create_inception_b(int in_ch, DType dtype, DeviceType device) {
    ConcatBlock* blk = create_concat_block("InceptionB", 3);
    if (!blk)
        return NULL;

    Sequential* b0 = nn_sequential();
    sequential_add(b0, conv_bn_relu(in_ch, 384, 3, 2, 0, dtype, device));
    blk->branches[0] = (Module*)b0;

    Sequential* b1 = nn_sequential();
    sequential_add(b1, conv_bn_relu(in_ch, 64, 1, 1, 0, dtype, device));
    sequential_add(b1, conv_bn_relu(64, 96, 3, 1, 1, dtype, device));
    sequential_add(b1, conv_bn_relu(96, 96, 3, 2, 0, dtype, device));
    blk->branches[1] = (Module*)b1;

    Sequential* b2 = nn_sequential();
    sequential_add(b2, (Module*)nn_maxpool2d(3, 2, 0, 1, false));
    blk->branches[2] = (Module*)b2;

    return (Module*)blk;
}

static Module* create_inception_c(int in_ch, int c7, DType dtype, DeviceType device) {
    ConcatBlock* blk = create_concat_block("InceptionC", 4);
    if (!blk)
        return NULL;

    Sequential* b0 = nn_sequential();
    sequential_add(b0, conv_bn_relu(in_ch, 192, 1, 1, 0, dtype, device));
    blk->branches[0] = (Module*)b0;

    Sequential* b1 = nn_sequential();
    sequential_add(b1, conv_bn_relu(in_ch, c7, 1, 1, 0, dtype, device));
    sequential_add(b1, conv_bn_relu(c7, c7, 7, 1, 3, dtype, device));
    sequential_add(b1, conv_bn_relu(c7, 192, 7, 1, 3, dtype, device));
    blk->branches[1] = (Module*)b1;

    Sequential* b2 = nn_sequential();
    sequential_add(b2, conv_bn_relu(in_ch, c7, 1, 1, 0, dtype, device));
    sequential_add(b2, conv_bn_relu(c7, c7, 7, 1, 3, dtype, device));
    sequential_add(b2, conv_bn_relu(c7, c7, 7, 1, 3, dtype, device));
    sequential_add(b2, conv_bn_relu(c7, c7, 7, 1, 3, dtype, device));
    sequential_add(b2, conv_bn_relu(c7, 192, 7, 1, 3, dtype, device));
    blk->branches[2] = (Module*)b2;

    Sequential* b3 = nn_sequential();
    sequential_add(b3, (Module*)nn_avgpool2d(3, 1, 1, false, true));
    sequential_add(b3, conv_bn_relu(in_ch, 192, 1, 1, 0, dtype, device));
    blk->branches[3] = (Module*)b3;

    return (Module*)blk;
}

static Module* create_inception_d(int in_ch, DType dtype, DeviceType device) {
    ConcatBlock* blk = create_concat_block("InceptionD", 3);
    if (!blk)
        return NULL;

    Sequential* b0 = nn_sequential();
    sequential_add(b0, conv_bn_relu(in_ch, 192, 1, 1, 0, dtype, device));
    sequential_add(b0, conv_bn_relu(192, 320, 3, 2, 0, dtype, device));
    blk->branches[0] = (Module*)b0;

    Sequential* b1 = nn_sequential();
    sequential_add(b1, conv_bn_relu(in_ch, 192, 1, 1, 0, dtype, device));
    sequential_add(b1, conv_bn_relu(192, 192, 7, 1, 3, dtype, device));
    sequential_add(b1, conv_bn_relu(192, 192, 7, 1, 3, dtype, device));
    sequential_add(b1, conv_bn_relu(192, 192, 3, 2, 0, dtype, device));
    blk->branches[1] = (Module*)b1;

    Sequential* b2 = nn_sequential();
    sequential_add(b2, (Module*)nn_maxpool2d(3, 2, 0, 1, false));
    blk->branches[2] = (Module*)b2;

    return (Module*)blk;
}

static Module* create_inception_e(int in_ch, DType dtype, DeviceType device) {
    ConcatBlock* blk = create_concat_block("InceptionE", 4);
    if (!blk)
        return NULL;

    Sequential* b0 = nn_sequential();
    sequential_add(b0, conv_bn_relu(in_ch, 320, 1, 1, 0, dtype, device));
    blk->branches[0] = (Module*)b0;

    Sequential* b1 = nn_sequential();
    sequential_add(b1, conv_bn_relu(in_ch, 384, 1, 1, 0, dtype, device));
    sequential_add(b1, conv_bn_relu(384, 384, 3, 1, 1, dtype, device));
    sequential_add(b1, conv_bn_relu(384, 384, 3, 1, 1, dtype, device));
    blk->branches[1] = (Module*)b1;

    Sequential* b2 = nn_sequential();
    sequential_add(b2, conv_bn_relu(in_ch, 448, 1, 1, 0, dtype, device));
    sequential_add(b2, conv_bn_relu(448, 384, 3, 1, 1, dtype, device));
    sequential_add(b2, conv_bn_relu(384, 384, 3, 1, 1, dtype, device));
    sequential_add(b2, conv_bn_relu(384, 384, 3, 1, 1, dtype, device));
    blk->branches[2] = (Module*)b2;

    Sequential* b3 = nn_sequential();
    sequential_add(b3, (Module*)nn_avgpool2d(3, 1, 1, false, true));
    sequential_add(b3, conv_bn_relu(in_ch, 192, 1, 1, 0, dtype, device));
    blk->branches[3] = (Module*)b3;

    return (Module*)blk;
}

Module* cml_zoo_inception_v3_create(int num_classes, DType dtype, DeviceType device) {
    if (num_classes <= 0)
        num_classes = 1000;

    Sequential* model = nn_sequential();
    if (!model)
        return NULL;

    sequential_add(model, conv_bn_relu(3, 32, 3, 2, 0, dtype, device));
    sequential_add(model, conv_bn_relu(32, 32, 3, 1, 0, dtype, device));
    sequential_add(model, conv_bn_relu(32, 64, 3, 1, 1, dtype, device));
    sequential_add(model, (Module*)nn_maxpool2d(3, 2, 0, 1, false));
    sequential_add(model, conv_bn_relu(64, 80, 1, 1, 0, dtype, device));
    sequential_add(model, conv_bn_relu(80, 192, 3, 1, 0, dtype, device));
    sequential_add(model, (Module*)nn_maxpool2d(3, 2, 0, 1, false));

    sequential_add(model, create_inception_a(192, 32, dtype, device));
    sequential_add(model, create_inception_a(256, 64, dtype, device));
    sequential_add(model, create_inception_a(288, 64, dtype, device));

    sequential_add(model, create_inception_b(288, dtype, device));

    sequential_add(model, create_inception_c(768, 128, dtype, device));
    sequential_add(model, create_inception_c(768, 160, dtype, device));
    sequential_add(model, create_inception_c(768, 160, dtype, device));
    sequential_add(model, create_inception_c(768, 192, dtype, device));

    sequential_add(model, create_inception_d(768, dtype, device));

    sequential_add(model, create_inception_e(1280, dtype, device));
    sequential_add(model, create_inception_e(2048, dtype, device));

    sequential_add(model, (Module*)nn_adaptive_avgpool2d(1, 1));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_dropout(0.2f, false));
    sequential_add(model, (Module*)nn_linear(2048, num_classes, dtype, device, true));

    LOG_INFO("Created Inception V3 (%d classes)", num_classes);
    return (Module*)model;
}
