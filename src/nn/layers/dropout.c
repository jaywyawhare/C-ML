#include "nn/layers/dropout.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

static Tensor* dropout_forward(Module* module, Tensor* input) {
    Dropout* dropout = (Dropout*)module;

    if (!dropout || !input)
        return NULL;
    if (module_is_training(module)) {
        TensorConfig config = (TensorConfig){
            .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

        if (dropout->p <= 0.0f) {
            Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
            if (!ones) return NULL;
            Tensor* out = uop_mul(input, ones);
            tensor_free(ones);
            return out;
        }
        if (dropout->p >= 1.0f) {
            return tensor_zeros(input->shape, input->ndim, &config);
        }

        Tensor* rand = tensor_rand(input->shape, input->ndim, &config);
        Tensor* threshold = tensor_full(input->shape, input->ndim, &config, dropout->p);
        Tensor* scale = tensor_full(input->shape, input->ndim, &config, 1.0f / (1.0f - dropout->p));
        if (!rand || !threshold || !scale) {
            tensor_free(rand);
            tensor_free(threshold);
            tensor_free(scale);
            return NULL;
        }

        Tensor* keep = uop_cmpgt(rand, threshold);
        tensor_free(rand);
        tensor_free(threshold);
        if (!keep) {
            tensor_free(scale);
            return NULL;
        }

        Tensor* mask = uop_mul(keep, scale);
        tensor_free(keep);
        tensor_free(scale);
        if (!mask)
            return NULL;

        Tensor* output = uop_mul(input, mask);
        tensor_free(mask);
        return output;
    }

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones)
        return NULL;
    Tensor* out = uop_mul(input, ones);
    tensor_free(ones);
    return out;
}

static void dropout_free(Module* module) { free(module); }

Dropout* nn_dropout(float p, bool inplace) {
    Dropout* dropout = malloc(sizeof(Dropout));
    if (!dropout)
        return NULL;

    if (module_init((Module*)dropout, "Dropout", dropout_forward, dropout_free) != 0) {
        free(dropout);
        return NULL;
    }

    dropout->p       = p;
    dropout->inplace = inplace;

    return dropout;
}
