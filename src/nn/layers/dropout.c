/**
 * @file dropout.c
 * @brief Dropout layer implementation
 */

#include "nn/layers/dropout.h"
#include "nn/module.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdlib.h>
#include <math.h>

static Tensor* dropout_forward(Module* module, Tensor* input) {
    Dropout* dropout = (Dropout*)module;

    if (!dropout || !input)
        return NULL;

    // In training mode, apply dropout
    if (module_is_training(module)) {
        // Create mask with dropout probability
        TensorConfig config = tensor_config_with_dtype_device(input->dtype, input->device);
        Tensor* mask        = tensor_empty(input->shape, input->ndim, &config);
        if (!mask)
            return NULL;

        float* mask_data  = (float*)tensor_data_ptr(mask);
        float* input_data = (float*)tensor_data_ptr(input);

        if (!mask_data || !input_data) {
            tensor_free(mask);
            return NULL;
        }

        for (size_t i = 0; i < mask->numel; i++) {
            mask_data[i] =
                ((float)rand() / RAND_MAX) > dropout->p ? 1.0f / (1.0f - dropout->p) : 0.0f;
        }

        Tensor* output = tensor_mul(input, mask);
        tensor_free(mask);

        return output;
    } else {
        return tensor_clone(input);
    }
}

static void dropout_free(Module* module) { CM_FREE(module); }

Dropout* nn_dropout(float p, bool inplace) {
    Dropout* dropout = CM_MALLOC(sizeof(Dropout));
    if (!dropout)
        return NULL;

    if (module_init((Module*)dropout, "Dropout", dropout_forward, dropout_free) != 0) {
        CM_FREE(dropout);
        return NULL;
    }

    dropout->p       = p;
    dropout->inplace = inplace;

    return dropout;
}
