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
        Tensor* mask = tensor_empty(input->shape, input->ndim, &config);
        if (!mask)
            return NULL;

        float* mask_data  = (float*)tensor_data_ptr(mask);
        float* input_data = (float*)tensor_data_ptr(input);

        if (!mask_data || !input_data) {
            tensor_free(mask);
            return NULL;
        }

        for (size_t i = 0; i < mask->numel; i++) {
            mask_data[i] = ((float)rand() / (float)(float)RAND_MAX) > dropout->p
                               ? 1.0f / (1.0f - dropout->p)
                               : 0.0f;
        }

        Tensor* output = uop_mul(input, mask);
        tensor_free(mask);

        return output;
    } else {
        return tensor_clone(input);
    }
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
