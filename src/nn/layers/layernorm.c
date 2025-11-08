/**
 * @file layernorm.c
 * @brief Layer Normalization layer implementation
 */

#include "nn/layers/layernorm.h"
#include "nn/module.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "autograd/autograd.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* layernorm_forward(Module* module, Tensor* input) {
    LayerNorm* ln = (LayerNorm*)module;

    if (!ln || !input)
        return NULL;

    if (input->ndim < 1) {
        LOG_ERROR("LayerNorm expects at least 1D input, got %dD", input->ndim);
        return NULL;
    }

    int last_dim = input->shape[input->ndim - 1];

    if (last_dim != ln->normalized_shape) {
        LOG_ERROR("LayerNorm: input last dimension (%d) doesn't match normalized_shape (%d)",
                  last_dim, ln->normalized_shape);
        return NULL;
    }

    size_t num_elements = 1;
    for (int i = 0; i < input->ndim - 1; i++) {
        num_elements *= input->shape[i];
    }

    TensorConfig config = tensor_config_with_dtype_device(input->dtype, input->device);
    Tensor* output      = tensor_empty(input->shape, input->ndim, &config);
    if (!output)
        return NULL;

    float* input_data  = (float*)tensor_data_ptr(input);
    float* output_data = (float*)tensor_data_ptr(output);

    if (!input_data || !output_data) {
        tensor_free(output);
        return NULL;
    }

    // Get weight (gamma) and bias (beta) if affine
    float* weight_data = NULL;
    float* bias_data   = NULL;

    if (ln->affine && ln->weight && ln->bias) {
        weight_data = (float*)tensor_data_ptr(ln->weight->tensor);
        bias_data   = (float*)tensor_data_ptr(ln->bias->tensor);
    }

    for (size_t i = 0; i < num_elements; i++) {
        // Compute mean over last dimension
        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            size_t idx = i * last_dim + j;
            sum += input_data[idx];
        }
        float mean = sum / last_dim;

        // Compute variance over last dimension
        float sum_sq = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            size_t idx = i * last_dim + j;
            float diff = input_data[idx] - mean;
            sum_sq += diff * diff;
        }
        float var = sum_sq / last_dim;
        float std = sqrtf(var + ln->eps);

        // Normalize: (x - mean) / sqrt(var + eps)
        // Then scale and shift: gamma * normalized + beta
        for (int j = 0; j < last_dim; j++) {
            size_t idx       = i * last_dim + j;
            float normalized = (input_data[idx] - mean) / std;

            float gamma = ln->affine && weight_data ? weight_data[j] : 1.0f;
            float beta  = ln->affine && bias_data ? bias_data[j] : 0.0f;

            output_data[idx] = gamma * normalized + beta;
        }
    }

    // Setup autograd if needed
    if (autograd_is_grad_enabled() && input->requires_grad) {
        // TODO
        // Note: LayerNorm backward pass would need to be implemented
        // For now, we'll mark output as requiring grad if input requires grad
        output->requires_grad = true;
    }

    return output;
}

static void layernorm_free(Module* module) {
    LayerNorm* ln = (LayerNorm*)module;
    if (!ln)
        return;

    module_free(module);
}

LayerNorm* nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                        DeviceType device) {
    if (normalized_shape <= 0) {
        LOG_ERROR("LayerNorm: normalized_shape must be positive, got %d", normalized_shape);
        return NULL;
    }

    LayerNorm* ln = CM_MALLOC(sizeof(LayerNorm));
    if (!ln)
        return NULL;

    // Initialize base module
    if (module_init((Module*)ln, "LayerNorm", layernorm_forward, layernorm_free) != 0) {
        CM_FREE(ln);
        return NULL;
    }
    ln->normalized_shape = normalized_shape;
    ln->eps              = eps > 0.0f ? eps : 1e-5f;
    ln->affine           = affine;
    ln->weight           = NULL;
    ln->bias             = NULL;

    // Create learnable parameters if affine
    if (affine) {
        int weight_shape[] = {normalized_shape};

        // Create weight (gamma) - initialized to ones
        TensorConfig config = tensor_config_with_dtype_device(dtype, device);
        Tensor* weight      = tensor_ones(weight_shape, 1, &config);
        if (!weight) {
            module_free((Module*)ln);
            return NULL;
        }

        // Add weight parameter
        if (module_add_parameter((Module*)ln, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)ln);
            return NULL;
        }

        ln->weight = module_get_parameter((Module*)ln, "weight");

        // Create bias (beta) - initialized to zeros
        Tensor* bias = tensor_zeros(weight_shape, 1, &config);
        if (!bias) {
            module_free((Module*)ln);
            return NULL;
        }

        // Add bias parameter
        if (module_add_parameter((Module*)ln, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)ln);
            return NULL;
        }

        ln->bias = module_get_parameter((Module*)ln, "bias");
    } else {
        ln->weight = NULL;
        ln->bias   = NULL;
    }

    LOG_DEBUG("Created LayerNorm layer: normalized_shape=%d, eps=%.6f, affine=%d", normalized_shape,
              ln->eps, affine);

    return ln;
}
