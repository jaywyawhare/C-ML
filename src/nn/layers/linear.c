/**
 * @file linear.c
 * @brief Linear (fully connected) layer implementation
 *
 */

#include "nn/layers/linear.h"
#include "nn/module.h"
#include "tensor/ops.h"
#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward pass: output = input @ weight.T + bias
static Tensor* linear_forward_fn(Module* module, Tensor* input) {
    Linear* linear = (Linear*)module;

    if (!linear || !input)
        return NULL;

    // Get weight and bias
    Parameter* weight_param = linear->weight;
    Parameter* bias_param   = linear->bias;

    if (!weight_param || !weight_param->tensor) {
        LOG_ERROR("Linear layer missing weight parameter");
        return NULL;
    }

    Tensor* weight = weight_param->tensor;

    // Compute output = input @ weight.T
    // If input is [batch, in_features] and weight is [out_features, in_features]
    // Then output = input @ weight.T = [batch, out_features]

    Tensor* weight_t = tensor_transpose(weight, 0, 1);
    if (!weight_t) {
        LOG_ERROR("Failed to transpose weight");
        return NULL;
    }

    Tensor* output = tensor_matmul(input, weight_t);

    if (!output) {
        LOG_ERROR("Failed matrix multiplication in Linear layer");
        tensor_free(weight_t);
        return NULL;
    }

    // Add bias if present
    if (linear->use_bias && bias_param && bias_param->tensor) {
        Tensor* bias             = bias_param->tensor;
        Tensor* output_with_bias = tensor_add(output, bias);
        // Don't free output here - it's part of the computation graph
        // and needed for the backward pass. The Add function's backward
        // needs to access output through fn->inputs[0].
        output = output_with_bias;
    }

    return output;
}

// Free function
static void linear_free_fn(Module* module) {
    Linear* linear = (Linear*)module;
    if (!linear)
        return;

    CM_FREE(linear);
}

// Initialize weight with Xavier/Glorot initialization
static void xavier_init(Tensor* tensor, int in_features, int out_features) {
    if (!tensor || !tensor->data)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    float scale  = sqrtf(2.0f / (in_features + out_features));
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

// Initialize bias to zeros
static void zeros_init(Tensor* tensor, int out_features) {
    (void)out_features;
    if (!tensor || !tensor->data)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    size_t numel = tensor->numel;
    for (size_t i = 0; i < numel; i++) {
        data[i] = 0.0f;
    }
}

Linear* nn_linear(int in_features, int out_features, DType dtype, DeviceType device,
                  bool use_bias) {
    return nn_linear_with_init(in_features, out_features, dtype, device, use_bias,
                               (void (*)(Tensor*, int, int))xavier_init,
                               (void (*)(Tensor*, int))zeros_init);
}

Linear* nn_linear_with_init(int in_features, int out_features, DType dtype, DeviceType device,
                            bool use_bias, void (*weight_init)(Tensor*, int, int),
                            void (*bias_init)(Tensor*, int)) {
    Linear* linear = CM_MALLOC(sizeof(Linear));
    if (!linear) {
        LOG_ERROR("Failed to allocate memory for Linear layer");
        return NULL;
    }

    // Initialize base module
    if (module_init((Module*)linear, "Linear", linear_forward_fn, linear_free_fn) != 0) {
        CM_FREE(linear);
        return NULL;
    }

    linear->in_features      = in_features;
    linear->out_features     = out_features;
    linear->use_bias         = use_bias;
    linear->transpose_weight = false;

    // Create weight tensor [out_features, in_features]
    int weight_shape[] = {out_features, in_features};
    Tensor* weight     = tensor_empty(weight_shape, 2, dtype, device);
    if (!weight) {
        module_free((Module*)linear);
        return NULL;
    }

    // Initialize weight
    if (weight_init) {
        weight_init(weight, in_features, out_features);
    } else {
        xavier_init(weight, in_features, out_features);
    }

    // Add weight parameter
    if (module_add_parameter((Module*)linear, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)linear);
        return NULL;
    }

    linear->weight = module_get_parameter((Module*)linear, "weight");

    // Create bias tensor [out_features] if needed
    if (use_bias) {
        int bias_shape[] = {out_features};
        Tensor* bias     = tensor_zeros(bias_shape, 1, dtype, device);
        if (!bias) {
            module_free((Module*)linear);
            return NULL;
        }

        // Initialize bias
        if (bias_init) {
            bias_init(bias, out_features);
        } else {
            zeros_init(bias, out_features);
        }

        // Add bias parameter
        if (module_add_parameter((Module*)linear, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)linear);
            return NULL;
        }

        linear->bias = module_get_parameter((Module*)linear, "bias");
    } else {
        linear->bias = NULL;
    }

    LOG_DEBUG("Created Linear layer: %d -> %d (bias=%d)", in_features, out_features, use_bias);

    return linear;
}

// Getter functions
int linear_get_in_features(Linear* linear) { return linear ? linear->in_features : 0; }

int linear_get_out_features(Linear* linear) { return linear ? linear->out_features : 0; }

Parameter* linear_get_weight(Linear* linear) { return linear ? linear->weight : NULL; }

Parameter* linear_get_bias(Linear* linear) { return linear ? linear->bias : NULL; }

int linear_set_weight(Linear* linear, Tensor* weight) {
    if (!linear || !weight)
        return -1;

    if (module_set_parameter((Module*)linear, "weight", weight) != 0) {
        return -1;
    }

    linear->weight = module_get_parameter((Module*)linear, "weight");
    return 0;
}

int linear_set_bias(Linear* linear, Tensor* bias) {
    if (!linear || !bias)
        return -1;

    if (!linear->use_bias) {
        LOG_WARNING("Cannot set bias on Linear layer without bias");
        return -1;
    }

    if (module_set_parameter((Module*)linear, "bias", bias) != 0) {
        return -1;
    }

    linear->bias = module_get_parameter((Module*)linear, "bias");
    return 0;
}

void linear_set_use_bias(Linear* linear, bool use_bias) {
    if (linear) {
        linear->use_bias = use_bias;
    }
}

bool linear_get_use_bias(Linear* linear) { return linear ? linear->use_bias : false; }

void linear_set_transpose_weight(Linear* linear, bool transpose) {
    if (linear) {
        linear->transpose_weight = transpose;
    }
}

bool linear_get_transpose_weight(Linear* linear) {
    return linear ? linear->transpose_weight : false;
}

void linear_print_summary(Linear* linear, int indent) {
    if (!linear)
        return;

    for (int i = 0; i < indent; i++)
        printf("  ");
    printf("Linear(%d, %d, bias=%s)\n", linear->in_features, linear->out_features,
           linear->use_bias ? "True" : "False");
}

int linear_get_parameter_count(Linear* linear) {
    if (!linear)
        return 0;
    return module_get_parameter_count((Module*)linear);
}

int linear_get_total_parameters(Linear* linear) {
    if (!linear)
        return 0;

    int weight_params =
        linear->weight && linear->weight->tensor ? (int)linear->weight->tensor->numel : 0;
    int bias_params = (linear->use_bias && linear->bias && linear->bias->tensor)
                          ? (int)linear->bias->tensor->numel
                          : 0;

    return weight_params + bias_params;
}
