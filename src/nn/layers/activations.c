/**
 * @file activations.c
 * @brief Activation function layers implementation
 */

#include "nn/layers/activations.h"
#include "nn.h"
#include "autograd/autograd.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// ReLU Forward - IR-based for autograd support
static Tensor* relu_forward(Module* module, Tensor* input) {
    (void)module; // ReLU is stateless

    if (!input)
        return NULL;

    LOG_DEBUG("ReLU forward: Computing max(0, input) (IR-based)");

    // Use IR-based ReLU for autograd support
    return uop_relu(input);
}

static void relu_free(Module* module) { free(module); }

ReLU* nn_relu(bool inplace) {
    ReLU* relu = malloc(sizeof(ReLU));
    if (!relu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for ReLU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)relu, "ReLU", relu_forward, relu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ReLU module", __FILE__,
                         __LINE__, __func__);
        free(relu);
        return NULL;
    }

    relu->inplace = inplace;
    return relu;
}

// LeakyReLU Forward
static Tensor* leaky_relu_forward(Module* module, Tensor* input) {
    LeakyReLU* leaky_relu = (LeakyReLU*)module;

    return uop_leaky_relu(input, leaky_relu->negative_slope);
}

static void leaky_relu_free(Module* module) { free(module); }

LeakyReLU* nn_leaky_relu(float negative_slope, bool inplace) {
    LeakyReLU* leaky_relu = malloc(sizeof(LeakyReLU));
    if (!leaky_relu)
        return NULL;

    if (module_init((Module*)leaky_relu, "LeakyReLU", leaky_relu_forward, leaky_relu_free) != 0) {
        free(leaky_relu);
        return NULL;
    }

    leaky_relu->negative_slope = negative_slope;
    leaky_relu->inplace        = inplace;
    return leaky_relu;
}

// Sigmoid Forward - IR-based for autograd support
static Tensor* sigmoid_forward(Module* module, Tensor* input) {
    (void)module; // Sigmoid is stateless

    if (!input)
        return NULL;

    LOG_DEBUG("Sigmoid forward: Computing 1/(1+exp(-x)) (IR-based)");

    // Use IR-based Sigmoid for autograd support
    return uop_sigmoid(input);
}

static void sigmoid_free(Module* module) { free(module); }

Sigmoid* nn_sigmoid(void) {
    Sigmoid* sigmoid = malloc(sizeof(Sigmoid));
    if (!sigmoid) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Sigmoid layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)sigmoid, "Sigmoid", sigmoid_forward, sigmoid_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Sigmoid module", __FILE__,
                         __LINE__, __func__);
        free(sigmoid);
        return NULL;
    }

    return sigmoid;
}

// Tanh Forward - IR-based for autograd support
static Tensor* tanh_forward(Module* module, Tensor* input) {
    (void)module; // Tanh is stateless

    if (!input)
        return NULL;

    LOG_DEBUG("Tanh forward: Computing tanh(x) (IR-based)");

    // Use IR-based Tanh for autograd support
    return uop_tanh(input);
}

static void tanh_free(Module* module) { free(module); }

Tanh* nn_tanh(void) {
    Tanh* tanh = malloc(sizeof(Tanh));
    if (!tanh) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Tanh layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)tanh, "Tanh", tanh_forward, tanh_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Tanh module", __FILE__,
                         __LINE__, __func__);
        free(tanh);
        return NULL;
    }

    return tanh;
}

// GELU Forward - Gaussian Error Linear Unit
static Tensor* gelu_forward(Module* module, Tensor* input) {
    GELU* gelu = (GELU*)module;

    if (!gelu || !input)
        return NULL;

    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Constants
    const float sqrt_2_pi = 0.7978845608f; // sqrt(2/π)
    Tensor* scaled        = NULL;
    Tensor* tanh_result   = NULL;
    Tensor* one_plus_tanh = NULL;
    Tensor* half          = NULL;
    Tensor* output        = NULL;

    int* input_shape    = input->shape;
    int input_ndim      = input->ndim;
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* ones = tensor_ones(input_shape, input_ndim, &config);
    if (!ones)
        return NULL;

    Tensor* half_const = tensor_ones(input_shape, input_ndim, &config);
    if (!half_const) {
        tensor_free(ones);
        return NULL;
    }
    float* half_data = (float*)tensor_data_ptr(half_const);
    if (half_data) {
        for (size_t i = 0; i < half_const->numel; i++) {
            half_data[i] = 0.5f;
        }
    }

    Tensor* sqrt_const = tensor_ones(input_shape, input_ndim, &config);
    if (!sqrt_const) {
        tensor_free(ones);
        tensor_free(half_const);
        return NULL;
    }
    float* sqrt_data = (float*)tensor_data_ptr(sqrt_const);
    if (sqrt_data) {
        for (size_t i = 0; i < sqrt_const->numel; i++) {
            sqrt_data[i] = sqrt_2_pi;
        }
    }

    scaled = tensor_mul(sqrt_const, input);
    tensor_free(sqrt_const);
    if (!scaled) {
        tensor_free(ones);
        tensor_free(half_const);
        return NULL;
    }

    tanh_result = tensor_tanh(scaled);
    tensor_free(scaled);
    if (!tanh_result) {
        tensor_free(ones);
        tensor_free(half_const);
        return NULL;
    }

    one_plus_tanh = tensor_add(ones, tanh_result);
    tensor_free(ones);
    tensor_free(tanh_result);
    if (!one_plus_tanh) {
        tensor_free(half_const);
        return NULL;
    }

    half = tensor_mul(half_const, one_plus_tanh);
    tensor_free(half_const);
    tensor_free(one_plus_tanh);
    if (!half)
        return NULL;

    output = tensor_mul(input, half);
    tensor_free(half);

    return output;
}

static void gelu_free(Module* module) { free(module); }

GELU* nn_gelu(bool approximate) {
    GELU* gelu = malloc(sizeof(GELU));
    if (!gelu)
        return NULL;

    if (module_init((Module*)gelu, "GELU", gelu_forward, gelu_free) != 0) {
        free(gelu);
        return NULL;
    }

    gelu->approximate = approximate;
    return gelu;
}

// Softmax Forward - Uses existing tensor_softmax function
static Tensor* softmax_forward(Module* module, Tensor* input) {
    Softmax* softmax = (Softmax*)module;

    if (!softmax || !input)
        return NULL;

    // Use existing tensor_softmax function which handles numerical stability
    return tensor_softmax(input, softmax->dim);
}

static void softmax_free(Module* module) { free(module); }

Softmax* nn_softmax(int dim) {
    Softmax* softmax = malloc(sizeof(Softmax));
    if (!softmax)
        return NULL;

    if (module_init((Module*)softmax, "Softmax", softmax_forward, softmax_free) != 0) {
        free(softmax);
        return NULL;
    }

    softmax->dim = dim;
    return softmax;
}

// LogSoftmax Forward - Log of softmax for numerical stability
static Tensor* log_softmax_forward(Module* module, Tensor* input) {
    LogSoftmax* log_softmax = (LogSoftmax*)module;

    if (!log_softmax || !input)
        return NULL;

    Tensor* softmax_result = tensor_softmax(input, log_softmax->dim);
    if (!softmax_result)
        return NULL;

    // Compute log of softmax
    Tensor* output = tensor_log(softmax_result);
    tensor_free(softmax_result);

    return output;
}

static void log_softmax_free(Module* module) { free(module); }

LogSoftmax* nn_log_softmax(int dim) {
    LogSoftmax* log_softmax = malloc(sizeof(LogSoftmax));
    if (!log_softmax)
        return NULL;

    if (module_init((Module*)log_softmax, "LogSoftmax", log_softmax_forward, log_softmax_free) !=
        0) {
        free(log_softmax);
        return NULL;
    }

    log_softmax->dim = dim;
    return log_softmax;
}

Tensor* f_relu(Tensor* input) {
    if (!input) {
        return NULL;
    }
    // Create a temporary ReLU module and use it
    Module* relu = (Module*)nn_relu(false);
    if (!relu) {
        return NULL;
    }
    Tensor* output = module_forward(relu, input);
    module_free(relu);
    return output;
}

Tensor* f_sigmoid(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* sigmoid = (Module*)nn_sigmoid();
    if (!sigmoid) {
        return NULL;
    }
    Tensor* output = module_forward(sigmoid, input);
    module_free(sigmoid);
    return output;
}

Tensor* f_tanh(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* tanh = (Module*)nn_tanh();
    if (!tanh) {
        return NULL;
    }
    Tensor* output = module_forward(tanh, input);
    module_free(tanh);
    return output;
}

Tensor* f_gelu(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* gelu = (Module*)nn_gelu(false);
    if (!gelu) {
        return NULL;
    }
    Tensor* output = module_forward(gelu, input);
    module_free(gelu);
    return output;
}
