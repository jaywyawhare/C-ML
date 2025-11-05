/**
 * @file activations.c
 * @brief Activation function layers implementation
 */

#include "nn/layers/activations.h"
#include "nn/module.h"
#include "autograd/autograd.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <math.h>

// ReLU Forward
static Tensor* relu_forward(Module* module, Tensor* input) {
    (void)module; // ReLU doesn't need module state
    return tensor_relu(input);
}

static void relu_free(Module* module) { CM_FREE(module); }

ReLU* nn_relu(bool inplace) {
    ReLU* relu = CM_MALLOC(sizeof(ReLU));
    if (!relu)
        return NULL;

    if (module_init((Module*)relu, "ReLU", relu_forward, relu_free) != 0) {
        CM_FREE(relu);
        return NULL;
    }

    relu->inplace = inplace;
    return relu;
}

// LeakyReLU Forward
static Tensor* leaky_relu_forward(Module* module, Tensor* input) {
    LeakyReLU* leaky_relu = (LeakyReLU*)module;
    return tensor_leaky_relu(input, leaky_relu->negative_slope);
}

static void leaky_relu_free(Module* module) { CM_FREE(module); }

LeakyReLU* nn_leaky_relu(float negative_slope, bool inplace) {
    LeakyReLU* leaky_relu = CM_MALLOC(sizeof(LeakyReLU));
    if (!leaky_relu)
        return NULL;

    if (module_init((Module*)leaky_relu, "LeakyReLU", leaky_relu_forward, leaky_relu_free) != 0) {
        CM_FREE(leaky_relu);
        return NULL;
    }

    leaky_relu->negative_slope = negative_slope;
    leaky_relu->inplace        = inplace;
    return leaky_relu;
}

// Sigmoid Forward
static Tensor* sigmoid_forward(Module* module, Tensor* input) {
    (void)module;
    return tensor_sigmoid(input);
}

static void sigmoid_free(Module* module) { CM_FREE(module); }

Sigmoid* nn_sigmoid(void) {
    Sigmoid* sigmoid = CM_MALLOC(sizeof(Sigmoid));
    if (!sigmoid)
        return NULL;

    if (module_init((Module*)sigmoid, "Sigmoid", sigmoid_forward, sigmoid_free) != 0) {
        CM_FREE(sigmoid);
        return NULL;
    }

    return sigmoid;
}

// Tanh Forward
static Tensor* tanh_forward(Module* module, Tensor* input) {
    (void)module;
    return tensor_tanh(input);
}

static void tanh_free(Module* module) { CM_FREE(module); }

Tanh* nn_tanh(void) {
    Tanh* tanh = CM_MALLOC(sizeof(Tanh));
    if (!tanh)
        return NULL;

    if (module_init((Module*)tanh, "Tanh", tanh_forward, tanh_free) != 0) {
        CM_FREE(tanh);
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

    int* input_shape = input->shape;
    int input_ndim   = input->ndim;
    Tensor* ones     = tensor_ones(input_shape, input_ndim, input->dtype, input->device);
    if (!ones)
        return NULL;

    Tensor* half_const = tensor_ones(input_shape, input_ndim, input->dtype, input->device);
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

    Tensor* sqrt_const = tensor_ones(input_shape, input_ndim, input->dtype, input->device);
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

static void gelu_free(Module* module) { CM_FREE(module); }

GELU* nn_gelu(bool approximate) {
    GELU* gelu = CM_MALLOC(sizeof(GELU));
    if (!gelu)
        return NULL;

    if (module_init((Module*)gelu, "GELU", gelu_forward, gelu_free) != 0) {
        CM_FREE(gelu);
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

static void softmax_free(Module* module) { CM_FREE(module); }

Softmax* nn_softmax(int dim) {
    Softmax* softmax = CM_MALLOC(sizeof(Softmax));
    if (!softmax)
        return NULL;

    if (module_init((Module*)softmax, "Softmax", softmax_forward, softmax_free) != 0) {
        CM_FREE(softmax);
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

static void log_softmax_free(Module* module) { CM_FREE(module); }

LogSoftmax* nn_log_softmax(int dim) {
    LogSoftmax* log_softmax = CM_MALLOC(sizeof(LogSoftmax));
    if (!log_softmax)
        return NULL;

    if (module_init((Module*)log_softmax, "LogSoftmax", log_softmax_forward, log_softmax_free) !=
        0) {
        CM_FREE(log_softmax);
        return NULL;
    }

    log_softmax->dim = dim;
    return log_softmax;
}

// ELU Forward
static Tensor* elu_forward(Module* module, Tensor* input) {
    ELU* elu = (ELU*)module;
    return tensor_elu(input, elu->alpha);
}

static void elu_free(Module* module) { CM_FREE(module); }

ELU* nn_elu(float alpha, bool inplace) {
    ELU* elu = CM_MALLOC(sizeof(ELU));
    if (!elu)
        return NULL;

    if (module_init((Module*)elu, "ELU", elu_forward, elu_free) != 0) {
        CM_FREE(elu);
        return NULL;
    }

    elu->alpha   = alpha > 0.0f ? alpha : 1.0f;
    elu->inplace = inplace;
    return elu;
}

// SELU Forward
static Tensor* selu_forward(Module* module, Tensor* input) {
    (void)module; // SELU doesn't need module state
    return tensor_selu(input);
}

static void selu_free(Module* module) { CM_FREE(module); }

SELU* nn_selu(bool inplace) {
    SELU* selu = CM_MALLOC(sizeof(SELU));
    if (!selu)
        return NULL;

    if (module_init((Module*)selu, "SELU", selu_forward, selu_free) != 0) {
        CM_FREE(selu);
        return NULL;
    }

    selu->inplace = inplace;
    return selu;
}

// Swish Forward
static Tensor* swish_forward(Module* module, Tensor* input) {
    (void)module; // Swish doesn't need module state
    return tensor_swish(input);
}

static void swish_free(Module* module) { CM_FREE(module); }

Swish* nn_swish(void) {
    Swish* swish = CM_MALLOC(sizeof(Swish));
    if (!swish)
        return NULL;

    if (module_init((Module*)swish, "Swish", swish_forward, swish_free) != 0) {
        CM_FREE(swish);
        return NULL;
    }

    return swish;
}

// Mish Forward
static Tensor* mish_forward(Module* module, Tensor* input) {
    (void)module; // Mish doesn't need module state
    return tensor_mish(input);
}

static void mish_free(Module* module) { CM_FREE(module); }

Mish* nn_mish(void) {
    Mish* mish = CM_MALLOC(sizeof(Mish));
    if (!mish)
        return NULL;

    if (module_init((Module*)mish, "Mish", mish_forward, mish_free) != 0) {
        CM_FREE(mish);
        return NULL;
    }

    return mish;
}

// Hard Swish Forward
static Tensor* hard_swish_forward(Module* module, Tensor* input) {
    (void)module;
    return tensor_hard_swish(input);
}

static void hard_swish_free(Module* module) { CM_FREE(module); }

HardSwish* nn_hard_swish(void) {
    HardSwish* hard_swish = CM_MALLOC(sizeof(HardSwish));
    if (!hard_swish)
        return NULL;

    if (module_init((Module*)hard_swish, "HardSwish", hard_swish_forward, hard_swish_free) != 0) {
        CM_FREE(hard_swish);
        return NULL;
    }

    return hard_swish;
}
