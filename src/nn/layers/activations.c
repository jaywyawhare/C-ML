
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

static Tensor* relu_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
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

static Tensor* sigmoid_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
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

static Tensor* tanh_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
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

static Tensor* gelu_forward(Module* module, Tensor* input) {
    GELU* gelu = (GELU*)module;

    if (!gelu || !input)
        return NULL;
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

static Tensor* softmax_forward(Module* module, Tensor* input) {
    Softmax* softmax = (Softmax*)module;

    if (!softmax || !input)
        return NULL;

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

static Tensor* log_softmax_forward(Module* module, Tensor* input) {
    LogSoftmax* log_softmax = (LogSoftmax*)module;

    if (!log_softmax || !input)
        return NULL;

    Tensor* softmax_result = tensor_softmax(input, log_softmax->dim);
    if (!softmax_result)
        return NULL;

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

static Tensor* elu_forward(Module* module, Tensor* input) {
    ELU* elu = (ELU*)module;

    if (!elu || !input)
        return NULL;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* zeros = tensor_zeros(input->shape, input->ndim, &config);
    if (!zeros)
        return NULL;

    Tensor* cond = uop_cmplt(input, zeros);

    Tensor* exp_x         = uop_exp(input);
    Tensor* ones          = tensor_ones(input->shape, input->ndim, &config);
    Tensor* exp_minus_one = uop_sub(exp_x, ones);

    Tensor* alpha_tensor = tensor_full(input->shape, input->ndim, &config, elu->alpha);
    Tensor* neg_part     = uop_mul(alpha_tensor, exp_minus_one);

    WhereParams wp = {.cond = cond, .a = neg_part, .b = input};
    return uop_where(&wp);
}

static void elu_free(Module* module) { free(module); }

ELU* nn_elu(float alpha, bool inplace) {
    ELU* elu = malloc(sizeof(ELU));
    if (!elu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for ELU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)elu, "ELU", elu_forward, elu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ELU module", __FILE__, __LINE__,
                         __func__);
        free(elu);
        return NULL;
    }

    elu->alpha   = alpha;
    elu->inplace = inplace;
    return elu;
}

static Tensor* selu_forward(Module* module, Tensor* input) {
    (void)module;

    if (!input)
        return NULL;

    const float selu_lambda = 1.0507009873554804934f;
    const float selu_alpha  = 1.6732632423543772848f;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* zeros = tensor_zeros(input->shape, input->ndim, &config);
    if (!zeros)
        return NULL;

    Tensor* cond = uop_cmplt(input, zeros);

    Tensor* exp_x         = uop_exp(input);
    Tensor* ones          = tensor_ones(input->shape, input->ndim, &config);
    Tensor* exp_minus_one = uop_sub(exp_x, ones);

    Tensor* alpha_tensor = tensor_full(input->shape, input->ndim, &config, selu_alpha);
    Tensor* neg_part     = uop_mul(alpha_tensor, exp_minus_one);

    WhereParams wp     = {.cond = cond, .a = neg_part, .b = input};
    Tensor* elu_result = uop_where(&wp);

    Tensor* lambda_tensor = tensor_full(input->shape, input->ndim, &config, selu_lambda);
    return uop_mul(lambda_tensor, elu_result);
}

static void selu_free(Module* module) { free(module); }

SELU* nn_selu(void) {
    SELU* selu = malloc(sizeof(SELU));
    if (!selu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for SELU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)selu, "SELU", selu_forward, selu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize SELU module", __FILE__,
                         __LINE__, __func__);
        free(selu);
        return NULL;
    }

    return selu;
}

static Tensor* silu_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
    return uop_mul(input, uop_sigmoid(input));
}

static void silu_free(Module* module) { free(module); }

SiLU* nn_silu(void) {
    SiLU* silu = malloc(sizeof(SiLU));
    if (!silu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for SiLU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)silu, "SiLU", silu_forward, silu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize SiLU module", __FILE__,
                         __LINE__, __func__);
        free(silu);
        return NULL;
    }

    return silu;
}

static Tensor* mish_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* exp_x         = uop_exp(input);
    Tensor* ones          = tensor_ones(input->shape, input->ndim, &config);
    Tensor* ones_plus_exp = uop_add(ones, exp_x);
    Tensor* softplus      = uop_log(ones_plus_exp);
    Tensor* tanh_sp       = uop_tanh(softplus);
    return uop_mul(input, tanh_sp);
}

static void mish_free(Module* module) { free(module); }

Mish* nn_mish(void) {
    Mish* mish = malloc(sizeof(Mish));
    if (!mish) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Mish layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)mish, "Mish", mish_forward, mish_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Mish module", __FILE__,
                         __LINE__, __func__);
        free(mish);
        return NULL;
    }

    return mish;
}

static Tensor* hardswish_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* three = tensor_full(input->shape, input->ndim, &config, 3.0f);
    Tensor* six   = tensor_full(input->shape, input->ndim, &config, 6.0f);
    Tensor* zeros = tensor_zeros(input->shape, input->ndim, &config);

    Tensor* x_plus_3 = uop_add(input, three);

    Tensor* clamped_low = uop_max(x_plus_3, zeros);
    Tensor* cmp_six = uop_cmplt(clamped_low, six);
    WhereParams wp  = {.cond = cmp_six, .a = clamped_low, .b = six};
    Tensor* clamped = uop_where(&wp);

    Tensor* scaled = uop_div(clamped, six);
    return uop_mul(input, scaled);
}

static void hardswish_free(Module* module) { free(module); }

HardSwish* nn_hardswish(void) {
    HardSwish* hardswish = malloc(sizeof(HardSwish));
    if (!hardswish) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for HardSwish layer", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)hardswish, "HardSwish", hardswish_forward, hardswish_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize HardSwish module", __FILE__,
                         __LINE__, __func__);
        free(hardswish);
        return NULL;
    }

    return hardswish;
}

Tensor* f_elu(Tensor* input, float alpha) {
    if (!input) {
        return NULL;
    }
    Module* elu = (Module*)nn_elu(alpha, false);
    if (!elu) {
        return NULL;
    }
    Tensor* output = module_forward(elu, input);
    module_free(elu);
    return output;
}

Tensor* f_selu(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* selu = (Module*)nn_selu();
    if (!selu) {
        return NULL;
    }
    Tensor* output = module_forward(selu, input);
    module_free(selu);
    return output;
}

Tensor* f_silu(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* silu = (Module*)nn_silu();
    if (!silu) {
        return NULL;
    }
    Tensor* output = module_forward(silu, input);
    module_free(silu);
    return output;
}

Tensor* f_mish(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* mish = (Module*)nn_mish();
    if (!mish) {
        return NULL;
    }
    Tensor* output = module_forward(mish, input);
    module_free(mish);
    return output;
}

Tensor* f_hardswish(Tensor* input) {
    if (!input) {
        return NULL;
    }
    Module* hardswish = (Module*)nn_hardswish();
    if (!hardswish) {
        return NULL;
    }
    Tensor* output = module_forward(hardswish, input);
    module_free(hardswish);
    return output;
}
