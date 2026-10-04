
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
#include "alloc/cml_allocator.h"

/** torch.nn.ReLU forward: elementwise max(0, x). */
static Tensor* relu_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
    return uop_relu(input);
}

/** Free the ReLU module (no owned parameters). */
static void relu_free(Module* module) { cml_free(module); }

/** Construct a ReLU layer. Returns NULL on allocation/init failure. */
ReLU* nn_relu(bool inplace) {
    ReLU* relu = cml_malloc(sizeof(ReLU));
    if (!relu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for ReLU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)relu, "ReLU", relu_forward, relu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ReLU module", __FILE__,
                         __LINE__, __func__);
        cml_free(relu);
        return NULL;
    }

    relu->inplace = inplace;
    return relu;
}

/** torch.nn.LeakyReLU forward: x for x >= 0, negative_slope * x below. */
static Tensor* leaky_relu_forward(Module* module, Tensor* input) {
    LeakyReLU* leaky_relu = (LeakyReLU*)module;

    return uop_leaky_relu(input, leaky_relu->negative_slope);
}

/** Free the LeakyReLU module (no owned parameters). */
static void leaky_relu_free(Module* module) { cml_free(module); }

/** Construct a LeakyReLU layer with the given negative_slope. NULL on failure. */
LeakyReLU* nn_leaky_relu(float negative_slope, bool inplace) {
    LeakyReLU* leaky_relu = cml_malloc(sizeof(LeakyReLU));
    if (!leaky_relu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for LeakyReLU layer", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)leaky_relu, "LeakyReLU", leaky_relu_forward, leaky_relu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize LeakyReLU module", __FILE__,
                         __LINE__, __func__);
        cml_free(leaky_relu);
        return NULL;
    }

    leaky_relu->negative_slope = negative_slope;
    leaky_relu->inplace        = inplace;
    return leaky_relu;
}

/** torch.nn.Sigmoid forward: elementwise 1/(1+exp(-x)). */
static Tensor* sigmoid_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
    return uop_sigmoid(input);
}

/** Free the Sigmoid module (no owned parameters). */
static void sigmoid_free(Module* module) { cml_free(module); }

/** Construct a Sigmoid layer. Returns NULL on failure. */
Sigmoid* nn_sigmoid(void) {
    Sigmoid* sigmoid = cml_malloc(sizeof(Sigmoid));
    if (!sigmoid) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Sigmoid layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)sigmoid, "Sigmoid", sigmoid_forward, sigmoid_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Sigmoid module", __FILE__,
                         __LINE__, __func__);
        cml_free(sigmoid);
        return NULL;
    }

    return sigmoid;
}

/** torch.nn.Tanh forward: elementwise hyperbolic tangent. */
static Tensor* tanh_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
    return uop_tanh(input);
}

/** Free the Tanh module (no owned parameters). */
static void tanh_free(Module* module) { cml_free(module); }

/** Construct a Tanh layer. Returns NULL on failure. */
Tanh* nn_tanh(void) {
    Tanh* tanh = cml_malloc(sizeof(Tanh));
    if (!tanh) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Tanh layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)tanh, "Tanh", tanh_forward, tanh_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Tanh module", __FILE__,
                         __LINE__, __func__);
        cml_free(tanh);
        return NULL;
    }

    return tanh;
}

/** torch.nn.GELU forward: the tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*x)). NULL on failure
 *  of any intermediate op. */
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

    /* Lazy constants (was: eager tensor_ones + tensor_data_ptr fill loop). */
    Tensor* half_const = uop_fill_ex(input_shape, input_ndim, 0.5f, input->dtype, input->device);
    if (!half_const) {
        tensor_free(ones);
        return NULL;
    }

    Tensor* sqrt_const =
        uop_fill_ex(input_shape, input_ndim, sqrt_2_pi, input->dtype, input->device);
    if (!sqrt_const) {
        tensor_free(ones);
        tensor_free(half_const);
        return NULL;
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

/** Free the GELU module (no owned parameters). */
static void gelu_free(Module* module) { cml_free(module); }

/** Construct a GELU layer. Returns NULL on failure. */
GELU* nn_gelu(bool approximate) {
    GELU* gelu = cml_malloc(sizeof(GELU));
    if (!gelu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for GELU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)gelu, "GELU", gelu_forward, gelu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize GELU module", __FILE__,
                         __LINE__, __func__);
        cml_free(gelu);
        return NULL;
    }

    gelu->approximate = approximate;
    return gelu;
}

/** torch.nn.Softmax forward: normalize to a probability distribution along dim. */
static Tensor* softmax_forward(Module* module, Tensor* input) {
    Softmax* softmax = (Softmax*)module;

    if (!softmax || !input)
        return NULL;

    return tensor_softmax(input, softmax->dim);
}

/** Free the Softmax module (no owned parameters). */
static void softmax_free(Module* module) { cml_free(module); }

/** Construct a Softmax layer over the given dim. Returns NULL on failure. */
Softmax* nn_softmax(int dim) {
    Softmax* softmax = cml_malloc(sizeof(Softmax));
    if (!softmax) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Softmax layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)softmax, "Softmax", softmax_forward, softmax_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Softmax module", __FILE__,
                         __LINE__, __func__);
        cml_free(softmax);
        return NULL;
    }

    softmax->dim = dim;
    return softmax;
}

/** torch.nn.LogSoftmax forward: log of the softmax along dim. NULL on op failure. */
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

/** Free the LogSoftmax module (no owned parameters). */
static void log_softmax_free(Module* module) { cml_free(module); }

/** Construct a LogSoftmax layer over the given dim. Returns NULL on failure. */
LogSoftmax* nn_log_softmax(int dim) {
    LogSoftmax* log_softmax = cml_malloc(sizeof(LogSoftmax));
    if (!log_softmax) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for LogSoftmax layer", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)log_softmax, "LogSoftmax", log_softmax_forward, log_softmax_free) !=
        0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize LogSoftmax module", __FILE__,
                         __LINE__, __func__);
        cml_free(log_softmax);
        return NULL;
    }

    log_softmax->dim = dim;
    return log_softmax;
}

/** Functional ReLU: build a transient ReLU module, run it, and free it. NULL on failure. */
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

/** Functional Sigmoid: transient module applied to input. NULL on failure. */
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

/** Functional Tanh: transient module applied to input. NULL on failure. */
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

/** Functional GELU: transient module applied to input. NULL on failure. */
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

/** The ELU curve: x where x >= 0, alpha * (exp(x) - 1) below. SELU is this
 *  same curve rescaled, so both share it. */
static Tensor* elu_curve(Tensor* input, float alpha) {
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* zeros = tensor_zeros(input->shape, input->ndim, &config);
    if (!zeros)
        return NULL;

    Tensor* cond          = uop_cmplt(input, zeros);
    Tensor* ones          = tensor_ones(input->shape, input->ndim, &config);
    Tensor* exp_minus_one = uop_sub(uop_exp(input), ones);
    Tensor* alpha_tensor  = tensor_full(input->shape, input->ndim, &config, alpha);
    Tensor* neg_part      = uop_mul(alpha_tensor, exp_minus_one);

    WhereParams wp = {.cond = cond, .a = neg_part, .b = input};
    return uop_where(&wp);
}

/** torch.nn.ELU forward: the ELU curve with the layer's alpha. */
static Tensor* elu_forward(Module* module, Tensor* input) {
    ELU* elu = (ELU*)module;

    if (!elu || !input)
        return NULL;

    return elu_curve(input, elu->alpha);
}

/** Free the ELU module (no owned parameters). */
static void elu_free(Module* module) { cml_free(module); }

/** Construct an ELU layer with the given alpha. Returns NULL on failure. */
ELU* nn_elu(float alpha, bool inplace) {
    ELU* elu = cml_malloc(sizeof(ELU));
    if (!elu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for ELU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)elu, "ELU", elu_forward, elu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ELU module", __FILE__, __LINE__,
                         __func__);
        cml_free(elu);
        return NULL;
    }

    elu->alpha   = alpha;
    elu->inplace = inplace;
    return elu;
}

/** torch.nn.SELU forward: scaled ELU, lambda * elu_curve(x, alpha) with the SELU constants. */
static Tensor* selu_forward(Module* module, Tensor* input) {
    (void)module;

    if (!input)
        return NULL;

    const float selu_lambda = 1.0507009873554804934f;
    const float selu_alpha  = 1.6732632423543772848f;

    Tensor* elu_result = elu_curve(input, selu_alpha);

    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* lambda_tensor = tensor_full(input->shape, input->ndim, &config, selu_lambda);
    return uop_mul(lambda_tensor, elu_result);
}

/** Free the SELU module (no owned parameters). */
static void selu_free(Module* module) { cml_free(module); }

/** Construct a SELU layer. Returns NULL on failure. */
SELU* nn_selu(void) {
    SELU* selu = cml_malloc(sizeof(SELU));
    if (!selu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for SELU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)selu, "SELU", selu_forward, selu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize SELU module", __FILE__,
                         __LINE__, __func__);
        cml_free(selu);
        return NULL;
    }

    return selu;
}

/** torch.nn.SiLU/Swish forward: x * sigmoid(x). */
static Tensor* silu_forward(Module* module, Tensor* input) {
    (void)module;
    if (!input)
        return NULL;
    return uop_mul(input, uop_sigmoid(input));
}

/** Free the SiLU module (no owned parameters). */
static void silu_free(Module* module) { cml_free(module); }

/** Construct a SiLU layer. Returns NULL on failure. */
SiLU* nn_silu(void) {
    SiLU* silu = cml_malloc(sizeof(SiLU));
    if (!silu) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for SiLU layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)silu, "SiLU", silu_forward, silu_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize SiLU module", __FILE__,
                         __LINE__, __func__);
        cml_free(silu);
        return NULL;
    }

    return silu;
}

/** torch.nn.Mish forward: x * tanh(softplus(x)), softplus = log(1 + exp(x)). */
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

/** Free the Mish module (no owned parameters). */
static void mish_free(Module* module) { cml_free(module); }

/** Construct a Mish layer. Returns NULL on failure. */
Mish* nn_mish(void) {
    Mish* mish = cml_malloc(sizeof(Mish));
    if (!mish) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Mish layer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)mish, "Mish", mish_forward, mish_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Mish module", __FILE__,
                         __LINE__, __func__);
        cml_free(mish);
        return NULL;
    }

    return mish;
}

/** torch.nn.Hardswish forward: x * clamp(x + 3, 0, 6) / 6. */
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
    Tensor* cmp_six     = uop_cmplt(clamped_low, six);
    WhereParams wp      = {.cond = cmp_six, .a = clamped_low, .b = six};
    Tensor* clamped     = uop_where(&wp);

    Tensor* scaled = uop_div(clamped, six);
    return uop_mul(input, scaled);
}

/** Free the HardSwish module (no owned parameters). */
static void hardswish_free(Module* module) { cml_free(module); }

/** Construct a HardSwish layer. Returns NULL on failure. */
HardSwish* nn_hardswish(void) {
    HardSwish* hardswish = cml_malloc(sizeof(HardSwish));
    if (!hardswish) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for HardSwish layer", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)hardswish, "HardSwish", hardswish_forward, hardswish_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize HardSwish module", __FILE__,
                         __LINE__, __func__);
        cml_free(hardswish);
        return NULL;
    }

    return hardswish;
}

/** Functional ELU: transient module applied to input with the given alpha. NULL on failure. */
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

/** Functional SELU: transient module applied to input. NULL on failure. */
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

/** Functional SiLU: transient module applied to input. NULL on failure. */
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

/** Functional Mish: transient module applied to input. NULL on failure. */
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

/** Functional HardSwish: transient module applied to input. NULL on failure. */
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
