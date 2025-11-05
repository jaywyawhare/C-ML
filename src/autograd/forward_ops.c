#include "autograd/autograd.h"
#include <math.h>
#include <string.h>
#include <float.h>

// External backward function declarations
extern void add_backward(Function* fn, Tensor* grad_output);
extern void sub_backward(Function* fn, Tensor* grad_output);
extern void mul_backward(Function* fn, Tensor* grad_output);
extern void div_backward(Function* fn, Tensor* grad_output);
extern void pow_backward(Function* fn, Tensor* grad_output);
extern void neg_backward(Function* fn, Tensor* grad_output);
extern void exp_backward(Function* fn, Tensor* grad_output);
extern void log_backward(Function* fn, Tensor* grad_output);
extern void sqrt_backward(Function* fn, Tensor* grad_output);
extern void sin_backward(Function* fn, Tensor* grad_output);
extern void cos_backward(Function* fn, Tensor* grad_output);
extern void tan_backward(Function* fn, Tensor* grad_output);
extern void tanh_backward(Function* fn, Tensor* grad_output);
extern void relu_backward(Function* fn, Tensor* grad_output);
extern void sigmoid_backward(Function* fn, Tensor* grad_output);
extern void leaky_relu_backward(Function* fn, Tensor* grad_output);
extern void elu_backward(Function* fn, Tensor* grad_output);
extern void selu_backward(Function* fn, Tensor* grad_output);
extern void swish_backward(Function* fn, Tensor* grad_output);
extern void mish_backward(Function* fn, Tensor* grad_output);
extern void hard_swish_backward(Function* fn, Tensor* grad_output);
extern void sum_backward(Function* fn, Tensor* grad_output);
extern void mean_backward(Function* fn, Tensor* grad_output);
extern void transpose_backward(Function* fn, Tensor* grad_output);
extern void matmul_backward(Function* fn, Tensor* grad_output);

// Helper: Create output tensor and setup grad_fn

static Tensor* create_output_with_grad_fn(Tensor* result, Function* fn, Tensor** inputs,
                                          int num_inputs) {
    if (!result || !fn)
        return result;

    // Only set up autograd if gradient tracking is enabled
    if (!autograd_is_grad_enabled()) {
        autograd_function_free(fn);
        return result;
    }

    // Check if any input requires gradients
    bool any_requires_grad = false;
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->requires_grad) {
            any_requires_grad = true;
            break;
        }
    }

    if (!any_requires_grad) {
        autograd_function_free(fn);
        return result;
    }

    // Set up inputs and grad_fn
    autograd_function_set_inputs(fn, inputs, num_inputs);
    result->grad_fn       = fn;
    result->requires_grad = true;

    LOG_DEBUG("Created output tensor with grad_fn (op=%s)", op_type_to_string(fn->op_type));

    return result;
}

// Binary Operations with Broadcasting

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_add");
        return NULL;
    }

    LOG_DEBUG("Computing Add: tensor %p (%dx%d...) + tensor %p (%dx%d...)", (void*)a,
              a->ndim > 0 ? a->shape[0] : 0, a->ndim > 1 ? a->shape[1] : 0, (void*)b,
              b->ndim > 0 ? b->shape[0] : 0, b->ndim > 1 ? b->shape[1] : 0);

    // Check if shapes can be broadcast
    if (!can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast together for Add");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // Create result tensor
    Tensor* result = tensor_empty(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Perform broadcasted addition
    float* result_data = (float*)result->data;
    float* a_data      = (float*)a->data;
    float* b_data      = (float*)b->data;

    for (size_t i = 0; i < result->numel; i++) {
        // Calculate indices for a and b with broadcasting
        size_t a_idx = 0, b_idx = 0;
        size_t temp = i;

        for (int d = result->ndim - 1; d >= 0; d--) {
            size_t coord = temp % result->shape[d];
            temp /= result->shape[d];

            // Map to a's dimensions
            int a_d = d - (result->ndim - a->ndim);
            if (a_d >= 0 && a->shape[a_d] > 1) {
                size_t a_stride = 1;
                for (int j = a_d + 1; j < a->ndim; j++)
                    a_stride *= a->shape[j];
                a_idx += (coord % a->shape[a_d]) * a_stride;
            }

            // Map to b's dimensions
            int b_d = d - (result->ndim - b->ndim);
            if (b_d >= 0 && b->shape[b_d] > 1) {
                size_t b_stride = 1;
                for (int j = b_d + 1; j < b->ndim; j++)
                    b_stride *= b->shape[j];
                b_idx += (coord % b->shape[b_d]) * b_stride;
            }
        }

        result_data[i] = a_data[a_idx] + b_data[b_idx];
    }

    // Set up autograd
    Function* fn = autograd_function_create(OP_ADD, "Add");
    if (fn) {
        autograd_function_set_backward(fn, add_backward);
        Tensor* inputs[] = {a, b};
        result           = create_output_with_grad_fn(result, fn, inputs, 2);
    }

    return result;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_sub");
        return NULL;
    }

    LOG_DEBUG("Computing Sub with broadcasting");

    // Check if shapes can be broadcast
    if (!can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast together for Sub");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // Create result tensor
    Tensor* result = tensor_empty(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Perform broadcasted subtraction
    float* result_data = (float*)result->data;
    float* a_data      = (float*)a->data;
    float* b_data      = (float*)b->data;

    for (size_t i = 0; i < result->numel; i++) {
        size_t a_idx = 0, b_idx = 0;
        size_t temp = i;

        for (int d = result->ndim - 1; d >= 0; d--) {
            size_t coord = temp % result->shape[d];
            temp /= result->shape[d];

            int a_d = d - (result->ndim - a->ndim);
            if (a_d >= 0 && a->shape[a_d] > 1) {
                size_t a_stride = 1;
                for (int j = a_d + 1; j < a->ndim; j++)
                    a_stride *= a->shape[j];
                a_idx += (coord % a->shape[a_d]) * a_stride;
            }

            int b_d = d - (result->ndim - b->ndim);
            if (b_d >= 0 && b->shape[b_d] > 1) {
                size_t b_stride = 1;
                for (int j = b_d + 1; j < b->ndim; j++)
                    b_stride *= b->shape[j];
                b_idx += (coord % b->shape[b_d]) * b_stride;
            }
        }

        result_data[i] = a_data[a_idx] - b_data[b_idx];
    }

    Function* fn = autograd_function_create(OP_SUB, "Sub");
    if (fn) {
        autograd_function_set_backward(fn, sub_backward);
        Tensor* inputs[] = {a, b};
        result           = create_output_with_grad_fn(result, fn, inputs, 2);
    }

    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_mul");
        return NULL;
    }

    LOG_DEBUG("Computing Mul with broadcasting");

    // Check if shapes can be broadcast
    if (!can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast together for Mul");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // Create result tensor
    Tensor* result = tensor_empty(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Perform broadcasted multiplication
    float* result_data = (float*)result->data;
    float* a_data      = (float*)a->data;
    float* b_data      = (float*)b->data;

    for (size_t i = 0; i < result->numel; i++) {
        size_t a_idx = 0, b_idx = 0;
        size_t temp = i;

        for (int d = result->ndim - 1; d >= 0; d--) {
            size_t coord = temp % result->shape[d];
            temp /= result->shape[d];

            int a_d = d - (result->ndim - a->ndim);
            if (a_d >= 0 && a->shape[a_d] > 1) {
                size_t a_stride = 1;
                for (int j = a_d + 1; j < a->ndim; j++)
                    a_stride *= a->shape[j];
                a_idx += (coord % a->shape[a_d]) * a_stride;
            }

            int b_d = d - (result->ndim - b->ndim);
            if (b_d >= 0 && b->shape[b_d] > 1) {
                size_t b_stride = 1;
                for (int j = b_d + 1; j < b->ndim; j++)
                    b_stride *= b->shape[j];
                b_idx += (coord % b->shape[b_d]) * b_stride;
            }
        }

        result_data[i] = a_data[a_idx] * b_data[b_idx];
    }

    Function* fn = autograd_function_create(OP_MUL, "Mul");
    if (fn) {
        // Save inputs for backward pass
        Tensor* saved[] = {a, b};
        autograd_context_save_for_backward(fn->ctx, saved, 2);
        autograd_function_set_backward(fn, mul_backward);
        Tensor* inputs[] = {a, b};
        result           = create_output_with_grad_fn(result, fn, inputs, 2);
    }

    return result;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_div");
        return NULL;
    }

    LOG_DEBUG("Computing Div with broadcasting");

    // Check if shapes can be broadcast
    if (!can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast together for Div");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // Create result tensor
    Tensor* result = tensor_empty(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Perform broadcasted division
    float* result_data = (float*)result->data;
    float* a_data      = (float*)a->data;
    float* b_data      = (float*)b->data;

    for (size_t i = 0; i < result->numel; i++) {
        size_t a_idx = 0, b_idx = 0;
        size_t temp = i;

        for (int d = result->ndim - 1; d >= 0; d--) {
            size_t coord = temp % result->shape[d];
            temp /= result->shape[d];

            int a_d = d - (result->ndim - a->ndim);
            if (a_d >= 0 && a->shape[a_d] > 1) {
                size_t a_stride = 1;
                for (int j = a_d + 1; j < a->ndim; j++)
                    a_stride *= a->shape[j];
                a_idx += (coord % a->shape[a_d]) * a_stride;
            }

            int b_d = d - (result->ndim - b->ndim);
            if (b_d >= 0 && b->shape[b_d] > 1) {
                size_t b_stride = 1;
                for (int j = b_d + 1; j < b->ndim; j++)
                    b_stride *= b->shape[j];
                b_idx += (coord % b->shape[b_d]) * b_stride;
            }
        }

        float divisor = b_data[b_idx];
        if (divisor == 0.0f) {
            LOG_ERROR("Division by zero at output index %zu", i);
            tensor_free(result);
            return NULL;
        }
        result_data[i] = a_data[a_idx] / divisor;
    }

    Function* fn = autograd_function_create(OP_DIV, "Div");
    if (fn) {
        Tensor* saved[] = {a, b};
        autograd_context_save_for_backward(fn->ctx, saved, 2);
        autograd_function_set_backward(fn, div_backward);
        Tensor* inputs[] = {a, b};
        result           = create_output_with_grad_fn(result, fn, inputs, 2);
    }

    return result;
}

Tensor* tensor_pow(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_pow");
        return NULL;
    }

    LOG_DEBUG("Computing Pow with broadcasting");

    // Check if shapes can be broadcast
    if (!can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast together for Pow");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // Create result tensor
    Tensor* result = tensor_empty(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Perform broadcasted power
    float* result_data = (float*)result->data;
    float* a_data      = (float*)a->data;
    float* b_data      = (float*)b->data;

    for (size_t i = 0; i < result->numel; i++) {
        size_t a_idx = 0, b_idx = 0;
        size_t temp = i;

        for (int d = result->ndim - 1; d >= 0; d--) {
            size_t coord = temp % result->shape[d];
            temp /= result->shape[d];

            int a_d = d - (result->ndim - a->ndim);
            if (a_d >= 0 && a->shape[a_d] > 1) {
                size_t a_stride = 1;
                for (int j = a_d + 1; j < a->ndim; j++)
                    a_stride *= a->shape[j];
                a_idx += (coord % a->shape[a_d]) * a_stride;
            }

            int b_d = d - (result->ndim - b->ndim);
            if (b_d >= 0 && b->shape[b_d] > 1) {
                size_t b_stride = 1;
                for (int j = b_d + 1; j < b->ndim; j++)
                    b_stride *= b->shape[j];
                b_idx += (coord % b->shape[b_d]) * b_stride;
            }
        }

        result_data[i] = powf(a_data[a_idx], b_data[b_idx]);
    }

    Function* fn = autograd_function_create(OP_POW, "Pow");
    if (fn) {
        Tensor* saved[] = {a, b};
        autograd_context_save_for_backward(fn->ctx, saved, 2);
        autograd_function_set_backward(fn, pow_backward);
        Tensor* inputs[] = {a, b};
        result           = create_output_with_grad_fn(result, fn, inputs, 2);
    }

    return result;
}

// Unary Operations

Tensor* tensor_neg(Tensor* a) {
    if (!a) {
        LOG_ERROR("NULL tensor input to tensor_neg");
        return NULL;
    }

    LOG_DEBUG("Computing Neg: -tensor %p", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        tensor_set_float(result, i, -tensor_get_float(a, i));
    }

    Function* fn = autograd_function_create(OP_NEG, "Neg");
    if (fn) {
        autograd_function_set_backward(fn, neg_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_exp(Tensor* a) {
    if (!a) {
        LOG_ERROR("NULL tensor input to tensor_exp");
        return NULL;
    }

    LOG_DEBUG("Computing Exp: exp(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        tensor_set_float(result, i, expf(tensor_get_float(a, i)));
    }

    Function* fn = autograd_function_create(OP_EXP, "Exp");
    if (fn) {
        // Save output for backward
        Tensor* saved[] = {result};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, exp_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_log(Tensor* a) {
    if (!a) {
        LOG_ERROR("NULL tensor input to tensor_log");
        return NULL;
    }

    LOG_DEBUG("Computing Log: log(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val = tensor_get_float(a, i);
        if (val <= 0.0f) {
            LOG_ERROR("Log of non-positive value at index %zu", i);
            tensor_free(result);
            return NULL;
        }
        tensor_set_float(result, i, logf(val));
    }

    Function* fn = autograd_function_create(OP_LOG, "Log");
    if (fn) {
        Tensor* saved[] = {a};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, log_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_sqrt(Tensor* a) {
    if (!a) {
        LOG_ERROR("NULL tensor input to tensor_sqrt");
        return NULL;
    }

    LOG_DEBUG("Computing Sqrt: sqrt(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val = tensor_get_float(a, i);
        if (val < 0.0f) {
            LOG_ERROR("Sqrt of negative value at index %zu", i);
            tensor_free(result);
            return NULL;
        }
        tensor_set_float(result, i, sqrtf(val));
    }

    Function* fn = autograd_function_create(OP_SQRT, "Sqrt");
    if (fn) {
        Tensor* saved[] = {result};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, sqrt_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_sin(Tensor* a) {
    if (!a)
        return NULL;

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        tensor_set_float(result, i, sinf(tensor_get_float(a, i)));
    }

    Function* fn = autograd_function_create(OP_SIN, "Sin");
    if (fn) {
        Tensor* saved[] = {a};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, sin_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_cos(Tensor* a) {
    if (!a)
        return NULL;

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        tensor_set_float(result, i, cosf(tensor_get_float(a, i)));
    }

    Function* fn = autograd_function_create(OP_COS, "Cos");
    if (fn) {
        Tensor* saved[] = {a};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, cos_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_tan(Tensor* a) {
    if (!a)
        return NULL;

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        tensor_set_float(result, i, tanf(tensor_get_float(a, i)));
    }

    Function* fn = autograd_function_create(OP_TAN, "Tan");
    if (fn) {
        Tensor* saved[] = {a};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, tan_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_tanh(Tensor* a) {
    if (!a)
        return NULL;

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        tensor_set_float(result, i, tanhf(tensor_get_float(a, i)));
    }

    Function* fn = autograd_function_create(OP_TANH, "Tanh");
    if (fn) {
        Tensor* saved[] = {result};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, tanh_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

// Activation Functions

Tensor* tensor_relu(Tensor* a) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing ReLU: relu(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val = tensor_get_float(a, i);
        tensor_set_float(result, i, val > 0 ? val : 0.0f);
    }

    Function* fn = autograd_function_create(OP_RELU, "ReLU");
    if (fn) {
        Tensor* saved[] = {a};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, relu_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_sigmoid(Tensor* a) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Sigmoid: sigmoid(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val = tensor_get_float(a, i);
        tensor_set_float(result, i, 1.0f / (1.0f + expf(-val)));
    }

    Function* fn = autograd_function_create(OP_SIGMOID, "Sigmoid");
    if (fn) {
        Tensor* saved[] = {result};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_function_set_backward(fn, sigmoid_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_leaky_relu(Tensor* a, float negative_slope) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Leaky ReLU: leaky_relu(tensor %p, alpha=%.3f)", (void*)a, negative_slope);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val = tensor_get_float(a, i);
        tensor_set_float(result, i, val > 0 ? val : negative_slope * val);
    }

    Function* fn = autograd_function_create(OP_LEAKY_RELU, "LeakyReLU");
    if (fn) {
        Tensor* saved[] = {a};
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        autograd_context_save_data(fn->ctx, &negative_slope, sizeof(float));
        autograd_function_set_backward(fn, leaky_relu_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

// ELU: Exponential Linear Unit
// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
Tensor* tensor_elu(Tensor* a, float alpha) {
    if (!a)
        return NULL;

    if (alpha <= 0.0f)
        alpha = 1.0f; // Default alpha

    LOG_DEBUG("Computing ELU: elu(tensor %p, alpha=%.3f)", (void*)a, alpha);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val     = tensor_get_float(a, i);
        float elu_val = val > 0.0f ? val : alpha * (expf(val) - 1.0f);
        tensor_set_float(result, i, elu_val);
    }

    if (autograd_is_grad_enabled() && a->requires_grad) {
        Function* fn = autograd_function_create(OP_ELU, "ELU");
        if (fn) {
            Tensor* saved[] = {a};
            autograd_context_save_for_backward(fn->ctx, saved, 1);
            float* alpha_ptr = CM_MALLOC(sizeof(float));
            if (alpha_ptr) {
                *alpha_ptr = alpha;
                autograd_context_save_data(fn->ctx, alpha_ptr, sizeof(float));
            }
            autograd_function_set_backward(fn, elu_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }
    }

    return result;
}

// SELU: Scaled Exponential Linear Unit
// SELU(x) = scale * ELU(x) where scale = 1.0507, alpha = 1.6733
Tensor* tensor_selu(Tensor* a) {
    if (!a)
        return NULL;

    const float scale = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    LOG_DEBUG("Computing SELU: selu(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val      = tensor_get_float(a, i);
        float selu_val = val > 0.0f ? scale * val : scale * alpha * (expf(val) - 1.0f);
        tensor_set_float(result, i, selu_val);
    }

    if (autograd_is_grad_enabled() && a->requires_grad) {
        Function* fn = autograd_function_create(OP_SELU, "SELU");
        if (fn) {
            Tensor* saved[] = {a};
            autograd_context_save_for_backward(fn->ctx, saved, 1);
            autograd_function_set_backward(fn, selu_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }
    }

    return result;
}

// Swish: x * sigmoid(x)
Tensor* tensor_swish(Tensor* a) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Swish: swish(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val         = tensor_get_float(a, i);
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        tensor_set_float(result, i, val * sigmoid_val);
    }

    if (autograd_is_grad_enabled() && a->requires_grad) {
        Function* fn = autograd_function_create(OP_SWISH, "Swish");
        if (fn) {
            Tensor* saved[] = {a, result};
            autograd_context_save_for_backward(fn->ctx, saved, 2);
            autograd_function_set_backward(fn, swish_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }
    }

    return result;
}

// Mish: x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
Tensor* tensor_mish(Tensor* a) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Mish: mish(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val           = tensor_get_float(a, i);
        float softplus      = logf(1.0f + expf(val));
        float tanh_softplus = tanhf(softplus);
        tensor_set_float(result, i, val * tanh_softplus);
    }

    if (autograd_is_grad_enabled() && a->requires_grad) {
        Function* fn = autograd_function_create(OP_MISH, "Mish");
        if (fn) {
            Tensor* saved[] = {a};
            autograd_context_save_for_backward(fn->ctx, saved, 1);
            autograd_function_set_backward(fn, mish_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }
    }

    return result;
}

// Hard Swish: x * ReLU6(x + 3) / 6 where ReLU6(x) = min(max(x, 0), 6)
Tensor* tensor_hard_swish(Tensor* a) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Hard Swish: hard_swish(tensor %p)", (void*)a);

    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->numel; i++) {
        float val   = tensor_get_float(a, i);
        float relu6 = val + 3.0f;
        if (relu6 < 0.0f)
            relu6 = 0.0f;
        if (relu6 > 6.0f)
            relu6 = 6.0f;
        tensor_set_float(result, i, val * relu6 / 6.0f);
    }

    if (autograd_is_grad_enabled() && a->requires_grad) {
        Function* fn = autograd_function_create(OP_HARD_SWISH, "HardSwish");
        if (fn) {
            Tensor* saved[] = {a};
            autograd_context_save_for_backward(fn->ctx, saved, 1);
            autograd_function_set_backward(fn, hard_swish_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }
    }

    return result;
}

Tensor* tensor_softmax(Tensor* a, int dim) {
    if (!a)
        return NULL;

    // Normalize dimension (support negative indexing)
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = a->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= a->ndim) {
        LOG_ERROR("Softmax: dimension %d out of range for %dD tensor", dim, a->ndim);
        return NULL;
    }

    LOG_DEBUG("Computing Softmax: softmax(tensor %p, dim=%d)", (void*)a, normalized_dim);

    // Create output tensor with same shape
    Tensor* result = tensor_empty(a->shape, a->ndim, a->dtype, a->device);
    if (!result)
        return NULL;

    // Calculate sizes for iteration along the specified dimension
    size_t dim_size   = a->shape[normalized_dim];
    size_t inner_size = 1;
    for (int i = normalized_dim + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }
    size_t outer_size = a->numel / (dim_size * inner_size);

    // Compute softmax along the specified dimension
    // For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {
            // Find max along the dimension for numerical stability
            float max_val = -INFINITY;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                float val  = tensor_get_float(a, idx);
                if (val > max_val)
                    max_val = val;
            }

            // Compute exp(x - max) and sum
            float sum_exp     = 0.0f;
            float* exp_values = CM_MALLOC(dim_size * sizeof(float));
            if (!exp_values) {
                tensor_free(result);
                return NULL;
            }

            for (size_t d = 0; d < dim_size; d++) {
                size_t idx    = outer * (dim_size * inner_size) + d * inner_size + inner;
                float val     = tensor_get_float(a, idx);
                float exp_val = expf(val - max_val);
                exp_values[d] = exp_val;
                sum_exp += exp_val;
            }

            // Compute softmax: exp(x - max) / sum(exp(x - max))
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx        = outer * (dim_size * inner_size) + d * inner_size + inner;
                float softmax_val = exp_values[d] / sum_exp;
                tensor_set_float(result, idx, softmax_val);
            }

            CM_FREE(exp_values);
        }
    }

    // Setup autograd (basic - backward function can be added later if needed)
    Function* fn = autograd_function_create(OP_SOFTMAX, "Softmax");
    if (fn) {
        Tensor* saved[] = {result}; // Save output for backward
        autograd_context_save_for_backward(fn->ctx, saved, 1);
        // Save dimension
        int* dim_data = CM_MALLOC(sizeof(int));
        if (dim_data) {
            dim_data[0] = normalized_dim;
            autograd_context_save_data(fn->ctx, dim_data, sizeof(int));
        }
        // Note: backward function can be added later if needed
        // For now, we'll skip backward to avoid linker errors
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

// Reduction Operations

Tensor* tensor_sum(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Sum: sum(tensor %p, dim=%d, keepdim=%d)", (void*)a, dim, keepdim);

    // If dim is -1 or invalid, sum all elements
    if (dim < 0 || dim >= a->ndim) {
        int shape[]    = {1};
        Tensor* result = tensor_empty(shape, 1, a->dtype, a->device);
        if (!result)
            return NULL;

        float sum = 0.0f;
        for (size_t i = 0; i < a->numel; i++) {
            sum += tensor_get_float(a, i);
        }
        tensor_set_float(result, 0, sum);

        Function* fn = autograd_function_create(OP_SUM, "Sum");
        if (fn) {
            autograd_context_save_shape(fn->ctx, a->shape, a->ndim);
            autograd_function_set_backward(fn, sum_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }

        return result;
    }

    // Compute output shape (reduce along specified dimension)
    int* out_shape = CM_MALLOC(a->ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    int out_ndim = keepdim ? a->ndim : (a->ndim - 1);
    int out_idx  = 0;

    if (keepdim) {
        for (int i = 0; i < a->ndim; i++) {
            out_shape[i] = (i == dim) ? 1 : a->shape[i];
        }
    } else {
        for (int i = 0; i < a->ndim; i++) {
            if (i != dim) {
                out_shape[out_idx++] = a->shape[i];
            }
        }
    }

    // Handle scalar result
    if (out_ndim == 0) {
        out_ndim     = 1;
        out_shape[0] = 1;
    }

    // Create output tensor
    Tensor* result = tensor_zeros(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Compute sum along dimension
    float* a_data      = (float*)a->data;
    float* result_data = (float*)result->data;

    // Calculate strides for iteration
    size_t dim_size   = a->shape[dim];
    size_t inner_size = 1;
    for (int i = dim + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }
    size_t outer_size = a->numel / (dim_size * inner_size);

    // Perform reduction
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {
            float sum = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                sum += a_data[idx];
            }
            size_t out_idx       = outer * inner_size + inner;
            result_data[out_idx] = sum;
        }
    }

    // Setup autograd
    Function* fn = autograd_function_create(OP_SUM, "Sum");
    if (fn) {
        autograd_context_save_shape(fn->ctx, a->shape, a->ndim);
        // Store dim and keepdim in saved_data
        int* dim_data = CM_MALLOC(2 * sizeof(int));
        dim_data[0]   = dim;
        dim_data[1]   = keepdim ? 1 : 0;
        autograd_context_save_data(fn->ctx, dim_data, 2 * sizeof(int));

        autograd_function_set_backward(fn, sum_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_mean(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;

    LOG_DEBUG("Computing Mean: mean(tensor %p, dim=%d, keepdim=%d)", (void*)a, dim, keepdim);

    // If dim is -1 or invalid, mean all elements
    if (dim < 0 || dim >= a->ndim) {
        int shape[]    = {1};
        Tensor* result = tensor_empty(shape, 1, a->dtype, a->device);
        if (!result)
            return NULL;

        float sum = 0.0f;
        for (size_t i = 0; i < a->numel; i++) {
            sum += tensor_get_float(a, i);
        }
        tensor_set_float(result, 0, sum / (float)a->numel);

        Function* fn = autograd_function_create(OP_MEAN, "Mean");
        if (fn) {
            autograd_context_save_shape(fn->ctx, a->shape, a->ndim);
            autograd_function_set_backward(fn, mean_backward);
            Tensor* inputs[] = {a};
            result           = create_output_with_grad_fn(result, fn, inputs, 1);
        }

        return result;
    }

    // Compute output shape (reduce along specified dimension)
    int* out_shape = CM_MALLOC(a->ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    int out_ndim = keepdim ? a->ndim : (a->ndim - 1);
    int out_idx  = 0;

    if (keepdim) {
        for (int i = 0; i < a->ndim; i++) {
            out_shape[i] = (i == dim) ? 1 : a->shape[i];
        }
    } else {
        for (int i = 0; i < a->ndim; i++) {
            if (i != dim) {
                out_shape[out_idx++] = a->shape[i];
            }
        }
    }

    // Handle scalar result
    if (out_ndim == 0) {
        out_ndim     = 1;
        out_shape[0] = 1;
    }

    // Create output tensor
    Tensor* result = tensor_zeros(out_shape, out_ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Compute mean along dimension
    float* a_data      = (float*)a->data;
    float* result_data = (float*)result->data;

    // Calculate strides for iteration
    size_t dim_size   = a->shape[dim];
    size_t inner_size = 1;
    for (int i = dim + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }
    size_t outer_size = a->numel / (dim_size * inner_size);
    float div_factor  = (float)dim_size;

    // Perform reduction
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {
            float sum = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                sum += a_data[idx];
            }
            size_t out_idx       = outer * inner_size + inner;
            result_data[out_idx] = sum / div_factor;
        }
    }

    // Setup autograd
    Function* fn = autograd_function_create(OP_MEAN, "Mean");
    if (fn) {
        autograd_context_save_shape(fn->ctx, a->shape, a->ndim);
        // Store dim and keepdim in saved_data
        int* dim_data = CM_MALLOC(2 * sizeof(int));
        dim_data[0]   = dim;
        dim_data[1]   = keepdim ? 1 : 0;
        autograd_context_save_data(fn->ctx, dim_data, 2 * sizeof(int));

        autograd_function_set_backward(fn, mean_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

// Matrix Operations

Tensor* tensor_transpose(Tensor* a, int dim0, int dim1) {
    if (!a)
        return NULL;

    // Default to 2D transpose if dims not specified
    if (dim0 < 0)
        dim0 = a->ndim >= 2 ? a->ndim - 2 : 0;
    if (dim1 < 0)
        dim1 = a->ndim >= 2 ? a->ndim - 1 : 0;

    // Validate dimensions
    if (dim0 >= a->ndim || dim1 >= a->ndim || dim0 < 0 || dim1 < 0) {
        LOG_ERROR("Transpose: invalid dimensions %d, %d for tensor with ndim=%d", dim0, dim1,
                  a->ndim);
        return NULL;
    }

    LOG_DEBUG("Computing Transpose: transpose(tensor %p, dim0=%d, dim1=%d)", (void*)a, dim0, dim1);

    // Create output shape (swap dimensions)
    int* out_shape = CM_MALLOC(a->ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    for (int i = 0; i < a->ndim; i++) {
        out_shape[i] = a->shape[i];
    }
    // Swap the two dimensions
    int temp        = out_shape[dim0];
    out_shape[dim0] = out_shape[dim1];
    out_shape[dim1] = temp;

    // Create result tensor
    Tensor* result = tensor_empty(out_shape, a->ndim, a->dtype, a->device);
    CM_FREE(out_shape);
    if (!result)
        return NULL;

    // Perform transpose
    float* a_data      = (float*)a->data;
    float* result_data = (float*)result->data;

    // Calculate strides for input
    size_t* in_strides = CM_MALLOC(a->ndim * sizeof(size_t));
    if (!in_strides) {
        tensor_free(result);
        return NULL;
    }

    in_strides[a->ndim - 1] = 1;
    for (int i = a->ndim - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * a->shape[i + 1];
    }

    // Calculate strides for output
    size_t* out_strides = CM_MALLOC(a->ndim * sizeof(size_t));
    if (!out_strides) {
        CM_FREE(in_strides);
        tensor_free(result);
        return NULL;
    }

    out_strides[a->ndim - 1] = 1;
    for (int i = a->ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * result->shape[i + 1];
    }

    // Transpose data
    size_t* indices = CM_CALLOC(a->ndim, sizeof(size_t));
    if (!indices) {
        CM_FREE(in_strides);
        CM_FREE(out_strides);
        tensor_free(result);
        return NULL;
    }

    for (size_t i = 0; i < a->numel; i++) {
        // Calculate input index
        size_t in_idx = 0;
        for (int d = 0; d < a->ndim; d++) {
            in_idx += indices[d] * in_strides[d];
        }

        // Calculate output index (with swapped dimensions)
        size_t out_idx = 0;
        for (int d = 0; d < a->ndim; d++) {
            int out_dim = d;
            if (d == dim0)
                out_dim = dim1;
            else if (d == dim1)
                out_dim = dim0;
            out_idx += indices[d] * out_strides[out_dim];
        }

        result_data[out_idx] = a_data[in_idx];

        // Increment indices
        for (int d = a->ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < (size_t)a->shape[d])
                break;
            indices[d] = 0;
        }
    }

    CM_FREE(indices);
    CM_FREE(in_strides);
    CM_FREE(out_strides);

    // Setup autograd
    Function* fn = autograd_function_create(OP_TRANSPOSE, "Transpose");
    if (fn) {
        // Save the dimensions that were transposed
        int* dim_data = CM_MALLOC(2 * sizeof(int));
        dim_data[0]   = dim0;
        dim_data[1]   = dim1;
        autograd_context_save_data(fn->ctx, dim_data, 2 * sizeof(int));

        autograd_function_set_backward(fn, transpose_backward);
        Tensor* inputs[] = {a};
        result           = create_output_with_grad_fn(result, fn, inputs, 1);
    }

    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;

    // Check dimensions
    if (a->ndim < 2 || b->ndim < 2) {
        LOG_ERROR("MatMul requires at least 2D tensors, got %dD and %dD", a->ndim, b->ndim);
        return NULL;
    }

    // For 2D matrices: (M, K) @ (K, N) = (M, N)
    int M   = a->shape[a->ndim - 2];
    int K_a = a->shape[a->ndim - 1];
    int K_b = b->shape[b->ndim - 2];
    int N   = b->shape[b->ndim - 1];

    if (K_a != K_b) {
        LOG_ERROR("MatMul dimension mismatch: (%d, %d) @ (%d, %d)", M, K_a, K_b, N);
        return NULL;
    }

    LOG_DEBUG("Computing MatMul: (%d, %d) @ (%d, %d) = (%d, %d)", M, K_a, K_b, N, M, N);

    // Create output shape
    int out_shape[2] = {M, N};
    Tensor* result   = tensor_zeros(out_shape, 2, a->dtype, a->device);
    if (!result)
        return NULL;

    // Perform matrix multiplication
    float* a_data      = (float*)a->data;
    float* b_data      = (float*)b->data;
    float* result_data = (float*)result->data;

    // Standard matrix multiplication: C[i][j] = sum_k(A[i][k] * B[k][j])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K_a; k++) {
                sum += a_data[i * K_a + k] * b_data[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }

    // Setup autograd
    Function* fn = autograd_function_create(OP_MATMUL, "MatMul");
    if (fn) {
        Tensor* saved[] = {a, b};
        autograd_context_save_for_backward(fn->ctx, saved, 2);
        autograd_function_set_backward(fn, matmul_backward);
        Tensor* inputs[] = {a, b};
        result           = create_output_with_grad_fn(result, fn, inputs, 2);
    }

    return result;
}
