#include "autograd/autograd.h"
#include <math.h>
#include <stdio.h>

// Forward declarations for backward functions
void transpose_backward(Function* fn, Tensor* grad_output);
void matmul_backward(Function* fn, Tensor* grad_output);
void sum_backward(Function* fn, Tensor* grad_output);
void mean_backward(Function* fn, Tensor* grad_output);

// Backward Functions for Binary Operations

// Backward for addition: d/dx (x + y) = 1, d/dy (x + y) = 1
// Handles broadcasting by unbroadcasting gradients
void add_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Add operation with broadcasting");

    Tensor* input0 = fn->inputs[0];
    Tensor* input1 = fn->inputs[1];

    // Gradient for first input (unbroadcast if needed)
    if (input0 && fn->needs_input_grad[0]) {
        Tensor* grad0 = NULL;
        compute_grad_for_broadcast(grad_output, input0->shape, input0->ndim, &grad0);
        if (grad0) {
            tensor_accumulate_grad(input0, grad0);
            tensor_free(grad0);
        }
    }

    // Gradient for second input (unbroadcast if needed)
    if (input1 && fn->needs_input_grad[1]) {
        Tensor* grad1 = NULL;
        compute_grad_for_broadcast(grad_output, input1->shape, input1->ndim, &grad1);
        if (grad1) {
            tensor_accumulate_grad(input1, grad1);
            tensor_free(grad1);
        }
    }
}

// Backward for subtraction: d/dx (x - y) = 1, d/dy (x - y) = -1
// Handles broadcasting by unbroadcasting gradients
void sub_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Sub operation with broadcasting");

    Tensor* input0 = fn->inputs[0];
    Tensor* input1 = fn->inputs[1];

    // Gradient for first input (unbroadcast if needed)
    if (input0 && fn->needs_input_grad[0]) {
        Tensor* grad0 = NULL;
        compute_grad_for_broadcast(grad_output, input0->shape, input0->ndim, &grad0);
        if (grad0) {
            tensor_accumulate_grad(input0, grad0);
            tensor_free(grad0);
        }
    }

    // Gradient for second input: -grad_output (unbroadcast if needed)
    if (input1 && fn->needs_input_grad[1]) {
        // Create negative gradient
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* neg_grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (neg_grad) {
            float* neg_data  = (float*)neg_grad->data;
            float* grad_data = (float*)grad_output->data;
            for (size_t i = 0; i < grad_output->numel; i++) {
                neg_data[i] = -grad_data[i];
            }

            Tensor* grad1 = NULL;
            compute_grad_for_broadcast(neg_grad, input1->shape, input1->ndim, &grad1);
            if (grad1) {
                tensor_accumulate_grad(input1, grad1);
                tensor_free(grad1);
            }
            tensor_free(neg_grad);
        }
    }
}

// Backward for multiplication: d/dx (x * y) = y, d/dy (x * y) = x
// Handles broadcasting by computing element-wise gradient then unbroadcasting
void mul_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Mul operation with broadcasting");

    Tensor* input0 = fn->ctx->saved_tensors[0];
    Tensor* input1 = fn->ctx->saved_tensors[1];

    // Gradient for first input: df/dx = y * grad_output
    if (fn->needs_input_grad[0]) {
        // Use tensor_mul to properly broadcast input1 with grad_output
        Tensor* grad0_broadcasted = tensor_mul(input1, grad_output);
        if (grad0_broadcasted) {
            Tensor* grad0 = NULL;
            compute_grad_for_broadcast(grad0_broadcasted, input0->shape, input0->ndim, &grad0);
            if (grad0) {
                tensor_accumulate_grad(fn->inputs[0], grad0);
                tensor_free(grad0);
            }
            tensor_free(grad0_broadcasted);
        }
    }

    // Gradient for second input: df/dy = x * grad_output
    if (fn->needs_input_grad[1]) {
        // Use tensor_mul to properly broadcast input0 with grad_output
        Tensor* grad1_broadcasted = tensor_mul(input0, grad_output);
        if (grad1_broadcasted) {
            Tensor* grad1 = NULL;
            compute_grad_for_broadcast(grad1_broadcasted, input1->shape, input1->ndim, &grad1);
            if (grad1) {
                tensor_accumulate_grad(fn->inputs[1], grad1);
                tensor_free(grad1);
            }
            tensor_free(grad1_broadcasted);
        }
    }
}

// Backward for division: d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
// Handles broadcasting by computing element-wise gradient then unbroadcasting
void div_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Div operation with broadcasting");

    Tensor* input0 = fn->ctx->saved_tensors[0];
    Tensor* input1 = fn->ctx->saved_tensors[1];

    // Gradient for first input: df/dx = 1/y * grad_output
    if (fn->needs_input_grad[0]) {
        // Use tensor_div to properly broadcast grad_output with input1
        Tensor* grad0_broadcasted = tensor_div(grad_output, input1);
        if (grad0_broadcasted) {
            Tensor* grad0 = NULL;
            compute_grad_for_broadcast(grad0_broadcasted, input0->shape, input0->ndim, &grad0);
            if (grad0) {
                tensor_accumulate_grad(fn->inputs[0], grad0);
                tensor_free(grad0);
            }
            tensor_free(grad0_broadcasted);
        }
    }

    // Gradient for second input: df/dy = -x/y^2 * grad_output
    if (fn->needs_input_grad[1]) {
        // Compute -x * grad_output / y^2
        Tensor* y_squared = tensor_mul(input1, input1);
        if (y_squared) {
            Tensor* neg_x = tensor_mul(input0, grad_output);
            if (neg_x) {
                float* neg_x_data = (float*)neg_x->data;
                for (size_t i = 0; i < neg_x->numel; i++) {
                    neg_x_data[i] = -neg_x_data[i];
                }

                Tensor* grad1_broadcasted = tensor_div(neg_x, y_squared);
                if (grad1_broadcasted) {
                    Tensor* grad1 = NULL;
                    compute_grad_for_broadcast(grad1_broadcasted, input1->shape, input1->ndim,
                                               &grad1);
                    if (grad1) {
                        tensor_accumulate_grad(fn->inputs[1], grad1);
                        tensor_free(grad1);
                    }
                    tensor_free(grad1_broadcasted);
                }
                tensor_free(neg_x);
            }
            tensor_free(y_squared);
        }
    }
}

// Backward for power: d/dx (x^y) = y * x^(y-1), d/dy (x^y) = x^y * ln(x)
// Handles broadcasting by computing element-wise gradient then unbroadcasting
void pow_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Pow operation with broadcasting");

    Tensor* input0 = fn->ctx->saved_tensors[0]; // base
    Tensor* input1 = fn->ctx->saved_tensors[1]; // exponent

    // Gradient for base: df/dx = y * x^(y-1) * grad_output
    if (fn->needs_input_grad[0]) {
        TensorConfig config = tensor_config_with_dtype_device(input1->dtype, input1->device);
        Tensor* one         = tensor_ones(input1->shape, input1->ndim, &config);
        if (one) {
            Tensor* y_minus_1 = tensor_sub(input1, one);
            if (y_minus_1) {
                Tensor* x_pow_y_minus_1 = tensor_pow(input0, y_minus_1);
                if (x_pow_y_minus_1) {
                    Tensor* temp = tensor_mul(input1, x_pow_y_minus_1);
                    if (temp) {
                        Tensor* grad0_broadcasted = tensor_mul(temp, grad_output);
                        if (grad0_broadcasted) {
                            Tensor* grad0 = NULL;
                            compute_grad_for_broadcast(grad0_broadcasted, input0->shape,
                                                       input0->ndim, &grad0);
                            if (grad0) {
                                tensor_accumulate_grad(fn->inputs[0], grad0);
                                tensor_free(grad0);
                            }
                            tensor_free(grad0_broadcasted);
                        }
                        tensor_free(temp);
                    }
                    tensor_free(x_pow_y_minus_1);
                }
                tensor_free(y_minus_1);
            }
            tensor_free(one);
        }
    }

    // Gradient for exponent: df/dy = x^y * ln(x) * grad_output
    if (fn->needs_input_grad[1]) {
        Tensor* x_pow_y = tensor_pow(input0, input1);
        if (x_pow_y) {
            Tensor* ln_x = tensor_log(input0);
            if (ln_x) {
                Tensor* temp = tensor_mul(x_pow_y, ln_x);
                if (temp) {
                    Tensor* grad1_broadcasted = tensor_mul(temp, grad_output);
                    if (grad1_broadcasted) {
                        Tensor* grad1 = NULL;
                        compute_grad_for_broadcast(grad1_broadcasted, input1->shape, input1->ndim,
                                                   &grad1);
                        if (grad1) {
                            tensor_accumulate_grad(fn->inputs[1], grad1);
                            tensor_free(grad1);
                        }
                        tensor_free(grad1_broadcasted);
                    }
                    tensor_free(temp);
                }
                tensor_free(ln_x);
            }
            tensor_free(x_pow_y);
        }
    }
}

// Backward Functions for Unary Operations

// Backward for negation: d/dx (-x) = -1
void neg_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Neg operation");

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                tensor_set_float(grad, i, -tensor_get_float(grad_output, i));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for exponential: d/dx (exp(x)) = exp(x)
void exp_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Exp operation");

    Tensor* output = fn->ctx->saved_tensors[0]; // We saved the output exp(x)

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val   = tensor_get_float(grad_output, i);
                float output_val = tensor_get_float(output, i);
                tensor_set_float(grad, i, grad_val * output_val);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for logarithm: d/dx (log(x)) = 1/x
void log_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Log operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                tensor_set_float(grad, i, grad_val / input_val);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for square root: d/dx (sqrt(x)) = 1/(2*sqrt(x))
void sqrt_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Sqrt operation");

    Tensor* output = fn->ctx->saved_tensors[0]; // We saved sqrt(x)

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val   = tensor_get_float(grad_output, i);
                float output_val = tensor_get_float(output, i);
                tensor_set_float(grad, i, grad_val / (2.0f * output_val));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for sine: d/dx (sin(x)) = cos(x)
void sin_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Sin operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                tensor_set_float(grad, i, grad_val * cosf(input_val));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for cosine: d/dx (cos(x)) = -sin(x)
void cos_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Cos operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                tensor_set_float(grad, i, -grad_val * sinf(input_val));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for tangent: d/dx (tan(x)) = 1/cos^2(x) = sec^2(x)
void tan_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Tan operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                float cos_val   = cosf(input_val);
                tensor_set_float(grad, i, grad_val / (cos_val * cos_val));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for tanh: d/dx (tanh(x)) = 1 - tanh^2(x)
void tanh_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Tanh operation");

    Tensor* output = fn->ctx->saved_tensors[0]; // We saved tanh(x)

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val   = tensor_get_float(grad_output, i);
                float output_val = tensor_get_float(output, i);
                tensor_set_float(grad, i, grad_val * (1.0f - output_val * output_val));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward Functions for Activation Functions

// Backward for ReLU: d/dx (max(0, x)) = 1 if x > 0, else 0
void relu_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for ReLU operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                tensor_set_float(grad, i, input_val > 0 ? grad_val : 0.0f);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for Sigmoid: d/dx (sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
void sigmoid_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Sigmoid operation");

    Tensor* output = fn->ctx->saved_tensors[0]; // We saved sigmoid(x)

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val   = tensor_get_float(grad_output, i);
                float output_val = tensor_get_float(output, i);
                tensor_set_float(grad, i, grad_val * output_val * (1.0f - output_val));
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward for Leaky ReLU: d/dx (leaky_relu(x)) = 1 if x > 0, else alpha
void leaky_relu_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Leaky ReLU operation");

    Tensor* input    = fn->ctx->saved_tensors[0];
    float* alpha_ptr = (float*)fn->ctx->saved_data;
    float alpha      = alpha_ptr ? *alpha_ptr : 0.01f;

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                tensor_set_float(grad, i, input_val > 0 ? grad_val : grad_val * alpha);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// ELU backward: d/dx ELU(x) = 1 if x > 0, else alpha * exp(x)
void elu_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for ELU operation");

    Tensor* input    = fn->ctx->saved_tensors[0];
    float* alpha_ptr = (float*)fn->ctx->saved_data;
    float alpha      = alpha_ptr ? *alpha_ptr : 1.0f;

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val   = tensor_get_float(grad_output, i);
                float input_val  = tensor_get_float(input, i);
                float grad_input = input_val > 0.0f ? grad_val : grad_val * alpha * expf(input_val);
                tensor_set_float(grad, i, grad_input);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// SELU backward: d/dx SELU(x) = scale if x > 0, else scale * alpha * exp(x)
void selu_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for SELU operation");

    Tensor* input     = fn->ctx->saved_tensors[0];
    const float scale = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val   = tensor_get_float(grad_output, i);
                float input_val  = tensor_get_float(input, i);
                float grad_input = input_val > 0.0f ? grad_val * scale
                                                    : grad_val * scale * alpha * expf(input_val);
                tensor_set_float(grad, i, grad_input);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Swish backward: d/dx Swish(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
void swish_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Swish operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val    = tensor_get_float(grad_output, i);
                float input_val   = tensor_get_float(input, i);
                float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
                float grad_input =
                    grad_val * sigmoid_val * (1.0f + input_val * (1.0f - sigmoid_val));
                tensor_set_float(grad, i, grad_input);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Mish backward: d/dx Mish(x) = tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh^2(softplus(x)))
void mish_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Mish operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val       = tensor_get_float(grad_output, i);
                float input_val      = tensor_get_float(input, i);
                float softplus       = logf(1.0f + expf(input_val));
                float tanh_softplus  = tanhf(softplus);
                float sigmoid_val    = 1.0f / (1.0f + expf(-input_val));
                float tanh_deriv     = 1.0f - tanh_softplus * tanh_softplus;
                float softplus_deriv = sigmoid_val;
                float grad_input     = grad_val * (tanh_softplus + input_val * sigmoid_val *
                                                                   tanh_deriv * softplus_deriv);
                tensor_set_float(grad, i, grad_input);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Hard Swish backward: d/dx HardSwish(x) = ReLU6(x + 3) / 6 + x * (x + 3 in [0, 6]) / 6
void hard_swish_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Hard Swish operation");

    Tensor* input = fn->ctx->saved_tensors[0];

    if (fn->needs_input_grad[0]) {
        TensorConfig config =
            tensor_config_with_dtype_device(grad_output->dtype, grad_output->device);
        Tensor* grad = tensor_empty(grad_output->shape, grad_output->ndim, &config);
        if (grad) {
            for (size_t i = 0; i < grad_output->numel; i++) {
                float grad_val  = tensor_get_float(grad_output, i);
                float input_val = tensor_get_float(input, i);
                float x_plus_3  = input_val + 3.0f;
                float grad_input;
                if (x_plus_3 < 0.0f) {
                    grad_input = 0.0f;
                } else if (x_plus_3 > 6.0f) {
                    grad_input = grad_val * input_val / 6.0f;
                } else {
                    grad_input = grad_val * (x_plus_3 / 6.0f + input_val / 6.0f);
                }
                tensor_set_float(grad, i, grad_input);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// Backward Functions for Reduction Operations

// Backward for sum: gradient is broadcast to match input shape
void sum_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Sum operation");

    Tensor* input = fn->inputs[0];
    if (!fn->needs_input_grad[0])
        return;

    // Get original shape from context
    int* input_shape = input->shape;
    int input_ndim   = input->ndim;
    if (fn->ctx && fn->ctx->saved_shape) {
        input_shape = fn->ctx->saved_shape;
        input_ndim  = fn->ctx->saved_ndim;
    }

    // Create gradient with same shape as input
    TensorConfig config = tensor_config_with_dtype_device(input->dtype, input->device);
    Tensor* grad        = tensor_empty(input_shape, input_ndim, &config);
    if (!grad)
        return;

    float* grad_data        = (float*)grad->data;
    float* grad_output_data = (float*)grad_output->data;

    // Broadcast grad_output to match input shape
    if (grad_output->numel == 1) {
        float grad_val = grad_output_data[0];
        for (size_t i = 0; i < grad->numel; i++) {
            grad_data[i] = grad_val;
        }
    } else {
        // Dimension-specific reduction - broadcast along reduced dimension
        for (size_t i = 0; i < grad->numel; i++) {
            grad_data[i] = grad_output_data[i % grad_output->numel];
        }
    }

    tensor_accumulate_grad(input, grad);
    tensor_free(grad);
}

// Backward for mean: gradient is broadcast and divided by number of elements
void mean_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_DEBUG("Computing backward for Mean operation");

    Tensor* input = fn->inputs[0];
    if (!fn->needs_input_grad[0])
        return;

    // Get original shape from context
    int* input_shape = input->shape;
    int input_ndim   = input->ndim;
    if (fn->ctx && fn->ctx->saved_shape) {
        input_shape = fn->ctx->saved_shape;
        input_ndim  = fn->ctx->saved_ndim;
    }

    // Create gradient with same shape as input
    TensorConfig config = tensor_config_with_dtype_device(input->dtype, input->device);
    Tensor* grad        = tensor_empty(input_shape, input_ndim, &config);
    if (!grad)
        return;

    float* grad_data        = (float*)grad->data;
    float* grad_output_data = (float*)grad_output->data;

    // Calculate scaling factor
    size_t input_numel = 1;
    for (int i = 0; i < input_ndim; i++) {
        input_numel *= input_shape[i];
    }

    if (grad_output->numel == 1) {
        // Scalar gradient - divide by total number of elements
        float grad_val = grad_output_data[0] / (float)input_numel;
        for (size_t i = 0; i < grad->numel; i++) {
            grad_data[i] = grad_val;
        }
    } else {
        // Dimension-specific reduction
        size_t reduced_elements = input_numel / grad_output->numel;
        float scale             = 1.0f / (float)reduced_elements;

        for (size_t i = 0; i < grad->numel; i++) {
            grad_data[i] = grad_output_data[i % grad_output->numel] * scale;
        }
    }

    tensor_accumulate_grad(input, grad);
    tensor_free(grad);
}

// Backward Functions for Loss Functions have been moved to autograd/loss_functions.c

// Matrix Operations Backward

// Backward for transpose operation
// Gradient flows back through transpose by transposing again
void transpose_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 1)
        return;

    LOG_INFO("Computing backward for Transpose operation");

    Tensor* input = fn->inputs[0];
    if (!input) {
        LOG_WARNING("Transpose backward: input is NULL");
        return;
    }
    if (!fn->needs_input_grad[0]) {
        LOG_DEBUG("Transpose backward: input doesn't need gradients");
        return;
    }

    // Get the dimensions that were transposed
    int dim0 = 0, dim1 = 1;
    if (fn->ctx && fn->ctx->saved_data && fn->ctx->saved_data_size >= 2 * sizeof(int)) {
        int* dim_data = (int*)fn->ctx->saved_data;
        dim0          = dim_data[0];
        dim1          = dim_data[1];
    }

    // Transpose gradient back (transpose is its own inverse)
    Tensor* grad_input = tensor_transpose(grad_output, dim0, dim1);
    if (grad_input) {
        LOG_DEBUG("Transpose backward: Accumulating gradient for input (requires_grad=%d, "
                  "has_grad_before=%d)",
                  input->requires_grad, input->grad != NULL);
        tensor_accumulate_grad(input, grad_input);
        LOG_DEBUG("Transpose backward: Accumulated gradient for input (has_grad_after=%d)",
                  input->grad != NULL);
        tensor_free(grad_input);
    } else {
        LOG_WARNING("Transpose backward: Failed to transpose grad_output");
    }
}

// Backward for matrix multiplication: C = A @ B
// dL/dA = dL/dC @ B^T
// dL/dB = A^T @ dL/dC
void matmul_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_INFO("Computing backward for MatMul operation");

    // Get saved tensors from forward pass
    if (!fn->ctx || !fn->ctx->saved_tensors || fn->ctx->num_saved_tensors < 2) {
        LOG_ERROR("MatMul backward: missing saved tensors");
        return;
    }

    Tensor* input_a = fn->ctx->saved_tensors[0]; // A
    Tensor* input_b = fn->ctx->saved_tensors[1]; // B

    if (!input_a || !input_b) {
        LOG_ERROR("MatMul backward: saved tensors are NULL");
        return;
    }

    // For 2D matrix multiplication: C = A @ B
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC

    // Check if inputs require gradients
    Tensor* input0_from_fn = (fn->num_inputs > 0) ? fn->inputs[0] : NULL;
    Tensor* input1_from_fn = (fn->num_inputs > 1) ? fn->inputs[1] : NULL;

    if (input0_from_fn && fn->needs_input_grad[0]) {
        // Compute dL/dA = dL/dC @ B^T
        // Transpose B
        Tensor* b_T = tensor_transpose(input_b, -2, -1); // Transpose last two dims
        if (b_T) {
            // Matrix multiply: grad_output @ B^T
            Tensor* grad_a = tensor_matmul(grad_output, b_T);
            if (grad_a) {
                tensor_accumulate_grad(input0_from_fn, grad_a);
                tensor_free(grad_a);
                LOG_DEBUG("Accumulated gradient for input A (matmul)");
            }
            tensor_free(b_T);
        }
    }

    if (input1_from_fn && fn->needs_input_grad[1]) {
        // Compute dL/dB = A^T @ dL/dC
        // Transpose A
        Tensor* a_T = tensor_transpose(input_a, -2, -1); // Transpose last two dims
        if (a_T) {
            // Matrix multiply: A^T @ grad_output
            Tensor* grad_b = tensor_matmul(a_T, grad_output);
            if (grad_b) {
                LOG_DEBUG("MatMul backward: Accumulating gradient for input1 (requires_grad=%d, "
                          "has_grad_before=%d)",
                          input1_from_fn->requires_grad, input1_from_fn->grad != NULL);
                tensor_accumulate_grad(input1_from_fn, grad_b);
                LOG_DEBUG("MatMul backward: Accumulated gradient for input1 (has_grad_after=%d)",
                          input1_from_fn->grad != NULL);
                tensor_free(grad_b);
            }
            tensor_free(a_T);
        } else {
            LOG_WARNING("MatMul backward: Failed to transpose input_a for gradient computation");
        }
    } else {
        LOG_DEBUG("MatMul backward: Skipping input1 (input1_from_fn=%p, needs_input_grad[1]=%d)",
                  (void*)input1_from_fn, fn->needs_input_grad[1]);
    }

    LOG_DEBUG("MatMul backward completed");
}
