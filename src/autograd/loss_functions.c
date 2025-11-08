/**
 * @file loss_functions.c
 * @brief Loss functions implementation with autograd support
 *
 * This file implements loss functions with automatic differentiation.
 * All loss functions support automatic differentiation.
 */

#include "autograd/autograd.h"
#include "autograd/loss_functions.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <math.h>
#include <string.h>

// Forward declaration for backward function
static void mse_loss_backward(Function* fn, Tensor* grad_output);
static void mae_loss_backward(Function* fn, Tensor* grad_output);
static void bce_loss_backward(Function* fn, Tensor* grad_output);
static void cross_entropy_loss_backward(Function* fn, Tensor* grad_output);
static void huber_loss_backward(Function* fn, Tensor* grad_output);
static void kl_div_loss_backward(Function* fn, Tensor* grad_output);

// Helper function to create output with grad_fn
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

    // Set up inputs and mark which need gradients
    autograd_function_set_inputs(fn, inputs, num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        fn->needs_input_grad[i] = inputs[i] && inputs[i]->requires_grad;
    }

    // Link result tensor to function
    result->grad_fn       = fn;
    result->requires_grad = true;

    LOG_DEBUG("Created loss output tensor with grad_fn (op=%s)", op_type_to_string(fn->op_type));

    return result;
}

// Mean Squared Error Loss

Tensor* tensor_mse_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("MSE Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing MSE Loss");

    // Check shape compatibility
    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("MSE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    int shape[]    = {1};
    Tensor* result = tensor_empty(shape, 1, input->dtype, input->device);
    if (!result) {
        LOG_ERROR("MSE Loss: failed to create output tensor");
        return NULL;
    }

    float mse = 0.0f;
    size_t n  = input->numel < target->numel ? input->numel : target->numel;

    // Handle broadcasting: if one is scalar, use it for all elements
    bool input_is_scalar  = (input->numel == 1);
    bool target_is_scalar = (target->numel == 1);

    for (size_t i = 0; i < n; i++) {
        float input_val  = tensor_get_float(input, input_is_scalar ? 0 : i);
        float target_val = tensor_get_float(target, target_is_scalar ? 0 : i);
        float diff       = input_val - target_val;
        mse += diff * diff;
    }

    tensor_set_float(result, 0, mse / (float)n);

    // Set up autograd if gradients are needed
    if (autograd_is_grad_enabled() && (input->requires_grad || target->requires_grad)) {
        Function* fn = autograd_function_create(OP_MSE_LOSS, "MSELoss");
        if (fn) {
            Tensor* saved[] = {input, target};
            autograd_context_save_for_backward(fn->ctx, saved, 2);
            autograd_function_set_backward(fn, mse_loss_backward);
            Tensor* inputs[] = {input, target};
            result           = create_output_with_grad_fn(result, fn, inputs, 2);
        }
    } else {
        // No gradients needed, result doesn't require grad
        result->requires_grad = false;
    }

    return result;
}

// Backward for MSE loss: d/dx (x - y)^2 = 2(x - y) / n
static void mse_loss_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for MSE Loss operation");

    Tensor* input  = fn->ctx->saved_tensors[0];
    Tensor* target = fn->ctx->saved_tensors[1];

    if (!input || !target) {
        LOG_ERROR("MSE Loss backward: saved tensors are NULL");
        return;
    }

    float grad_output_val = tensor_get_float(grad_output, 0);
    float scale           = 2.0f / (float)input->numel;

    // Gradient for input: df/dx = 2(x - y) * grad_output / n
    if (fn->needs_input_grad[0]) {
        Tensor* grad = tensor_empty(input->shape, input->ndim, input->dtype, input->device);
        if (grad) {
            for (size_t i = 0; i < input->numel; i++) {
                float input_val  = tensor_get_float(input, i);
                float target_val = tensor_get_float(target, i < target->numel ? i : 0);
                float grad_val   = scale * grad_output_val * (input_val - target_val);
                tensor_set_float(grad, i, grad_val);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }

    // Gradient for target: df/dy = -2(x - y) * grad_output / n
    if (fn->needs_input_grad[1]) {
        Tensor* grad = tensor_empty(target->shape, target->ndim, target->dtype, target->device);
        if (grad) {
            for (size_t i = 0; i < target->numel; i++) {
                float input_val  = tensor_get_float(input, i < input->numel ? i : 0);
                float target_val = tensor_get_float(target, i);
                float grad_val   = -scale * grad_output_val * (input_val - target_val);
                tensor_set_float(grad, i, grad_val);
            }
            tensor_accumulate_grad(fn->inputs[1], grad);
            tensor_free(grad);
        }
    }
}

// Mean Absolute Error Loss (L1 Loss)

Tensor* tensor_mae_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("MAE Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing MAE Loss");

    // Check shape compatibility
    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("MAE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    int shape[]    = {1};
    Tensor* result = tensor_empty(shape, 1, input->dtype, input->device);
    if (!result) {
        LOG_ERROR("MAE Loss: failed to create output tensor");
        return NULL;
    }

    float mae = 0.0f;
    size_t n  = input->numel < target->numel ? input->numel : target->numel;

    // Handle broadcasting: if one is scalar, use it for all elements
    bool input_is_scalar  = (input->numel == 1);
    bool target_is_scalar = (target->numel == 1);

    for (size_t i = 0; i < n; i++) {
        float input_val  = tensor_get_float(input, input_is_scalar ? 0 : i);
        float target_val = tensor_get_float(target, target_is_scalar ? 0 : i);
        mae += fabsf(input_val - target_val);
    }

    tensor_set_float(result, 0, mae / (float)n);

    // Set up autograd if gradients are needed
    if (autograd_is_grad_enabled() && (input->requires_grad || target->requires_grad)) {
        Function* fn = autograd_function_create(OP_MAE_LOSS, "MAELoss");
        if (fn) {
            Tensor* saved[] = {input, target};
            autograd_context_save_for_backward(fn->ctx, saved, 2);
            autograd_function_set_backward(fn, mae_loss_backward);
            Tensor* inputs[] = {input, target};
            result           = create_output_with_grad_fn(result, fn, inputs, 2);
        }
    } else {
        result->requires_grad = false;
    }

    return result;
}

// Backward for MAE loss: d/dx |x - y| = sign(x - y) / n
static void mae_loss_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for MAE Loss operation");

    Tensor* input  = fn->ctx->saved_tensors[0];
    Tensor* target = fn->ctx->saved_tensors[1];

    if (!input || !target) {
        LOG_ERROR("MAE Loss backward: saved tensors are NULL");
        return;
    }

    float grad_output_val = tensor_get_float(grad_output, 0);
    float scale           = grad_output_val / (float)input->numel;

    // Gradient for input: df/dx = sign(x - y) * grad_output / n
    if (fn->needs_input_grad[0]) {
        Tensor* grad = tensor_empty(input->shape, input->ndim, input->dtype, input->device);
        if (grad) {
            for (size_t i = 0; i < input->numel; i++) {
                float input_val  = tensor_get_float(input, i);
                float target_val = tensor_get_float(target, i < target->numel ? i : 0);
                float diff       = input_val - target_val;
                float sign       = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
                tensor_set_float(grad, i, scale * sign);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }

    // Gradient for target: df/dy = -sign(x - y) * grad_output / n
    if (fn->needs_input_grad[1]) {
        Tensor* grad = tensor_empty(target->shape, target->ndim, target->dtype, target->device);
        if (grad) {
            for (size_t i = 0; i < target->numel; i++) {
                float input_val  = tensor_get_float(input, i < input->numel ? i : 0);
                float target_val = tensor_get_float(target, i);
                float diff       = input_val - target_val;
                float sign       = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
                tensor_set_float(grad, i, -scale * sign);
            }
            tensor_accumulate_grad(fn->inputs[1], grad);
            tensor_free(grad);
        }
    }
}

// Binary Cross Entropy Loss

Tensor* tensor_bce_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("BCE Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing BCE Loss");

    // Check shape compatibility
    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("BCE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    int shape[]    = {1};
    Tensor* result = tensor_empty(shape, 1, input->dtype, input->device);
    if (!result) {
        LOG_ERROR("BCE Loss: failed to create output tensor");
        return NULL;
    }

    float bce     = 0.0f;
    size_t n      = input->numel < target->numel ? input->numel : target->numel;
    float epsilon = 1e-8f; // Small value to prevent log(0)

    // Handle broadcasting: if one is scalar, use it for all elements
    bool input_is_scalar  = (input->numel == 1);
    bool target_is_scalar = (target->numel == 1);

    for (size_t i = 0; i < n; i++) {
        float input_val  = tensor_get_float(input, input_is_scalar ? 0 : i);
        float target_val = tensor_get_float(target, target_is_scalar ? 0 : i);

        // Clamp input to [epsilon, 1-epsilon] to prevent log(0)
        float clamped_input = input_val;
        if (clamped_input < epsilon)
            clamped_input = epsilon;
        if (clamped_input > 1.0f - epsilon)
            clamped_input = 1.0f - epsilon;

        // BCE: -[target * log(input) + (1 - target) * log(1 - input)]
        float term1 = target_val * logf(clamped_input);
        float term2 = (1.0f - target_val) * logf(1.0f - clamped_input);
        bce += -(term1 + term2);
    }

    tensor_set_float(result, 0, bce / (float)n);

    // Set up autograd if gradients are needed
    if (autograd_is_grad_enabled() && (input->requires_grad || target->requires_grad)) {
        Function* fn = autograd_function_create(OP_BCE_LOSS, "BCELoss");
        if (fn) {
            Tensor* saved[] = {input, target};
            autograd_context_save_for_backward(fn->ctx, saved, 2);
            autograd_function_set_backward(fn, bce_loss_backward);
            Tensor* inputs[] = {input, target};
            result           = create_output_with_grad_fn(result, fn, inputs, 2);
        }
    } else {
        result->requires_grad = false;
    }

    return result;
}

// Backward for BCE loss: d/dx BCE = (input - target) / (input * (1 - input)) / n
static void bce_loss_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for BCE Loss operation");

    Tensor* input  = fn->ctx->saved_tensors[0];
    Tensor* target = fn->ctx->saved_tensors[1];

    if (!input || !target) {
        LOG_ERROR("BCE Loss backward: saved tensors are NULL");
        return;
    }

    float grad_output_val = tensor_get_float(grad_output, 0);
    float scale           = grad_output_val / (float)input->numel;
    float epsilon         = 1e-8f;

    // Gradient for input: df/dx = (input - target) / (input * (1 - input)) * grad_output / n
    if (fn->needs_input_grad[0]) {
        Tensor* grad = tensor_empty(input->shape, input->ndim, input->dtype, input->device);
        if (grad) {
            for (size_t i = 0; i < input->numel; i++) {
                float input_val  = tensor_get_float(input, i);
                float target_val = tensor_get_float(target, i < target->numel ? i : 0);

                // Clamp input to prevent division by zero
                float clamped_input = input_val;
                if (clamped_input < epsilon)
                    clamped_input = epsilon;
                if (clamped_input > 1.0f - epsilon)
                    clamped_input = 1.0f - epsilon;

                // Gradient: (clamped_input - target) / (clamped_input * (1 - clamped_input))
                float grad_val =
                    (clamped_input - target_val) / (clamped_input * (1.0f - clamped_input));
                tensor_set_float(grad, i, scale * grad_val);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }

    // Target gradients are typically not needed (targets are constants)
    if (fn->needs_input_grad[1]) {
        // For completeness, but usually not used
        Tensor* grad = tensor_empty(target->shape, target->ndim, target->dtype, target->device);
        if (grad) {
            for (size_t i = 0; i < target->numel; i++) {
                float input_val     = tensor_get_float(input, i < input->numel ? i : 0);
                float clamped_input = input_val;
                if (clamped_input < epsilon)
                    clamped_input = epsilon;
                if (clamped_input > 1.0f - epsilon)
                    clamped_input = 1.0f - epsilon;

                // d/dy BCE = -log(input) + log(1 - input)
                float grad_val = -logf(clamped_input) + logf(1.0f - clamped_input);
                tensor_set_float(grad, i, scale * grad_val);
            }
            tensor_accumulate_grad(fn->inputs[1], grad);
            tensor_free(grad);
        }
    }
}

// Cross Entropy Loss
// Note: This is a simplified version that assumes logits input and class index targets

Tensor* tensor_cross_entropy_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("Cross Entropy Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing Cross Entropy Loss");

    // For cross entropy, input should be logits [N, C] and target should be class indices [N]
    // Simplified: assume input is 1D with numel elements representing logits
    // and target is 1D with class indices (0 to numel-1)

    if (target->ndim != 1 || input->ndim < 1) {
        LOG_ERROR("Cross Entropy Loss: target must be 1D, input must be at least 1D");
        return NULL;
    }

    // Check if shapes are compatible
    // If input is [N, C], target should be [N] with values in [0, C-1]
    size_t n_samples = target->numel;

    // For simplicity, assume input is flattened or matches target length
    if (input->numel < n_samples) {
        LOG_ERROR("Cross Entropy Loss: input has fewer elements than target samples");
        return NULL;
    }

    int shape[]    = {1};
    Tensor* result = tensor_empty(shape, 1, input->dtype, input->device);
    if (!result) {
        LOG_ERROR("Cross Entropy Loss: failed to create output tensor");
        return NULL;
    }

    float ce_loss = 0.0f;
    float epsilon = 1e-8f;

    // Compute cross entropy: -log(softmax(input)[target_class])
    // Simplified: for each sample, get the target class logit and compute softmax
    for (size_t i = 0; i < n_samples; i++) {
        int target_class = (int)tensor_get_float(target, i);
        if (target_class < 0 || target_class >= (int)input->numel) {
            LOG_WARNING("Cross Entropy Loss: invalid target class %d at index %zu", target_class,
                        i);
            continue;
        }

        // Get the logit for the target class
        float target_logit = tensor_get_float(input, target_class);

        // Compute softmax denominator (sum of exp of all logits)
        float max_logit = target_logit;
        for (size_t j = 0; j < input->numel; j++) {
            float logit = tensor_get_float(input, j);
            if (logit > max_logit)
                max_logit = logit;
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < input->numel; j++) {
            float logit = tensor_get_float(input, j);
            sum_exp += expf(logit - max_logit); // Numerical stability
        }

        // Cross entropy: -log(exp(target_logit - max) / sum)
        float softmax_target = expf(target_logit - max_logit) / sum_exp;
        float clamped_prob =
            softmax_target < epsilon
                ? epsilon
                : (softmax_target > 1.0f - epsilon ? 1.0f - epsilon : softmax_target);
        ce_loss += -logf(clamped_prob);
    }

    tensor_set_float(result, 0, ce_loss / (float)n_samples);

    // Set up autograd if gradients are needed
    if (autograd_is_grad_enabled() && (input->requires_grad || target->requires_grad)) {
        Function* fn = autograd_function_create(OP_CROSS_ENTROPY_LOSS, "CrossEntropyLoss");
        if (fn) {
            Tensor* saved[] = {input, target};
            autograd_context_save_for_backward(fn->ctx, saved, 2);
            autograd_function_set_backward(fn, cross_entropy_loss_backward);
            Tensor* inputs[] = {input, target};
            result           = create_output_with_grad_fn(result, fn, inputs, 2);
        }
    } else {
        result->requires_grad = false;
    }

    return result;
}

// Backward for Cross Entropy loss: d/dx CE = (softmax(input) - one_hot(target)) / n
static void cross_entropy_loss_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Cross Entropy Loss operation");

    Tensor* input  = fn->ctx->saved_tensors[0];
    Tensor* target = fn->ctx->saved_tensors[1];

    if (!input || !target) {
        LOG_ERROR("Cross Entropy Loss backward: saved tensors are NULL");
        return;
    }

    float grad_output_val = tensor_get_float(grad_output, 0);
    size_t n_samples      = target->numel;
    float scale           = grad_output_val / (float)n_samples;

    // Gradient for input: df/dx = (softmax(input) - one_hot(target)) * grad_output / n
    if (fn->needs_input_grad[0]) {
        Tensor* grad = tensor_empty(input->shape, input->ndim, input->dtype, input->device);
        if (grad) {
            // Initialize gradient with softmax values
            // First compute softmax of input
            float max_logit = tensor_get_float(input, 0);
            for (size_t i = 1; i < input->numel; i++) {
                float logit = tensor_get_float(input, i);
                if (logit > max_logit)
                    max_logit = logit;
            }

            float sum_exp = 0.0f;
            for (size_t i = 0; i < input->numel; i++) {
                float logit = tensor_get_float(input, i);
                sum_exp += expf(logit - max_logit);
            }

            // Set gradient to softmax(input)
            for (size_t i = 0; i < input->numel; i++) {
                float logit       = tensor_get_float(input, i);
                float softmax_val = expf(logit - max_logit) / sum_exp;
                tensor_set_float(grad, i, scale * softmax_val);
            }

            // Subtract one-hot encoding of target
            for (size_t i = 0; i < n_samples; i++) {
                int target_class = (int)tensor_get_float(target, i);
                if (target_class >= 0 && target_class < (int)input->numel) {
                    float current = tensor_get_float(grad, target_class);
                    tensor_set_float(grad, target_class, current - scale);
                }
            }

            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }

    // Target gradients are not needed (targets are constants)
}

// Huber Loss
Tensor* tensor_huber_loss(Tensor* input, Tensor* target, float delta) {
    if (!input || !target) {
        LOG_ERROR("Huber Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Huber Loss: input and target must have the same number of elements");
        return NULL;
    }

    if (delta <= 0.0f) {
        delta = 1.0f; // Default delta
    }

    LOG_DEBUG("Computing Huber Loss with delta=%.2f", delta);

    size_t n       = input->numel;
    float loss_sum = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float input_val  = tensor_get_float(input, i);
        float target_val = tensor_get_float(target, i);
        float diff       = input_val - target_val;
        float abs_diff   = fabsf(diff);

        if (abs_diff < delta) {
            loss_sum += 0.5f * diff * diff;
        } else {
            loss_sum += delta * abs_diff - 0.5f * delta * delta;
        }
    }

    // Create scalar output tensor
    int shape[]    = {1};
    Tensor* result = tensor_empty(shape, 1, input->dtype, input->device);
    if (!result)
        return NULL;

    tensor_set_float(result, 0, loss_sum / (float)n);

    // Set up autograd if gradients are needed
    if (autograd_is_grad_enabled() && (input->requires_grad || target->requires_grad)) {
        Function* fn = autograd_function_create(OP_HUBER_LOSS, "HuberLoss");
        if (fn) {
            // Save input, target, and delta
            Tensor* saved[] = {input, target};
            autograd_context_save_for_backward(fn->ctx, saved, 2);

            // Save delta value
            float* delta_ptr = CM_MALLOC(sizeof(float));
            if (delta_ptr) {
                *delta_ptr = delta;
                autograd_context_save_data(fn->ctx, delta_ptr, sizeof(float));
            }

            autograd_function_set_backward(fn, huber_loss_backward);
            Tensor* inputs[] = {input, target};
            result           = create_output_with_grad_fn(result, fn, inputs, 2);
        }
    } else {
        result->requires_grad = false;
    }

    return result;
}

// Backward for Huber loss
static void huber_loss_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for Huber Loss operation");

    Tensor* input  = fn->ctx->saved_tensors[0];
    Tensor* target = fn->ctx->saved_tensors[1];

    if (!input || !target) {
        LOG_ERROR("Huber Loss backward: saved tensors are NULL");
        return;
    }

    // Get delta from saved data
    float delta = 1.0f;
    if (fn->ctx->saved_data && fn->ctx->saved_data_size >= sizeof(float)) {
        delta = *(float*)fn->ctx->saved_data;
    }

    float grad_output_val = tensor_get_float(grad_output, 0);
    float scale           = grad_output_val / (float)input->numel;

    // Gradient for input: df/dx depends on |x - y| vs delta
    if (fn->needs_input_grad[0]) {
        Tensor* grad = tensor_empty(input->shape, input->ndim, input->dtype, input->device);
        if (grad) {
            for (size_t i = 0; i < input->numel; i++) {
                float input_val  = tensor_get_float(input, i);
                float target_val = tensor_get_float(target, i);
                float diff       = input_val - target_val;
                float abs_diff   = fabsf(diff);

                float grad_val;
                if (abs_diff < delta) {
                    // Quadratic region: grad = (x - y) * scale
                    grad_val = diff * scale;
                } else {
                    // Linear region: grad = delta * sign(x - y) * scale
                    grad_val = (diff > 0.0f ? delta : -delta) * scale;
                }

                tensor_set_float(grad, i, grad_val);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}

// KL Divergence Loss
Tensor* tensor_kl_div_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("KL Divergence Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("KL Divergence Loss: input and target must have the same number of elements");
        return NULL;
    }

    LOG_DEBUG("Computing KL Divergence Loss");

    size_t n     = input->numel;
    float kl_sum = 0.0f;

    // KL(P||Q) = sum(P * log(P / Q)) = sum(P * log(P) - P * log(Q))
    for (size_t i = 0; i < n; i++) {
        float p_val = tensor_get_float(target, i);
        float q_val = tensor_get_float(input, i);

        // Avoid log(0) and division by zero
        if (p_val <= 0.0f || q_val <= 0.0f) {
            continue; // Skip this element
        }

        kl_sum += p_val * (logf(p_val) - logf(q_val));
    }

    // Create scalar output tensor
    int shape[]    = {1};
    Tensor* result = tensor_empty(shape, 1, input->dtype, input->device);
    if (!result)
        return NULL;

    tensor_set_float(result, 0, kl_sum);

    // Set up autograd if gradients are needed
    if (autograd_is_grad_enabled() && (input->requires_grad || target->requires_grad)) {
        Function* fn = autograd_function_create(OP_KL_DIV_LOSS, "KLDivLoss");
        if (fn) {
            Tensor* saved[] = {input, target};
            autograd_context_save_for_backward(fn->ctx, saved, 2);
            autograd_function_set_backward(fn, kl_div_loss_backward);
            Tensor* inputs[] = {input, target};
            result           = create_output_with_grad_fn(result, fn, inputs, 2);
        }
    } else {
        result->requires_grad = false;
    }

    return result;
}

// Backward for KL divergence loss
static void kl_div_loss_backward(Function* fn, Tensor* grad_output) {
    if (!fn || !grad_output || fn->num_inputs < 2)
        return;

    LOG_DEBUG("Computing backward for KL Divergence Loss operation");

    Tensor* input  = fn->ctx->saved_tensors[0]; // Q (predicted)
    Tensor* target = fn->ctx->saved_tensors[1]; // P (target)

    if (!input || !target) {
        LOG_ERROR("KL Divergence Loss backward: saved tensors are NULL");
        return;
    }

    float grad_output_val = tensor_get_float(grad_output, 0);

    // Gradient for input (Q): d/dQ KL(P||Q) = -P / Q
    if (fn->needs_input_grad[0]) {
        Tensor* grad = tensor_empty(input->shape, input->ndim, input->dtype, input->device);
        if (grad) {
            for (size_t i = 0; i < input->numel; i++) {
                float p_val = tensor_get_float(target, i);
                float q_val = tensor_get_float(input, i);

                float grad_val = 0.0f;
                if (q_val > 0.0f && p_val > 0.0f) {
                    grad_val = -p_val / q_val * grad_output_val;
                }

                tensor_set_float(grad, i, grad_val);
            }
            tensor_accumulate_grad(fn->inputs[0], grad);
            tensor_free(grad);
        }
    }
}
