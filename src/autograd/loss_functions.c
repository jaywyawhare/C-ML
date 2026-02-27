/**
 * @file loss_functions.c
 * @brief Loss functions implementation with autograd support
 *
 * This file implements loss functions with automatic differentiation.
 * All loss functions support automatic differentiation.
 */

#include "autograd/autograd.h"
#include "autograd/loss_functions.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// Mean Squared Error Loss: mean((input - target)^2)
// Implemented using IR: diff = sub(input, target), squared = mul(diff, diff), mean(squared)

Tensor* tensor_mse_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("MSE Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing MSE Loss using IR");

    // Check shape compatibility
    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("MSE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    // MSE = mean((input - target)^2)
    // Step 1: diff = input - target
    Tensor* diff = uop_sub(input, target);
    if (!diff) {
        LOG_ERROR("MSE Loss: failed to compute diff");
        return NULL;
    }

    // Step 2: squared = diff * diff
    Tensor* squared = uop_mul(diff, diff);
    if (!squared) {
        LOG_ERROR("MSE Loss: failed to compute squared diff");
        return NULL;
    }

    // Step 3: result = mean(squared) - reduce all dimensions
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0; // Reduce all dimensions
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(squared, &reduce_params);

    if (!result) {
        LOG_ERROR("MSE Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("MSE Loss computed using IR (lazy)");
    return result;
}

// Mean Absolute Error Loss (L1 Loss): mean(|input - target|)
// Implemented using IR: diff = sub(input, target), abs_diff = abs(diff), mean(abs_diff)

Tensor* tensor_mae_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("MAE Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing MAE Loss using IR");

    // Check shape compatibility
    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("MAE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    // MAE = mean(|input - target|)
    // Step 1: diff = input - target
    Tensor* diff = uop_sub(input, target);
    if (!diff) {
        LOG_ERROR("MAE Loss: failed to compute diff");
        return NULL;
    }

    // Step 2: abs_diff = |diff|
    Tensor* abs_diff = uop_abs(diff);
    if (!abs_diff) {
        LOG_ERROR("MAE Loss: failed to compute abs");
        return NULL;
    }

    // Step 3: result = mean(abs_diff) - reduce all dimensions
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0; // Reduce all dimensions
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(abs_diff, &reduce_params);

    if (!result) {
        LOG_ERROR("MAE Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("MAE Loss computed using IR (lazy)");
    return result;
}

// Binary Cross Entropy Loss: -mean(target * log(input) + (1-target) * log(1-input))
// Note: Clamping to prevent log(0) is handled at execution time
// Implemented using IR: log_input = log(input), log_1_minus_input = log(1-input),
//                       term = target * log_input + (1-target) * log_1_minus_input,
//                       result = -mean(term)

Tensor* tensor_bce_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("BCE Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing BCE Loss using IR");

    // Check shape compatibility
    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("BCE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    // BCE = -mean(target * log(input) + (1 - target) * log(1 - input))
    // Note: For now, we use IR operations. Clamping for numerical stability
    // would ideally be handled by a clamp uop, but we'll rely on IR execution
    // to handle edge cases.

    // log(input)
    Tensor* log_input = uop_log(input);
    if (!log_input) {
        LOG_ERROR("BCE Loss: failed to compute log(input)");
        return NULL;
    }

    // 1 - input
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones) {
        // Don't free log_input - it's part of the IR graph
        return NULL;
    }
    Tensor* one_minus_input = uop_sub(ones, input);
    // Don't free ones - it's part of the IR graph
    if (!one_minus_input) {
        // Don't free log_input - it's part of the IR graph
        return NULL;
    }

    // log(1 - input)
    Tensor* log_1_minus_input = uop_log(one_minus_input);
    // Don't free one_minus_input - it's part of the IR graph
    if (!log_1_minus_input) {
        // Don't free log_input - it's part of the IR graph
        return NULL;
    }

    // target * log(input)
    Tensor* term1 = uop_mul(target, log_input);
    // Don't free log_input - it's part of the IR graph
    if (!term1) {
        // Don't free log_1_minus_input - it's part of the IR graph
        return NULL;
    }

    // (1 - target) * log(1 - input)
    Tensor* ones_target = tensor_ones(target->shape, target->ndim, &config);
    if (!ones_target) {
        // Don't free term1 and log_1_minus_input - they're part of the IR graph
        return NULL;
    }
    Tensor* one_minus_target = uop_sub(ones_target, target);
    // Don't free ones_target - it's part of the IR graph
    if (!one_minus_target) {
        // Don't free term1 and log_1_minus_input - they're part of the IR graph
        return NULL;
    }
    Tensor* term2 = uop_mul(one_minus_target, log_1_minus_input);
    // Don't free one_minus_target and log_1_minus_input - they're part of the IR graph
    if (!term2) {
        // Don't free term1 - it's part of the IR graph
        return NULL;
    }

    // term1 + term2
    Tensor* combined = uop_add(term1, term2);
    // Don't free term1 and term2 - they're part of the IR graph
    if (!combined) {
        return NULL;
    }

    // -mean(combined)
    Tensor* neg_combined = uop_neg(combined);
    // Don't free combined - it's part of the IR graph
    if (!neg_combined) {
        return NULL;
    }

    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0; // Reduce all dimensions
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(neg_combined, &reduce_params);
    // Don't free neg_combined - it's part of the IR graph

    if (!result) {
        LOG_ERROR("BCE Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("BCE Loss computed using IR (lazy)");
    return result;
}

// Cross Entropy Loss (FULLY LAZY)
// Input: logits [N, C] where N is batch size, C is number of classes
// Target: class indices [N] with values in [0, C-1]
// Formula: -mean(log(softmax(logits))[target]) = mean(-gather(log_softmax, target, dim=-1))
//
// Implementation uses lazy IR operations throughout:
// 1. softmax(input, dim=-1) -> [N, C]
// 2. log(softmax) -> [N, C]
// 3. gather(log_softmax, target, dim=-1) -> [N]  (selects log_softmax[i, target[i]])
// 4. neg(gathered) -> [N]
// 5. mean(neg_gathered) -> scalar

Tensor* tensor_cross_entropy_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("Cross Entropy Loss: input or target is NULL");
        return NULL;
    }

    LOG_DEBUG("Computing Cross Entropy Loss using IR (lazy)");

    // Validate input shapes
    if (target->ndim != 1) {
        LOG_ERROR("Cross Entropy Loss: target must be 1D (class indices)");
        return NULL;
    }

    if (input->ndim < 2) {
        LOG_ERROR("Cross Entropy Loss: input must be at least 2D [N, C]");
        return NULL;
    }

    size_t n_samples = target->numel;

    // Check if batch sizes match
    size_t input_batch_size = 1;
    for (int i = 0; i < input->ndim - 1; i++) {
        input_batch_size *= (size_t)input->shape[i];
    }

    if (input_batch_size != n_samples) {
        LOG_ERROR("Cross Entropy Loss: batch size mismatch. Input batch: %zu, target batch: %zu",
                  input_batch_size, n_samples);
        return NULL;
    }

    // Step 1: Apply softmax along the last dimension (class dimension) - LAZY
    int softmax_dim        = input->ndim - 1;
    Tensor* softmax_output = tensor_softmax(input, softmax_dim);
    if (!softmax_output) {
        LOG_ERROR("Cross Entropy Loss: failed to compute softmax");
        return NULL;
    }

    // Step 2: Compute log-softmax for numerical stability - LAZY
    Tensor* log_softmax = tensor_log(softmax_output);
    // Don't free softmax_output - it's part of the IR graph
    if (!log_softmax) {
        LOG_ERROR("Cross Entropy Loss: failed to compute log-softmax");
        return NULL;
    }

    // Step 3: Gather log probabilities for target classes - LAZY
    // This selects log_softmax[i, target[i]] for each sample i
    Tensor* gathered = uop_gather(log_softmax, target, -1);
    // Don't free log_softmax - it's part of the IR graph
    if (!gathered) {
        LOG_ERROR("Cross Entropy Loss: failed to gather log probabilities");
        return NULL;
    }

    // Step 4: Negate (NLL = -log_prob) - LAZY
    Tensor* neg_gathered = uop_neg(gathered);
    // Don't free gathered - it's part of the IR graph
    if (!neg_gathered) {
        LOG_ERROR("Cross Entropy Loss: failed to negate log probabilities");
        return NULL;
    }

    // Step 5: Compute mean over all samples - LAZY
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0; // Reduce all dimensions
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(neg_gathered, &reduce_params);
    // Don't free neg_gathered - it's part of the IR graph

    if (!result) {
        LOG_ERROR("Cross Entropy Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("Cross Entropy Loss computed using IR (lazy)");
    return result;
}

// Formula: where(|diff| < delta, 0.5*diff^2, delta*|diff| - 0.5*delta^2)
// All intermediate tensors are part of the IR graph and should NOT be freed manually
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

    LOG_DEBUG("Computing Huber Loss with delta=%.2f using IR", (double)delta);

    // Huber Loss: where(|diff| < delta, 0.5*diff^2, delta*|diff| - 0.5*delta^2)
    // diff = input - target
    Tensor* diff = uop_sub(input, target);
    if (!diff)
        return NULL;

    // abs_diff = |diff|
    Tensor* abs_diff = uop_abs(diff);
    if (!abs_diff)
        return NULL;

    // Create delta tensor for comparison and computation
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones)
        return NULL;

    // delta_tensor = delta * ones (broadcasts delta to input shape)
    float delta_array[1] = {delta};
    Tensor* delta_scalar = tensor_from_array_2d(delta_array, 1, 1);
    if (!delta_scalar)
        return NULL;

    Tensor* delta_tensor = uop_mul(ones, delta_scalar);
    // Don't free ones or delta_scalar - they're part of the IR graph
    if (!delta_tensor)
        return NULL;

    // condition = abs_diff < delta
    Tensor* condition = uop_cmplt(abs_diff, delta_tensor);
    if (!condition)
        return NULL;

    // squared_term = 0.5 * diff^2
    Tensor* diff_squared = uop_mul(diff, diff);
    if (!diff_squared)
        return NULL;

    float half_array[1] = {0.5f};
    Tensor* half_scalar = tensor_from_array_2d(half_array, 1, 1);
    if (!half_scalar)
        return NULL;

    Tensor* ones_for_half = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_half)
        return NULL;

    Tensor* half_tensor = uop_mul(ones_for_half, half_scalar);
    // Don't free ones_for_half or half_scalar - they're part of the IR graph
    if (!half_tensor)
        return NULL;

    Tensor* squared_term = uop_mul(half_tensor, diff_squared);
    // Don't free half_tensor or diff_squared - they're part of the IR graph
    if (!squared_term)
        return NULL;

    // linear_term = delta * abs_diff - 0.5 * delta^2
    Tensor* delta_abs = uop_mul(delta_tensor, abs_diff);
    if (!delta_abs)
        return NULL;

    float half_delta_sq          = 0.5f * delta * delta;
    float half_delta_sq_array[1] = {half_delta_sq};
    Tensor* half_delta_sq_scalar = tensor_from_array_2d(half_delta_sq_array, 1, 1);
    if (!half_delta_sq_scalar)
        return NULL;

    Tensor* ones_for_half_delta_sq = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_half_delta_sq)
        return NULL;

    Tensor* half_delta_sq_tensor = uop_mul(ones_for_half_delta_sq, half_delta_sq_scalar);
    // Don't free - they're part of the IR graph
    if (!half_delta_sq_tensor)
        return NULL;

    Tensor* linear_term = uop_sub(delta_abs, half_delta_sq_tensor);
    // Don't free delta_abs or half_delta_sq_tensor - they're part of the IR graph
    if (!linear_term)
        return NULL;

    // result = where(condition, squared_term, linear_term)
    WhereParams where_params = {.cond = condition, .a = squared_term, .b = linear_term};
    Tensor* loss_per_element = uop_where(&where_params);
    // Don't free condition, squared_term, linear_term - they're part of the IR graph

    if (!loss_per_element)
        return NULL;

    // Take mean over all elements
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0;
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(loss_per_element, &reduce_params);
    // Don't free loss_per_element - it's part of the IR graph

    LOG_DEBUG("Huber Loss computed using IR (lazy)");
    return result;
}

// Hinge Loss: mean(max(0, 1 - target * input))
// Used for binary classification where target is +1 or -1
// All intermediate tensors are part of the IR graph and should NOT be freed manually
Tensor* tensor_hinge_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("Hinge Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Hinge Loss: input and target must have the same number of elements");
        return NULL;
    }

    LOG_DEBUG("Computing Hinge Loss using IR");

    // Hinge Loss = mean(max(0, 1 - target * input))

    // Create ones tensor for the margin
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones)
        return NULL;

    // Create zeros tensor for max(0, ...)
    float zero_array[1] = {0.0f};
    Tensor* zero_scalar = tensor_from_array_2d(zero_array, 1, 1);
    if (!zero_scalar)
        return NULL;

    Tensor* ones_for_zeros = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_zeros)
        return NULL;

    Tensor* zeros = uop_mul(ones_for_zeros, zero_scalar);
    // Don't free ones_for_zeros or zero_scalar - they're part of the IR graph
    if (!zeros)
        return NULL;

    // target * input
    Tensor* target_times_input = uop_mul(target, input);
    if (!target_times_input)
        return NULL;

    // margin = 1 - target * input
    Tensor* margin = uop_sub(ones, target_times_input);
    // Don't free ones or target_times_input - they're part of the IR graph
    if (!margin)
        return NULL;

    // hinge = max(0, margin)
    Tensor* hinge = uop_max(zeros, margin);
    // Don't free zeros or margin - they're part of the IR graph
    if (!hinge)
        return NULL;

    // result = mean(hinge)
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0;
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(hinge, &reduce_params);
    // Don't free hinge - it's part of the IR graph

    if (!result) {
        LOG_ERROR("Hinge Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("Hinge Loss computed using IR (lazy)");
    return result;
}

// Focal Loss: -mean(alpha * (1 - p_t)^gamma * log(p_t))
// where p_t = target * p + (1 - target) * (1 - p), p = sigmoid(input)
// All intermediate tensors are part of the IR graph and should NOT be freed manually
Tensor* tensor_focal_loss(Tensor* input, Tensor* target, float alpha, float gamma) {
    if (!input || !target) {
        LOG_ERROR("Focal Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Focal Loss: input and target must have the same number of elements");
        return NULL;
    }

    if (alpha <= 0.0f) {
        alpha = 0.25f; // Default alpha
    }

    if (gamma <= 0.0f) {
        gamma = 2.0f; // Default gamma
    }

    LOG_DEBUG("Computing Focal Loss with alpha=%.2f, gamma=%.2f using IR", (double)alpha,
              (double)gamma);

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    // p = sigmoid(input)
    Tensor* p = uop_sigmoid(input);
    if (!p)
        return NULL;

    // ones tensor
    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones)
        return NULL;

    // 1 - target
    Tensor* one_minus_target = uop_sub(ones, target);
    // Don't free ones - it's part of the IR graph
    if (!one_minus_target)
        return NULL;

    // 1 - p
    Tensor* ones2 = tensor_ones(input->shape, input->ndim, &config);
    if (!ones2)
        return NULL;

    Tensor* one_minus_p = uop_sub(ones2, p);
    // Don't free ones2 or p - they're part of the IR graph
    if (!one_minus_p)
        return NULL;

    // p_t = target * p + (1 - target) * (1 - p)
    Tensor* term1 = uop_mul(target, p);
    if (!term1)
        return NULL;

    Tensor* term2 = uop_mul(one_minus_target, one_minus_p);
    // Don't free one_minus_target or one_minus_p - they're part of the IR graph
    if (!term2)
        return NULL;

    Tensor* p_t = uop_add(term1, term2);
    // Don't free term1 or term2 - they're part of the IR graph
    if (!p_t)
        return NULL;

    // (1 - p_t)
    Tensor* ones3 = tensor_ones(input->shape, input->ndim, &config);
    if (!ones3)
        return NULL;

    Tensor* one_minus_pt = uop_sub(ones3, p_t);
    // Don't free ones3 - it's part of the IR graph
    if (!one_minus_pt)
        return NULL;

    // (1 - p_t)^gamma - create gamma tensor
    float gamma_array[1] = {gamma};
    Tensor* gamma_scalar = tensor_from_array_2d(gamma_array, 1, 1);
    if (!gamma_scalar)
        return NULL;

    Tensor* ones_for_gamma = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_gamma)
        return NULL;

    Tensor* gamma_tensor = uop_mul(ones_for_gamma, gamma_scalar);
    // Don't free ones_for_gamma or gamma_scalar - they're part of the IR graph
    if (!gamma_tensor)
        return NULL;

    Tensor* focal_weight = uop_pow(one_minus_pt, gamma_tensor);
    // Don't free one_minus_pt or gamma_tensor - they're part of the IR graph
    if (!focal_weight)
        return NULL;

    // log(p_t)
    Tensor* log_pt = uop_log(p_t);
    // Don't free p_t - it's part of the IR graph
    if (!log_pt)
        return NULL;

    // ce = -log(p_t)
    Tensor* ce = uop_neg(log_pt);
    // Don't free log_pt - it's part of the IR graph
    if (!ce)
        return NULL;

    // focal_weight * ce
    Tensor* weighted_ce = uop_mul(focal_weight, ce);
    // Don't free focal_weight or ce - they're part of the IR graph
    if (!weighted_ce)
        return NULL;

    // alpha * focal_weight * ce - create alpha tensor
    float alpha_array[1] = {alpha};
    Tensor* alpha_scalar = tensor_from_array_2d(alpha_array, 1, 1);
    if (!alpha_scalar)
        return NULL;

    Tensor* ones_for_alpha = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_alpha)
        return NULL;

    Tensor* alpha_tensor = uop_mul(ones_for_alpha, alpha_scalar);
    // Don't free ones_for_alpha or alpha_scalar - they're part of the IR graph
    if (!alpha_tensor)
        return NULL;

    Tensor* loss = uop_mul(alpha_tensor, weighted_ce);
    // Don't free alpha_tensor or weighted_ce - they're part of the IR graph
    if (!loss)
        return NULL;

    // result = mean(loss)
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0;
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(loss, &reduce_params);
    // Don't free loss - it's part of the IR graph

    if (!result) {
        LOG_ERROR("Focal Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("Focal Loss computed using IR (lazy)");
    return result;
}

// Smooth L1 Loss: where(|x| < beta, 0.5 * x^2 / beta, |x| - 0.5 * beta)
// where x = input - target
// All intermediate tensors are part of the IR graph and should NOT be freed manually
Tensor* tensor_smooth_l1_loss(Tensor* input, Tensor* target, float beta) {
    if (!input || !target) {
        LOG_ERROR("Smooth L1 Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Smooth L1 Loss: input and target must have the same number of elements");
        return NULL;
    }

    if (beta <= 0.0f) {
        beta = 1.0f; // Default beta
    }

    LOG_DEBUG("Computing Smooth L1 Loss with beta=%.2f using IR", (double)beta);

    // Smooth L1 Loss: where(|diff| < beta, 0.5 * diff^2 / beta, |diff| - 0.5 * beta)
    // diff = input - target
    Tensor* diff = uop_sub(input, target);
    if (!diff)
        return NULL;

    // abs_diff = |diff|
    Tensor* abs_diff = uop_abs(diff);
    if (!abs_diff)
        return NULL;

    // Create beta tensor
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    float beta_array[1] = {beta};
    Tensor* beta_scalar = tensor_from_array_2d(beta_array, 1, 1);
    if (!beta_scalar)
        return NULL;

    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones)
        return NULL;

    Tensor* beta_tensor = uop_mul(ones, beta_scalar);
    // Don't free ones or beta_scalar - they're part of the IR graph
    if (!beta_tensor)
        return NULL;

    // condition = abs_diff < beta
    Tensor* condition = uop_cmplt(abs_diff, beta_tensor);
    if (!condition)
        return NULL;

    // squared_term = 0.5 * diff^2 / beta
    Tensor* diff_squared = uop_mul(diff, diff);
    if (!diff_squared)
        return NULL;

    float half_array[1] = {0.5f};
    Tensor* half_scalar = tensor_from_array_2d(half_array, 1, 1);
    if (!half_scalar)
        return NULL;

    Tensor* ones_for_half = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_half)
        return NULL;

    Tensor* half_tensor = uop_mul(ones_for_half, half_scalar);
    // Don't free ones_for_half or half_scalar - they're part of the IR graph
    if (!half_tensor)
        return NULL;

    Tensor* half_diff_squared = uop_mul(half_tensor, diff_squared);
    // Don't free half_tensor or diff_squared - they're part of the IR graph
    if (!half_diff_squared)
        return NULL;

    Tensor* squared_term = uop_div(half_diff_squared, beta_tensor);
    // Don't free half_diff_squared or beta_tensor - they're part of the IR graph
    if (!squared_term)
        return NULL;

    // linear_term = |diff| - 0.5 * beta
    float half_beta          = 0.5f * beta;
    float half_beta_array[1] = {half_beta};
    Tensor* half_beta_scalar = tensor_from_array_2d(half_beta_array, 1, 1);
    if (!half_beta_scalar)
        return NULL;

    Tensor* ones_for_half_beta = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_half_beta)
        return NULL;

    Tensor* half_beta_tensor = uop_mul(ones_for_half_beta, half_beta_scalar);
    // Don't free ones_for_half_beta or half_beta_scalar - they're part of the IR graph
    if (!half_beta_tensor)
        return NULL;

    Tensor* linear_term = uop_sub(abs_diff, half_beta_tensor);
    // Don't free abs_diff or half_beta_tensor - they're part of the IR graph
    if (!linear_term)
        return NULL;

    // result = where(condition, squared_term, linear_term)
    WhereParams where_params = {.cond = condition, .a = squared_term, .b = linear_term};
    Tensor* loss_per_element = uop_where(&where_params);
    // Don't free condition, squared_term, linear_term - they're part of the IR graph

    if (!loss_per_element)
        return NULL;

    // Take mean over all elements
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0;
    reduce_params.keepdim      = false;
    Tensor* result             = uop_mean(loss_per_element, &reduce_params);
    // Don't free loss_per_element - it's part of the IR graph

    if (!result) {
        LOG_ERROR("Smooth L1 Loss: failed to compute mean");
        return NULL;
    }

    LOG_DEBUG("Smooth L1 Loss computed using IR (lazy)");
    return result;
}

// Formula: sum(target * log(target / input)) = sum(target * log(target) - target * log(input))
// All intermediate tensors are part of the IR graph and should NOT be freed manually
Tensor* tensor_kl_div_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("KL Divergence Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("KL Divergence Loss: input and target must have the same number of elements");
        return NULL;
    }

    LOG_DEBUG("Computing KL Divergence Loss using IR");

    // KL Divergence Loss using IR: sum(target * log(target / input)) = sum(target * log(target) -
    // target * log(input)) KL(P||Q) = sum(P * log(P) - P * log(Q))

    // log(target)
    Tensor* log_target = uop_log(target);
    if (!log_target)
        return NULL;

    // target * log(target)
    Tensor* term1 = uop_mul(target, log_target);
    // Don't free log_target - it's part of the IR graph
    if (!term1)
        return NULL;

    // log(input)
    Tensor* log_input = uop_log(input);
    if (!log_input)
        return NULL;

    // target * log(input)
    Tensor* term2 = uop_mul(target, log_input);
    // Don't free log_input - it's part of the IR graph
    if (!term2)
        return NULL;

    // term1 - term2
    Tensor* diff = uop_sub(term1, term2);
    // Don't free term1 and term2 - they're part of the IR graph
    if (!diff)
        return NULL;

    // sum(diff)
    ReduceParams reduce_params = {0};
    reduce_params.dims         = NULL;
    reduce_params.num_dims     = 0;
    reduce_params.keepdim      = false;
    Tensor* result             = uop_sum(diff, &reduce_params);
    // Don't free diff - it's part of the IR graph

    LOG_DEBUG("KL Divergence Loss computed using IR (lazy)");
    return result;
}
