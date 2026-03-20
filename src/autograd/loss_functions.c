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

Tensor* tensor_mse_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("MSE Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("MSE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    Tensor* diff = uop_sub(input, target);
    if (!diff) return NULL;

    Tensor* squared = uop_mul(diff, diff);
    if (!squared) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(squared, &reduce_params);
}

Tensor* tensor_mae_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("MAE Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("MAE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    Tensor* diff = uop_sub(input, target);
    if (!diff) return NULL;

    Tensor* abs_diff = uop_abs(diff);
    if (!abs_diff) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(abs_diff, &reduce_params);
}

Tensor* tensor_bce_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("BCE Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel && input->numel != 1 && target->numel != 1) {
        LOG_ERROR("BCE Loss: shape mismatch - input: %zu, target: %zu", input->numel,
                  target->numel);
        return NULL;
    }

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* log_input = uop_log(input);
    if (!log_input) return NULL;

    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones) return NULL;
    Tensor* one_minus_input = uop_sub(ones, input);
    if (!one_minus_input) return NULL;

    Tensor* log_1_minus_input = uop_log(one_minus_input);
    if (!log_1_minus_input) return NULL;

    Tensor* term1 = uop_mul(target, log_input);
    if (!term1) return NULL;

    Tensor* ones_target = tensor_ones(target->shape, target->ndim, &config);
    if (!ones_target) return NULL;
    Tensor* one_minus_target = uop_sub(ones_target, target);
    if (!one_minus_target) return NULL;
    Tensor* term2 = uop_mul(one_minus_target, log_1_minus_input);
    if (!term2) return NULL;

    Tensor* combined = uop_add(term1, term2);
    if (!combined) return NULL;

    Tensor* neg_combined = uop_neg(combined);
    if (!neg_combined) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(neg_combined, &reduce_params);
}

Tensor* tensor_cross_entropy_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("Cross Entropy Loss: input or target is NULL");
        return NULL;
    }

    if (target->ndim != 1) {
        LOG_ERROR("Cross Entropy Loss: target must be 1D");
        return NULL;
    }

    if (input->ndim < 2) {
        LOG_ERROR("Cross Entropy Loss: input must be at least 2D [N, C]");
        return NULL;
    }

    size_t n_samples = target->numel;
    size_t input_batch_size = 1;
    for (int i = 0; i < input->ndim - 1; i++)
        input_batch_size *= (size_t)input->shape[i];

    if (input_batch_size != n_samples) {
        LOG_ERROR("Cross Entropy Loss: batch size mismatch - input: %zu, target: %zu",
                  input_batch_size, n_samples);
        return NULL;
    }

    Tensor* softmax_output = tensor_softmax(input, input->ndim - 1);
    if (!softmax_output) return NULL;

    Tensor* log_softmax = tensor_log(softmax_output);
    if (!log_softmax) return NULL;

    Tensor* gathered = uop_gather(log_softmax, target, -1);
    if (!gathered) return NULL;

    Tensor* neg_gathered = uop_neg(gathered);
    if (!neg_gathered) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(neg_gathered, &reduce_params);
}

Tensor* tensor_huber_loss(Tensor* input, Tensor* target, float delta) {
    if (!input || !target) {
        LOG_ERROR("Huber Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Huber Loss: shape mismatch");
        return NULL;
    }

    if (delta <= 0.0f) delta = 1.0f;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* diff = uop_sub(input, target);
    if (!diff) return NULL;

    Tensor* abs_diff = uop_abs(diff);
    if (!abs_diff) return NULL;

    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones) return NULL;

    float delta_array[1] = {delta};
    Tensor* delta_scalar = tensor_from_array_2d(delta_array, 1, 1);
    if (!delta_scalar) return NULL;

    Tensor* delta_tensor = uop_mul(ones, delta_scalar);
    if (!delta_tensor) return NULL;

    Tensor* condition = uop_cmplt(abs_diff, delta_tensor);
    if (!condition) return NULL;

    Tensor* diff_squared = uop_mul(diff, diff);
    if (!diff_squared) return NULL;

    float half_array[1] = {0.5f};
    Tensor* half_scalar = tensor_from_array_2d(half_array, 1, 1);
    if (!half_scalar) return NULL;

    Tensor* ones_for_half = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_half) return NULL;

    Tensor* half_tensor = uop_mul(ones_for_half, half_scalar);
    if (!half_tensor) return NULL;

    Tensor* squared_term = uop_mul(half_tensor, diff_squared);
    if (!squared_term) return NULL;

    Tensor* delta_abs = uop_mul(delta_tensor, abs_diff);
    if (!delta_abs) return NULL;

    float half_delta_sq_array[1] = {0.5f * delta * delta};
    Tensor* half_delta_sq_scalar = tensor_from_array_2d(half_delta_sq_array, 1, 1);
    if (!half_delta_sq_scalar) return NULL;

    Tensor* ones_for_offset = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_offset) return NULL;

    Tensor* half_delta_sq_tensor = uop_mul(ones_for_offset, half_delta_sq_scalar);
    if (!half_delta_sq_tensor) return NULL;

    Tensor* linear_term = uop_sub(delta_abs, half_delta_sq_tensor);
    if (!linear_term) return NULL;

    WhereParams where_params = {.cond = condition, .a = squared_term, .b = linear_term};
    Tensor* loss_per_element = uop_where(&where_params);
    if (!loss_per_element) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(loss_per_element, &reduce_params);
}

Tensor* tensor_hinge_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("Hinge Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Hinge Loss: shape mismatch");
        return NULL;
    }

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones) return NULL;

    float zero_array[1] = {0.0f};
    Tensor* zero_scalar = tensor_from_array_2d(zero_array, 1, 1);
    if (!zero_scalar) return NULL;

    Tensor* ones_for_zeros = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_zeros) return NULL;

    Tensor* zeros = uop_mul(ones_for_zeros, zero_scalar);
    if (!zeros) return NULL;

    Tensor* target_times_input = uop_mul(target, input);
    if (!target_times_input) return NULL;

    Tensor* margin = uop_sub(ones, target_times_input);
    if (!margin) return NULL;

    Tensor* hinge = uop_max(zeros, margin);
    if (!hinge) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(hinge, &reduce_params);
}

Tensor* tensor_focal_loss(Tensor* input, Tensor* target, float alpha, float gamma) {
    if (!input || !target) {
        LOG_ERROR("Focal Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Focal Loss: shape mismatch");
        return NULL;
    }

    if (alpha <= 0.0f) alpha = 0.25f;
    if (gamma <= 0.0f) gamma = 2.0f;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* p = uop_sigmoid(input);
    if (!p) return NULL;

    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones) return NULL;

    Tensor* one_minus_target = uop_sub(ones, target);
    if (!one_minus_target) return NULL;

    Tensor* ones2 = tensor_ones(input->shape, input->ndim, &config);
    if (!ones2) return NULL;

    Tensor* one_minus_p = uop_sub(ones2, p);
    if (!one_minus_p) return NULL;

    Tensor* term1 = uop_mul(target, p);
    if (!term1) return NULL;

    Tensor* term2 = uop_mul(one_minus_target, one_minus_p);
    if (!term2) return NULL;

    Tensor* p_t = uop_add(term1, term2);
    if (!p_t) return NULL;

    Tensor* ones3 = tensor_ones(input->shape, input->ndim, &config);
    if (!ones3) return NULL;

    Tensor* one_minus_pt = uop_sub(ones3, p_t);
    if (!one_minus_pt) return NULL;

    float gamma_array[1] = {gamma};
    Tensor* gamma_scalar = tensor_from_array_2d(gamma_array, 1, 1);
    if (!gamma_scalar) return NULL;

    Tensor* ones_for_gamma = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_gamma) return NULL;

    Tensor* gamma_tensor = uop_mul(ones_for_gamma, gamma_scalar);
    if (!gamma_tensor) return NULL;

    Tensor* focal_weight = uop_pow(one_minus_pt, gamma_tensor);
    if (!focal_weight) return NULL;

    Tensor* ce = uop_neg(uop_log(p_t));
    if (!ce) return NULL;

    Tensor* weighted_ce = uop_mul(focal_weight, ce);
    if (!weighted_ce) return NULL;

    float alpha_array[1] = {alpha};
    Tensor* alpha_scalar = tensor_from_array_2d(alpha_array, 1, 1);
    if (!alpha_scalar) return NULL;

    Tensor* ones_for_alpha = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_alpha) return NULL;

    Tensor* alpha_tensor = uop_mul(ones_for_alpha, alpha_scalar);
    if (!alpha_tensor) return NULL;

    Tensor* loss = uop_mul(alpha_tensor, weighted_ce);
    if (!loss) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(loss, &reduce_params);
}

Tensor* tensor_smooth_l1_loss(Tensor* input, Tensor* target, float beta) {
    if (!input || !target) {
        LOG_ERROR("Smooth L1 Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("Smooth L1 Loss: shape mismatch");
        return NULL;
    }

    if (beta <= 0.0f) beta = 1.0f;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* diff = uop_sub(input, target);
    if (!diff) return NULL;

    Tensor* abs_diff = uop_abs(diff);
    if (!abs_diff) return NULL;

    float beta_array[1] = {beta};
    Tensor* beta_scalar = tensor_from_array_2d(beta_array, 1, 1);
    if (!beta_scalar) return NULL;

    Tensor* ones = tensor_ones(input->shape, input->ndim, &config);
    if (!ones) return NULL;

    Tensor* beta_tensor = uop_mul(ones, beta_scalar);
    if (!beta_tensor) return NULL;

    Tensor* condition = uop_cmplt(abs_diff, beta_tensor);
    if (!condition) return NULL;

    Tensor* diff_squared = uop_mul(diff, diff);
    if (!diff_squared) return NULL;

    float half_array[1] = {0.5f};
    Tensor* half_scalar = tensor_from_array_2d(half_array, 1, 1);
    if (!half_scalar) return NULL;

    Tensor* ones_for_half = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_half) return NULL;

    Tensor* half_tensor = uop_mul(ones_for_half, half_scalar);
    if (!half_tensor) return NULL;

    Tensor* half_diff_squared = uop_mul(half_tensor, diff_squared);
    if (!half_diff_squared) return NULL;

    Tensor* squared_term = uop_div(half_diff_squared, beta_tensor);
    if (!squared_term) return NULL;

    float half_beta_array[1] = {0.5f * beta};
    Tensor* half_beta_scalar = tensor_from_array_2d(half_beta_array, 1, 1);
    if (!half_beta_scalar) return NULL;

    Tensor* ones_for_offset = tensor_ones(input->shape, input->ndim, &config);
    if (!ones_for_offset) return NULL;

    Tensor* half_beta_tensor = uop_mul(ones_for_offset, half_beta_scalar);
    if (!half_beta_tensor) return NULL;

    Tensor* linear_term = uop_sub(abs_diff, half_beta_tensor);
    if (!linear_term) return NULL;

    WhereParams where_params = {.cond = condition, .a = squared_term, .b = linear_term};
    Tensor* loss_per_element = uop_where(&where_params);
    if (!loss_per_element) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(loss_per_element, &reduce_params);
}

Tensor* tensor_kl_div_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("KL Divergence Loss: input or target is NULL");
        return NULL;
    }

    if (input->numel != target->numel) {
        LOG_ERROR("KL Divergence Loss: shape mismatch");
        return NULL;
    }

    Tensor* log_target = uop_log(target);
    if (!log_target) return NULL;

    Tensor* term1 = uop_mul(target, log_target);
    if (!term1) return NULL;

    Tensor* log_input = uop_log(input);
    if (!log_input) return NULL;

    Tensor* term2 = uop_mul(target, log_input);
    if (!term2) return NULL;

    Tensor* diff = uop_sub(term1, term2);
    if (!diff) return NULL;

    ReduceParams reduce_params = {0};
    return uop_sum(diff, &reduce_params);
}

Tensor* tensor_sparse_cross_entropy_loss(Tensor* input, Tensor* target) {
    if (!input || !target) {
        LOG_ERROR("Sparse Cross Entropy Loss: input or target is NULL");
        return NULL;
    }

    if (target->ndim != 1) {
        LOG_ERROR("Sparse Cross Entropy Loss: target must be 1D");
        return NULL;
    }

    if (input->ndim < 2) {
        LOG_ERROR("Sparse Cross Entropy Loss: input must be at least 2D [N, C]");
        return NULL;
    }

    size_t n_samples = target->numel;
    size_t input_batch_size = 1;
    for (int i = 0; i < input->ndim - 1; i++)
        input_batch_size *= (size_t)input->shape[i];

    if (input_batch_size != n_samples) {
        LOG_ERROR("Sparse Cross Entropy Loss: batch size mismatch - input: %zu, target: %zu",
                  input_batch_size, n_samples);
        return NULL;
    }

    Tensor* target_logits = uop_gather(input, target, -1);
    if (!target_logits) return NULL;

    Tensor* exp_logits = uop_exp(input);
    if (!exp_logits) return NULL;

    int sum_dim = input->ndim - 1;
    int sum_dims[] = {sum_dim};
    ReduceParams sum_params = { .dims = sum_dims, .num_dims = 1, .keepdim = false };

    Tensor* sum_exp = uop_sum(exp_logits, &sum_params);
    if (!sum_exp) return NULL;

    Tensor* log_sum_exp = uop_log(sum_exp);
    if (!log_sum_exp) return NULL;

    Tensor* loss_per_sample = uop_add(uop_neg(target_logits), log_sum_exp);
    if (!loss_per_sample) return NULL;

    ReduceParams mean_params = {0};
    return uop_mean(loss_per_sample, &mean_params);
}

Tensor* tensor_triplet_margin_loss(Tensor* anchor, Tensor* positive, Tensor* negative,
                                   float margin) {
    if (!anchor || !positive || !negative) {
        LOG_ERROR("Triplet Margin Loss: anchor, positive, or negative is NULL");
        return NULL;
    }

    if (anchor->numel != positive->numel || anchor->numel != negative->numel) {
        LOG_ERROR("Triplet Margin Loss: shape mismatch");
        return NULL;
    }

    if (margin <= 0.0f) margin = 1.0f;

    Tensor* diff_pos = uop_sub(anchor, positive);
    if (!diff_pos) return NULL;

    Tensor* diff_neg = uop_sub(anchor, negative);
    if (!diff_neg) return NULL;

    Tensor* diff_pos_sq = uop_mul(diff_pos, diff_pos);
    if (!diff_pos_sq) return NULL;

    Tensor* diff_neg_sq = uop_mul(diff_neg, diff_neg);
    if (!diff_neg_sq) return NULL;

    Tensor* dist_diff = uop_sub(diff_pos_sq, diff_neg_sq);
    if (!dist_diff) return NULL;

    ReduceParams reduce_all = {0};
    Tensor* dist_diff_sum = uop_sum(dist_diff, &reduce_all);
    if (!dist_diff_sum) return NULL;

    TensorConfig config = (TensorConfig){
        .dtype = anchor->dtype, .device = anchor->device, .has_dtype = true, .has_device = true};
    int scalar_shape[] = {1};
    Tensor* margin_tensor = tensor_full(scalar_shape, 1, &config, margin);
    if (!margin_tensor) return NULL;

    Tensor* with_margin = uop_add(dist_diff_sum, margin_tensor);
    if (!with_margin) return NULL;

    return uop_clamp(with_margin, 0.0f, INFINITY);
}

Tensor* tensor_cosine_embedding_loss(Tensor* x1, Tensor* x2, Tensor* target, float margin) {
    if (!x1 || !x2 || !target) {
        LOG_ERROR("Cosine Embedding Loss: x1, x2, or target is NULL");
        return NULL;
    }

    if (x1->numel != x2->numel) {
        LOG_ERROR("Cosine Embedding Loss: x1 and x2 shape mismatch");
        return NULL;
    }

    TensorConfig config = (TensorConfig){
        .dtype = x1->dtype, .device = x1->device, .has_dtype = true, .has_device = true};
    ReduceParams reduce_all = {0};
    int scalar_shape[] = {1};

    Tensor* x1_x2 = uop_mul(x1, x2);
    if (!x1_x2) return NULL;
    Tensor* dot = uop_sum(x1_x2, &reduce_all);
    if (!dot) return NULL;

    Tensor* x1_sq = uop_mul(x1, x1);
    if (!x1_sq) return NULL;
    Tensor* x1_norm = uop_sqrt(uop_sum(x1_sq, &reduce_all));
    if (!x1_norm) return NULL;

    Tensor* x2_sq = uop_mul(x2, x2);
    if (!x2_sq) return NULL;
    Tensor* x2_norm = uop_sqrt(uop_sum(x2_sq, &reduce_all));
    if (!x2_norm) return NULL;

    Tensor* cos_sim = uop_div(dot, uop_mul(x1_norm, x2_norm));
    if (!cos_sim) return NULL;

    Tensor* ones = tensor_full(scalar_shape, 1, &config, 1.0f);
    if (!ones) return NULL;
    Tensor* pos_loss = uop_sub(ones, cos_sim);
    if (!pos_loss) return NULL;

    Tensor* margin_tensor = tensor_full(scalar_shape, 1, &config, margin);
    if (!margin_tensor) return NULL;
    Tensor* neg_loss = uop_clamp(uop_sub(cos_sim, margin_tensor), 0.0f, INFINITY);
    if (!neg_loss) return NULL;

    Tensor* half = tensor_full(scalar_shape, 1, &config, 0.5f);
    if (!half) return NULL;
    Tensor* ones2 = tensor_full(scalar_shape, 1, &config, 1.0f);
    if (!ones2) return NULL;
    Tensor* weight_pos = uop_mul(uop_add(ones2, target), half);
    if (!weight_pos) return NULL;

    Tensor* half2 = tensor_full(scalar_shape, 1, &config, 0.5f);
    if (!half2) return NULL;
    Tensor* ones3 = tensor_full(scalar_shape, 1, &config, 1.0f);
    if (!ones3) return NULL;
    Tensor* weight_neg = uop_mul(uop_sub(ones3, target), half2);
    if (!weight_neg) return NULL;

    return uop_add(uop_mul(weight_pos, pos_loss), uop_mul(weight_neg, neg_loss));
}

Tensor* tensor_cross_entropy_loss_smooth(Tensor* input, Tensor* target,
                                          float label_smoothing) {
    if (!input || !target) {
        LOG_ERROR("Cross Entropy Loss (smooth): input or target is NULL");
        return NULL;
    }

    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        LOG_ERROR("Cross Entropy Loss (smooth): label_smoothing must be in [0, 1]");
        return NULL;
    }

    if (label_smoothing == 0.0f)
        return tensor_cross_entropy_loss(input, target);

    if (target->ndim != 1) {
        LOG_ERROR("Cross Entropy Loss (smooth): target must be 1D");
        return NULL;
    }

    if (input->ndim < 2) {
        LOG_ERROR("Cross Entropy Loss (smooth): input must be at least 2D [N, C]");
        return NULL;
    }

    size_t n_samples = target->numel;
    size_t input_batch_size = 1;
    for (int i = 0; i < input->ndim - 1; i++)
        input_batch_size *= (size_t)input->shape[i];

    if (input_batch_size != n_samples) {
        LOG_ERROR("Cross Entropy Loss (smooth): batch size mismatch - input: %zu, target: %zu",
                  input_batch_size, n_samples);
        return NULL;
    }

    int num_classes = input->shape[input->ndim - 1];

    Tensor* softmax_output = tensor_softmax(input, input->ndim - 1);
    if (!softmax_output) return NULL;

    Tensor* log_softmax = tensor_log(softmax_output);
    if (!log_softmax) return NULL;

    Tensor* gathered = uop_gather(log_softmax, target, -1);
    if (!gathered) return NULL;

    Tensor* neg_gathered = uop_neg(gathered);
    if (!neg_gathered) return NULL;

    ReduceParams ce_reduce = {0};
    Tensor* ce_loss = uop_mean(neg_gathered, &ce_reduce);
    if (!ce_loss) return NULL;

    Tensor* neg_log_softmax = uop_neg(log_softmax);
    if (!neg_log_softmax) return NULL;

    ReduceParams all_reduce = {0};
    Tensor* sum_neg_log = uop_mean(neg_log_softmax, &all_reduce);
    if (!sum_neg_log) return NULL;

    float inv_k = 1.0f / (float)num_classes;
    float inv_k_arr[1] = {inv_k};
    Tensor* inv_k_t = tensor_from_array_2d(inv_k_arr, 1, 1);
    if (!inv_k_t) return NULL;

    Tensor* uniform_loss = uop_mul(sum_neg_log, inv_k_t);
    if (!uniform_loss) return NULL;

    float eps = label_smoothing;
    float one_minus_eps_arr[1] = {1.0f - eps};
    Tensor* one_minus_eps_t = tensor_from_array_2d(one_minus_eps_arr, 1, 1);
    if (!one_minus_eps_t) return NULL;

    float eps_arr[1] = {eps};
    Tensor* eps_t = tensor_from_array_2d(eps_arr, 1, 1);
    if (!eps_t) return NULL;

    Tensor* weighted_ce = uop_mul(one_minus_eps_t, ce_loss);
    if (!weighted_ce) return NULL;

    Tensor* weighted_uniform = uop_mul(eps_t, uniform_loss);
    if (!weighted_uniform) return NULL;

    return uop_add(weighted_ce, weighted_uniform);
}

Tensor* tensor_sparse_cross_entropy_loss_smooth(Tensor* input, Tensor* target,
                                                 float label_smoothing) {
    if (!input || !target) {
        LOG_ERROR("Sparse Cross Entropy Loss (smooth): input or target is NULL");
        return NULL;
    }

    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        LOG_ERROR("Sparse Cross Entropy Loss (smooth): label_smoothing must be in [0, 1]");
        return NULL;
    }

    if (label_smoothing == 0.0f)
        return tensor_sparse_cross_entropy_loss(input, target);

    if (target->ndim != 1) {
        LOG_ERROR("Sparse Cross Entropy Loss (smooth): target must be 1D");
        return NULL;
    }

    if (input->ndim < 2) {
        LOG_ERROR("Sparse Cross Entropy Loss (smooth): input must be at least 2D [N, C]");
        return NULL;
    }

    size_t n_samples = target->numel;
    size_t input_batch_size = 1;
    for (int i = 0; i < input->ndim - 1; i++)
        input_batch_size *= (size_t)input->shape[i];

    if (input_batch_size != n_samples) {
        LOG_ERROR("Sparse Cross Entropy Loss (smooth): batch size mismatch - input: %zu, target: %zu",
                  input_batch_size, n_samples);
        return NULL;
    }

    int num_classes = input->shape[input->ndim - 1];

    Tensor* target_logits = uop_gather(input, target, -1);
    if (!target_logits) return NULL;

    Tensor* exp_logits = uop_exp(input);
    if (!exp_logits) return NULL;

    int sum_dim = input->ndim - 1;
    int sum_dims[] = {sum_dim};
    ReduceParams sum_params = { .dims = sum_dims, .num_dims = 1, .keepdim = false };

    Tensor* sum_exp = uop_sum(exp_logits, &sum_params);
    if (!sum_exp) return NULL;

    Tensor* log_sum_exp = uop_log(sum_exp);
    if (!log_sum_exp) return NULL;

    Tensor* loss_per_sample = uop_add(uop_neg(target_logits), log_sum_exp);
    if (!loss_per_sample) return NULL;

    ReduceParams mean_params = {0};
    Tensor* ce_loss = uop_mean(loss_per_sample, &mean_params);
    if (!ce_loss) return NULL;

    Tensor* neg_input = uop_neg(input);
    if (!neg_input) return NULL;

    Tensor* exp_for_lse = uop_exp(input);
    if (!exp_for_lse) return NULL;

    Tensor* sum_exp_2 = uop_sum(exp_for_lse, &sum_params);
    if (!sum_exp_2) return NULL;

    Tensor* log_sum_exp_2 = uop_log(sum_exp_2);
    if (!log_sum_exp_2) return NULL;

    ReduceParams class_reduce = { .dims = sum_dims, .num_dims = 1, .keepdim = false };
    Tensor* input_sum = uop_sum(input, &class_reduce);
    if (!input_sum) return NULL;

    float k_f = (float)num_classes;
    float k_arr[1] = {k_f};
    Tensor* k_t = tensor_from_array_2d(k_arr, 1, 1);
    if (!k_t) return NULL;

    Tensor* scaled_lse = uop_mul(k_t, log_sum_exp_2);
    if (!scaled_lse) return NULL;

    Tensor* uniform_per_sample = uop_sub(scaled_lse, input_sum);
    if (!uniform_per_sample) return NULL;

    float inv_k = 1.0f / k_f;
    float inv_k_arr[1] = {inv_k};
    Tensor* inv_k_t = tensor_from_array_2d(inv_k_arr, 1, 1);
    if (!inv_k_t) return NULL;

    Tensor* uniform_scaled = uop_mul(inv_k_t, uniform_per_sample);
    if (!uniform_scaled) return NULL;

    ReduceParams uniform_mean = {0};
    Tensor* uniform_loss = uop_mean(uniform_scaled, &uniform_mean);
    if (!uniform_loss) return NULL;

    float eps = label_smoothing;
    float one_minus_eps_arr[1] = {1.0f - eps};
    Tensor* one_minus_eps_t = tensor_from_array_2d(one_minus_eps_arr, 1, 1);
    if (!one_minus_eps_t) return NULL;

    float eps_arr[1] = {eps};
    Tensor* eps_t = tensor_from_array_2d(eps_arr, 1, 1);
    if (!eps_t) return NULL;

    Tensor* weighted_ce = uop_mul(one_minus_eps_t, ce_loss);
    if (!weighted_ce) return NULL;

    Tensor* weighted_uniform = uop_mul(eps_t, uniform_loss);
    if (!weighted_uniform) return NULL;

    return uop_add(weighted_ce, weighted_uniform);
}

Tensor* tensor_nll_loss(Tensor* log_probs, Tensor* targets) {
    if (!log_probs || !targets) {
        LOG_ERROR("NLL Loss: log_probs or targets is NULL");
        return NULL;
    }

    if (targets->ndim != 1) {
        LOG_ERROR("NLL Loss: targets must be 1D");
        return NULL;
    }

    if (log_probs->ndim < 2) {
        LOG_ERROR("NLL Loss: log_probs must be at least 2D [N, C]");
        return NULL;
    }

    size_t n_samples = targets->numel;
    size_t input_batch_size = 1;
    for (int i = 0; i < log_probs->ndim - 1; i++)
        input_batch_size *= (size_t)log_probs->shape[i];

    if (input_batch_size != n_samples) {
        LOG_ERROR("NLL Loss: batch size mismatch - log_probs: %zu, targets: %zu",
                  input_batch_size, n_samples);
        return NULL;
    }

    Tensor* gathered = uop_gather(log_probs, targets, -1);
    if (!gathered) return NULL;

    Tensor* neg_gathered = uop_neg(gathered);
    if (!neg_gathered) return NULL;

    ReduceParams reduce_params = {0};
    return uop_mean(neg_gathered, &reduce_params);
}
