#include "nn/layers/batchnorm3d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* batchnorm3d_forward(Module* module, Tensor* input) {
    BatchNorm3d* bn = (BatchNorm3d*)module;

    if (!bn || !input)
        return NULL;
    if (input->ndim != 5) {
        LOG_ERROR("BatchNorm3d expects 5D input [N, C, D, H, W], got %dD", input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int depth    = input->shape[2];
    int height   = input->shape[3];
    int width    = input->shape[4];

    if (channels != bn->num_features) {
        LOG_ERROR("BatchNorm3d: input channels (%d) doesn't match num_features (%d)", channels,
                  bn->num_features);
        return NULL;
    }

    int spatial            = depth * height * width;
    int flat_spatial_batch = batch * spatial;

    bool training = module_is_training(module);

    Tensor* mean_tensor = NULL;
    Tensor* var_tensor  = NULL;
    if (training) {
        if (!bn->current_mean) {
            int stat_shape[]    = {channels};
            TensorConfig config = (TensorConfig){.dtype      = input->dtype,
                                                 .device     = input->device,
                                                 .has_dtype  = true,
                                                 .has_device = true};
            bn->current_mean    = tensor_zeros(stat_shape, 1, &config);
            bn->current_var     = tensor_zeros(stat_shape, 1, &config);
            if (!bn->current_mean || !bn->current_var) {
                return NULL;
            }
        }
        ReshapeParams reshape_params;
        int reshaped_shape[]     = {channels, flat_spatial_batch};
        reshape_params.new_shape = reshaped_shape;
        reshape_params.new_ndim  = 2;

        Tensor* input_reshaped = uop_reshape(input, &reshape_params);
        if (!input_reshaped) {
            return NULL;
        }
        ReduceParams mean_params;
        int mean_dims[]      = {1};
        mean_params.dims     = mean_dims;
        mean_params.num_dims = 1;
        mean_params.keepdim  = false;

        Tensor* mean_reduced = uop_mean(input_reshaped, &mean_params);
        if (!mean_reduced)
            return NULL;

        float* mean_reduced_data = (float*)tensor_data_ptr(mean_reduced);
        float* current_mean_data = (float*)tensor_data_ptr(bn->current_mean);
        if (mean_reduced_data && current_mean_data)
            memcpy(current_mean_data, mean_reduced_data, (size_t)channels * sizeof(float));

        ReshapeParams reshape_mean_2d;
        int mean_2d_shape[]        = {channels, 1};
        reshape_mean_2d.new_shape  = mean_2d_shape;
        reshape_mean_2d.new_ndim   = 2;

        Tensor* mean_2d = uop_reshape(bn->current_mean, &reshape_mean_2d);
        if (!mean_2d)
            return NULL;

        ExpandParams expand_params;
        int reshaped_shape_expand[] = {channels, flat_spatial_batch};
        expand_params.new_shape     = reshaped_shape_expand;
        expand_params.new_ndim      = 2;

        Tensor* mean_broadcast = uop_expand(mean_2d, &expand_params);
        if (!mean_broadcast)
            return NULL;

        Tensor* diff = uop_sub(input_reshaped, mean_broadcast);
        if (!diff)
            return NULL;

        Tensor* diff_sq = uop_mul(diff, diff);
        if (!diff_sq)
            return NULL;

        ReduceParams var_params;
        int var_dims[]      = {1};
        var_params.dims     = var_dims;
        var_params.num_dims = 1;
        var_params.keepdim  = false;

        Tensor* var_reduced = uop_mean(diff_sq, &var_params);
        if (!var_reduced)
            return NULL;

        float* var_reduced_data = (float*)tensor_data_ptr(var_reduced);
        float* current_var_data = (float*)tensor_data_ptr(bn->current_var);
        if (var_reduced_data && current_var_data)
            memcpy(current_var_data, var_reduced_data, (size_t)channels * sizeof(float));
        if (bn->track_running_stats && bn->running_mean && bn->running_var) {
            float* running_mean = (float*)tensor_data_ptr(bn->running_mean);
            float* running_var  = (float*)tensor_data_ptr(bn->running_var);

            if (running_mean && running_var && current_mean_data && current_var_data) {
                for (int c = 0; c < channels; c++) {
                    running_mean[c] = bn->momentum * running_mean[c] +
                                      (1.0f - bn->momentum) * current_mean_data[c];
                    running_var[c] =
                        bn->momentum * running_var[c] + (1.0f - bn->momentum) * current_var_data[c];
                }
            }
        }

        mean_tensor = bn->current_mean;
        var_tensor  = bn->current_var;
    } else {
        mean_tensor = bn->running_mean;
        var_tensor  = bn->running_var;
    }

    if (!mean_tensor || !var_tensor) {
        LOG_ERROR("BatchNorm3d: missing mean or variance tensor");
        return NULL;
    }

    ReshapeParams reshape_1d_to_5d;
    int stat_5d_shape[]      = {1, channels, 1, 1, 1};
    reshape_1d_to_5d.new_shape = stat_5d_shape;
    reshape_1d_to_5d.new_ndim  = 5;

    Tensor* mean_5d = uop_reshape(mean_tensor, &reshape_1d_to_5d);
    if (!mean_5d)
        return NULL;

    ExpandParams expand_mean;
    int input_shape[]     = {batch, channels, depth, height, width};
    expand_mean.new_shape = input_shape;
    expand_mean.new_ndim  = 5;

    Tensor* mean_broadcast = uop_expand(mean_5d, &expand_mean);
    if (!mean_broadcast)
        return NULL;

    Tensor* centered = uop_sub(input, mean_broadcast);
    if (!centered)
        return NULL;

    int var_shape[]     = {channels};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* eps_tensor = tensor_full(var_shape, 1, &config, bn->eps);
    if (!eps_tensor)
        return NULL;

    Tensor* var_eps = uop_add(var_tensor, eps_tensor);
    tensor_free(eps_tensor);
    if (!var_eps)
        return NULL;

    Tensor* std_tensor = uop_sqrt(var_eps);
    if (!std_tensor)
        return NULL;

    Tensor* std_5d = uop_reshape(std_tensor, &reshape_1d_to_5d);
    if (!std_5d)
        return NULL;

    ExpandParams expand_std;
    expand_std.new_shape  = input_shape;
    expand_std.new_ndim   = 5;
    Tensor* std_broadcast = uop_expand(std_5d, &expand_std);
    if (!std_broadcast)
        return NULL;

    Tensor* normalized = uop_div(centered, std_broadcast);
    if (!normalized)
        return NULL;

    Tensor* output = normalized;

    if (bn->affine && bn->weight && bn->bias) {
        Tensor* weight_5d = uop_reshape(bn->weight->tensor, &reshape_1d_to_5d);
        if (!weight_5d)
            return NULL;

        ExpandParams expand_weight;
        expand_weight.new_shape = input_shape;
        expand_weight.new_ndim  = 5;
        Tensor* weight_broadcast = uop_expand(weight_5d, &expand_weight);
        if (!weight_broadcast)
            return NULL;

        Tensor* scaled = uop_mul(weight_broadcast, output);
        if (!scaled)
            return NULL;

        Tensor* bias_5d = uop_reshape(bn->bias->tensor, &reshape_1d_to_5d);
        if (!bias_5d)
            return NULL;

        ExpandParams expand_bias;
        expand_bias.new_shape = input_shape;
        expand_bias.new_ndim  = 5;
        Tensor* bias_broadcast = uop_expand(bias_5d, &expand_bias);
        if (!bias_broadcast)
            return NULL;

        output = uop_add(scaled, bias_broadcast);
        if (!output)
            return NULL;
    }

    return output;
}

static void batchnorm3d_free(Module* module) {
    BatchNorm3d* bn = (BatchNorm3d*)module;
    if (!bn)
        return;

    if (bn->running_mean)
        tensor_free(bn->running_mean);
    if (bn->running_var)
        tensor_free(bn->running_var);
    if (bn->current_mean)
        tensor_free(bn->current_mean);
    if (bn->current_var)
        tensor_free(bn->current_var);

    free(bn);
}

BatchNorm3d* nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device) {
    BatchNorm3d* bn = malloc(sizeof(BatchNorm3d));
    if (!bn)
        return NULL;

    if (module_init((Module*)bn, "BatchNorm3d", batchnorm3d_forward, batchnorm3d_free) != 0) {
        free(bn);
        return NULL;
    }

    bn->num_features        = num_features;
    bn->eps                 = eps > 0.0f ? eps : 1e-5f;
    bn->momentum            = momentum > 0.0f ? momentum : 0.1f;
    bn->affine              = affine;
    bn->track_running_stats = track_running_stats;

    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    if (affine) {
        int param_shape[] = {num_features};

        Tensor* weight = tensor_ones(param_shape, 1, &config);
        if (!weight) {
            module_free((Module*)bn);
            return NULL;
        }
        if (module_add_parameter((Module*)bn, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)bn);
            return NULL;
        }
        bn->weight = module_get_parameter((Module*)bn, "weight");

        Tensor* bias = tensor_zeros(param_shape, 1, &config);
        if (!bias) {
            module_free((Module*)bn);
            return NULL;
        }
        if (module_add_parameter((Module*)bn, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)bn);
            return NULL;
        }
        bn->bias = module_get_parameter((Module*)bn, "bias");
    } else {
        bn->weight = NULL;
        bn->bias   = NULL;
    }

    if (track_running_stats) {
        int stat_shape[] = {num_features};
        bn->running_mean = tensor_zeros(stat_shape, 1, &config);
        bn->running_var  = tensor_ones(stat_shape, 1, &config);
    } else {
        bn->running_mean = NULL;
        bn->running_var  = NULL;
    }

    bn->current_mean = NULL;
    bn->current_var  = NULL;

    return bn;
}
