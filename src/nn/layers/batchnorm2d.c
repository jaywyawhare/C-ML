#include "nn/layers/batchnorm2d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* batchnorm2d_forward(Module* module, Tensor* input) {
    BatchNorm2d* bn = (BatchNorm2d*)module;

    if (!bn || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("BatchNorm2d expects 4D input [batch, channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    if (channels != bn->num_features) {
        LOG_ERROR("BatchNorm2d: input channels (%d) doesn't match num_features (%d)", channels,
                  bn->num_features);
        return NULL;
    }

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
        int reshaped_shape[]     = {channels, batch * height * width};
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
        if (!mean_reduced) {
            tensor_free(input_reshaped);
            return NULL;
        }
        float* mean_reduced_data = (float*)tensor_data_ptr(mean_reduced);
        float* current_mean_data = (float*)tensor_data_ptr(bn->current_mean);
        if (mean_reduced_data && current_mean_data) {
            memcpy(current_mean_data, mean_reduced_data, (size_t)channels * sizeof(float));
        }
        tensor_free(mean_reduced);
        ExpandParams expand_params;
        int reshaped_shape_expand[] = {channels, batch * height * width};
        expand_params.new_shape     = reshaped_shape_expand;
        expand_params.new_ndim      = 2;

        Tensor* mean_broadcast = uop_expand(bn->current_mean, &expand_params);
        if (!mean_broadcast) {
            tensor_free(input_reshaped);
            return NULL;
        }
        Tensor* diff = uop_sub(input_reshaped, mean_broadcast);
        tensor_free(input_reshaped);
        tensor_free(mean_broadcast);
        if (!diff) {
            return NULL;
        }
        Tensor* diff_sq = uop_mul(diff, diff);
        tensor_free(diff);
        if (!diff_sq) {
            return NULL;
        }
        ReduceParams var_params;
        int var_dims[]      = {1};
        var_params.dims     = var_dims;
        var_params.num_dims = 1;
        var_params.keepdim  = false;

        Tensor* var_reduced = uop_mean(diff_sq, &var_params);
        tensor_free(diff_sq);
        if (!var_reduced) {
            return NULL;
        }
        float* var_reduced_data = (float*)tensor_data_ptr(var_reduced);
        float* current_var_data = (float*)tensor_data_ptr(bn->current_var);
        if (var_reduced_data && current_var_data) {
            memcpy(current_var_data, var_reduced_data, (size_t)channels * sizeof(float));
        }
        tensor_free(var_reduced);
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
        LOG_ERROR("BatchNorm2d: missing mean or variance tensor");
        return NULL;
    }
    ExpandParams expand_mean;
    int input_shape[]     = {batch, channels, height, width};
    expand_mean.new_shape = input_shape;
    expand_mean.new_ndim  = 4;

    Tensor* mean_broadcast = uop_expand(mean_tensor, &expand_mean);
    if (!mean_broadcast) {
        return NULL;
    }
    Tensor* centered = uop_sub(input, mean_broadcast);
    tensor_free(mean_broadcast);
    if (!centered) {
        return NULL;
    }
    int var_shape[]     = {channels};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* eps_tensor = tensor_zeros(var_shape, 1, &config);
    if (!eps_tensor) {
        tensor_free(centered);
        return NULL;
    }
    float* eps_data = (float*)tensor_data_ptr(eps_tensor);
    if (eps_data) {
        for (int c = 0; c < channels; c++) {
            eps_data[c] = bn->eps;
        }
    }
    Tensor* var_eps = uop_add(var_tensor, eps_tensor);
    tensor_free(eps_tensor);
    if (!var_eps) {
        tensor_free(centered);
        return NULL;
    }
    Tensor* std_tensor = uop_sqrt(var_eps);
    tensor_free(var_eps);
    if (!std_tensor) {
        tensor_free(centered);
        return NULL;
    }
    ExpandParams expand_std;
    expand_std.new_shape  = input_shape;
    expand_std.new_ndim   = 4;
    Tensor* std_broadcast = uop_expand(std_tensor, &expand_std);
    tensor_free(std_tensor);
    if (!std_broadcast) {
        tensor_free(centered);
        return NULL;
    }
    Tensor* normalized = uop_div(centered, std_broadcast);
    tensor_free(centered);
    tensor_free(std_broadcast);
    if (!normalized) {
        return NULL;
    }
    Tensor* output = normalized;

    if (bn->affine && bn->weight && bn->bias) {
        ExpandParams expand_weight;
        expand_weight.new_shape = input_shape;
        expand_weight.new_ndim  = 4;

        Tensor* weight_broadcast = uop_expand(bn->weight->tensor, &expand_weight);
        if (!weight_broadcast) {
            tensor_free(output);
            return NULL;
        }
        Tensor* scaled = uop_mul(weight_broadcast, output);
        tensor_free(weight_broadcast);
        tensor_free(output);
        if (!scaled) {
            return NULL;
        }
        ExpandParams expand_bias;
        expand_bias.new_shape = input_shape;
        expand_bias.new_ndim  = 4;

        Tensor* bias_broadcast = uop_expand(bn->bias->tensor, &expand_bias);
        if (!bias_broadcast) {
            tensor_free(scaled);
            return NULL;
        }
        output = uop_add(scaled, bias_broadcast);
        tensor_free(scaled);
        tensor_free(bias_broadcast);
        if (!output) {
            return NULL;
        }
    }

    return output;
}

static void batchnorm2d_free(Module* module) {
    BatchNorm2d* bn = (BatchNorm2d*)module;
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

BatchNorm2d* nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device) {
    BatchNorm2d* bn = malloc(sizeof(BatchNorm2d));
    if (!bn)
        return NULL;

    if (module_init((Module*)bn, "BatchNorm2d", batchnorm2d_forward, batchnorm2d_free) != 0) {
        free(bn);
        return NULL;
    }

    bn->num_features        = num_features;
    bn->eps                 = eps;
    bn->momentum            = momentum;
    bn->affine              = affine;
    bn->track_running_stats = track_running_stats;
    if (affine) {
        int param_shape[] = {num_features};
        TensorConfig config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
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
        TensorConfig config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        bn->running_mean = tensor_zeros(stat_shape, 1, &config);
        bn->running_var  = tensor_ones(stat_shape, 1, &config);

        if (!bn->running_mean || !bn->running_var) {
            if (bn->running_mean)
                tensor_free(bn->running_mean);
            if (bn->running_var)
                tensor_free(bn->running_var);
            module_free((Module*)bn);
            return NULL;
        }
    } else {
        bn->running_mean = NULL;
        bn->running_var  = NULL;
    }

    bn->current_mean = NULL;
    bn->current_var  = NULL;

    return bn;
}
