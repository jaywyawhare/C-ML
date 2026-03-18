#include "nn/layers/batchnorm3d.h"
#include "nn.h"
#include "tensor/tensor.h"
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
        LOG_ERROR("BatchNorm3d: input channels (%d) doesn't match num_features (%d)",
                  channels, bn->num_features);
        return NULL;
    }

    tensor_ensure_executed(input);
    float* input_data = (float*)tensor_data_ptr(input);
    if (!input_data)
        return NULL;

    bool training = module_is_training(module);
    int spatial = depth * height * width;
    int total_per_channel = batch * spatial;
    float* channel_mean = calloc((size_t)channels, sizeof(float));
    float* channel_var  = calloc((size_t)channels, sizeof(float));
    if (!channel_mean || !channel_var) {
        free(channel_mean);
        free(channel_var);
        return NULL;
    }

    if (training) {
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                int offset = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++) {
                    channel_mean[c] += input_data[offset + s];
                }
            }
        }
        for (int c = 0; c < channels; c++)
            channel_mean[c] /= (float)total_per_channel;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                int offset = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++) {
                    float diff = input_data[offset + s] - channel_mean[c];
                    channel_var[c] += diff * diff;
                }
            }
        }
        for (int c = 0; c < channels; c++)
            channel_var[c] /= (float)total_per_channel;
        if (bn->track_running_stats && bn->running_mean && bn->running_var) {
            float* rm = (float*)tensor_data_ptr(bn->running_mean);
            float* rv = (float*)tensor_data_ptr(bn->running_var);
            if (rm && rv) {
                for (int c = 0; c < channels; c++) {
                    rm[c] = bn->momentum * rm[c] + (1.0f - bn->momentum) * channel_mean[c];
                    rv[c] = bn->momentum * rv[c] + (1.0f - bn->momentum) * channel_var[c];
                }
            }
        }
    } else {
        if (bn->running_mean && bn->running_var) {
            float* rm = (float*)tensor_data_ptr(bn->running_mean);
            float* rv = (float*)tensor_data_ptr(bn->running_var);
            if (rm && rv) {
                memcpy(channel_mean, rm, (size_t)channels * sizeof(float));
                memcpy(channel_var, rv, (size_t)channels * sizeof(float));
            }
        }
    }
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, input->ndim, &config);
    if (!output) {
        free(channel_mean);
        free(channel_var);
        return NULL;
    }
    tensor_ensure_executed(output);
    float* out_data = (float*)tensor_data_ptr(output);

    float* weight_data = NULL;
    float* bias_data   = NULL;
    if (bn->affine && bn->weight && bn->weight->tensor) {
        tensor_ensure_executed(bn->weight->tensor);
        weight_data = (float*)tensor_data_ptr(bn->weight->tensor);
    }
    if (bn->affine && bn->bias && bn->bias->tensor) {
        tensor_ensure_executed(bn->bias->tensor);
        bias_data = (float*)tensor_data_ptr(bn->bias->tensor);
    }

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float inv_std = 1.0f / sqrtf(channel_var[c] + bn->eps);
            float w = weight_data ? weight_data[c] : 1.0f;
            float bi = bias_data ? bias_data[c] : 0.0f;
            int offset = (b * channels + c) * spatial;

            for (int s = 0; s < spatial; s++) {
                out_data[offset + s] =
                    (input_data[offset + s] - channel_mean[c]) * inv_std * w + bi;
            }
        }
    }

    free(channel_mean);
    free(channel_var);

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

    return bn;
}
