/**
 * @file groupnorm.c
 * @brief Group Normalization layer implementation
 *
 * Normalizes within groups of channels. For input [N, C, ...]:
 * - Divide C channels into num_groups groups, each with C/num_groups channels
 * - For each sample and group, compute mean and variance across the channels
 *   in that group and all spatial dimensions
 * - Normalize: (x - mean) / sqrt(var + eps)
 * - Apply affine transform if enabled: y = weight * x_norm + bias (per-channel)
 */

#include "nn/layers/groupnorm.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

static Tensor* groupnorm_forward(Module* module, Tensor* input) {
    GroupNorm* gn = (GroupNorm*)module;

    if (!gn || !input)
        return NULL;

    if (input->ndim < 2) {
        LOG_ERROR("GroupNorm expects at least 2D input, got %dD", input->ndim);
        return NULL;
    }

    int C = input->shape[1];
    if (C != gn->num_channels) {
        LOG_ERROR("GroupNorm: input channels (%d) doesn't match num_channels (%d)",
                  C, gn->num_channels);
        return NULL;
    }

    tensor_ensure_executed(input);

    int N = input->shape[0];
    int G = gn->num_groups;
    int channels_per_group = C / G;

    int spatial = 1;
    for (int i = 2; i < input->ndim; i++)
        spatial *= input->shape[i];

    Tensor* output = tensor_clone(input);
    if (!output)
        return NULL;

    tensor_ensure_executed(output);
    float* out_data = (float*)tensor_data_ptr(output);
    float* in_data  = (float*)tensor_data_ptr(input);

    if (!out_data || !in_data) {
        tensor_free(output);
        return NULL;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            int group_size = channels_per_group * spatial;

            float mean = 0.0f;
            for (int c = 0; c < channels_per_group; c++) {
                int ch = g * channels_per_group + c;
                for (int s = 0; s < spatial; s++) {
                    mean += in_data[n * C * spatial + ch * spatial + s];
                }
            }
            mean /= (float)group_size;

            float var = 0.0f;
            for (int c = 0; c < channels_per_group; c++) {
                int ch = g * channels_per_group + c;
                for (int s = 0; s < spatial; s++) {
                    float diff = in_data[n * C * spatial + ch * spatial + s] - mean;
                    var += diff * diff;
                }
            }
            var /= (float)group_size;

            float inv_std = 1.0f / sqrtf(var + gn->eps);

            for (int c = 0; c < channels_per_group; c++) {
                int ch = g * channels_per_group + c;
                for (int s = 0; s < spatial; s++) {
                    int idx = n * C * spatial + ch * spatial + s;
                    out_data[idx] = (in_data[idx] - mean) * inv_std;
                }
            }
        }
    }

    if (gn->affine && gn->weight && gn->bias) {
        tensor_ensure_executed(gn->weight->tensor);
        tensor_ensure_executed(gn->bias->tensor);
        float* w = (float*)tensor_data_ptr(gn->weight->tensor);
        float* b = (float*)tensor_data_ptr(gn->bias->tensor);

        if (!w || !b) {
            tensor_free(output);
            return NULL;
        }

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int s = 0; s < spatial; s++) {
                    int idx = n * C * spatial + c * spatial + s;
                    out_data[idx] = out_data[idx] * w[c] + b[c];
                }
            }
        }
    }

    return output;
}

static void groupnorm_free(Module* module) {
    free(module);
}

GroupNorm* nn_groupnorm(int num_groups, int num_channels, float eps, bool affine,
                        DType dtype, DeviceType device) {
    if (num_channels % num_groups != 0) {
        LOG_ERROR("num_channels (%d) must be divisible by num_groups (%d)",
                  num_channels, num_groups);
        return NULL;
    }

    GroupNorm* gn = malloc(sizeof(GroupNorm));
    if (!gn)
        return NULL;

    if (module_init((Module*)gn, "GroupNorm", groupnorm_forward, groupnorm_free) != 0) {
        free(gn);
        return NULL;
    }

    gn->num_groups   = num_groups;
    gn->num_channels = num_channels;
    gn->eps          = eps > 0 ? eps : 1e-5f;
    gn->affine       = affine;

    if (affine) {
        int param_shape[] = {num_channels};
        TensorConfig config = (TensorConfig){
            .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

        Tensor* weight = tensor_ones(param_shape, 1, &config);
        if (!weight) {
            module_free((Module*)gn);
            return NULL;
        }

        if (module_add_parameter((Module*)gn, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)gn);
            return NULL;
        }

        gn->weight = module_get_parameter((Module*)gn, "weight");

        Tensor* bias = tensor_zeros(param_shape, 1, &config);
        if (!bias) {
            module_free((Module*)gn);
            return NULL;
        }

        if (module_add_parameter((Module*)gn, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)gn);
            return NULL;
        }

        gn->bias = module_get_parameter((Module*)gn, "bias");
    } else {
        gn->weight = NULL;
        gn->bias   = NULL;
    }

    return gn;
}
