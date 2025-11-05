/**
 * @file batchnorm2d.c
 * @brief Batch Normalization 2D layer implementation
 */

#include "nn/layers/batchnorm2d.h"
#include "nn/module.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* batchnorm2d_forward(Module* module, Tensor* input) {
    BatchNorm2d* bn = (BatchNorm2d*)module;

    if (!bn || !input)
        return NULL;

    // Input shape: [batch, channels, height, width]
    if (input->ndim != 4) {
        LOG_ERROR("BatchNorm2d expects 4D input [batch, channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    // Validate channels match num_features
    if (channels != bn->num_features) {
        LOG_ERROR("BatchNorm2d: input channels (%d) doesn't match num_features (%d)", channels,
                  bn->num_features);
        return NULL;
    }

    // Allocate output tensor
    int output_shape[] = {batch, channels, height, width};
    Tensor* output     = tensor_empty(output_shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* input_data  = (float*)tensor_data_ptr(input);
    float* output_data = (float*)tensor_data_ptr(output);

    if (!input_data || !output_data) {
        tensor_free(output);
        return NULL;
    }

    bool training = module_is_training(module);

    // Compute mean and variance per channel
    if (training) {
        // Allocate current statistics if not already allocated
        if (!bn->current_mean) {
            int stat_shape[] = {channels};
            bn->current_mean = tensor_zeros(stat_shape, 1, input->dtype, input->device);
            bn->current_var  = tensor_zeros(stat_shape, 1, input->dtype, input->device);
            if (!bn->current_mean || !bn->current_var) {
                tensor_free(output);
                return NULL;
            }
        }

        float* mean_data = (float*)tensor_data_ptr(bn->current_mean);
        float* var_data  = (float*)tensor_data_ptr(bn->current_var);

        if (!mean_data || !var_data) {
            tensor_free(output);
            return NULL;
        }

        // Compute mean per channel: mean over [batch, height, width]
        size_t spatial_size = batch * height * width;
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int indices[] = {b, c, h, w};
                        size_t offset = tensor_compute_offset(input, indices);
                        sum += input_data[offset];
                    }
                }
            }
            mean_data[c] = sum / spatial_size;
        }

        // Compute variance per channel
        for (int c = 0; c < channels; c++) {
            float sum_sq = 0.0f;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int indices[] = {b, c, h, w};
                        size_t offset = tensor_compute_offset(input, indices);
                        float diff    = input_data[offset] - mean_data[c];
                        sum_sq += diff * diff;
                    }
                }
            }
            var_data[c] = sum_sq / spatial_size;
        }

        // Update running statistics
        if (bn->track_running_stats && bn->running_mean && bn->running_var) {
            float* running_mean = (float*)tensor_data_ptr(bn->running_mean);
            float* running_var  = (float*)tensor_data_ptr(bn->running_var);

            if (running_mean && running_var) {
                for (int c = 0; c < channels; c++) {
                    running_mean[c] =
                        bn->momentum * running_mean[c] + (1.0f - bn->momentum) * mean_data[c];
                    running_var[c] =
                        bn->momentum * running_var[c] + (1.0f - bn->momentum) * var_data[c];
                }
            }
        }
    }

    // Get mean and variance (use running stats in eval mode)
    float* mean_data = training ? (float*)tensor_data_ptr(bn->current_mean)
                                : (float*)tensor_data_ptr(bn->running_mean);
    float* var_data  = training ? (float*)tensor_data_ptr(bn->current_var)
                                : (float*)tensor_data_ptr(bn->running_var);

    if (!mean_data || !var_data) {
        tensor_free(output);
        return NULL;
    }

    // Get weight (gamma) and bias (beta) if affine
    float* weight_data = NULL;
    float* bias_data   = NULL;

    if (bn->affine && bn->weight && bn->bias) {
        weight_data = (float*)tensor_data_ptr(bn->weight->tensor);
        bias_data   = (float*)tensor_data_ptr(bn->bias->tensor);
    }

    // Normalize: (x - mean) / sqrt(var + eps)
    // Then scale and shift: gamma * normalized + beta
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float mean = mean_data[c];
            float var  = var_data[c];
            float std  = sqrtf(var + bn->eps);

            float gamma = bn->affine && weight_data ? weight_data[c] : 1.0f;
            float beta  = bn->affine && bias_data ? bias_data[c] : 0.0f;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int indices[] = {b, c, h, w};
                    size_t offset = tensor_compute_offset(input, indices);
                    float x       = input_data[offset];

                    // Normalize
                    float normalized = (x - mean) / std;

                    // Scale and shift
                    float result = gamma * normalized + beta;

                    size_t out_offset       = tensor_compute_offset(output, indices);
                    output_data[out_offset] = result;
                }
            }
        }
    }

    return output;
}

static void batchnorm2d_free(Module* module) {
    BatchNorm2d* bn = (BatchNorm2d*)module;
    if (!bn)
        return;

    // Free running statistics
    if (bn->running_mean)
        tensor_free(bn->running_mean);
    if (bn->running_var)
        tensor_free(bn->running_var);
    if (bn->current_mean)
        tensor_free(bn->current_mean);
    if (bn->current_var)
        tensor_free(bn->current_var);

    // Parameters are freed by module system
    CM_FREE(bn);
}

BatchNorm2d* nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device) {
    BatchNorm2d* bn = CM_MALLOC(sizeof(BatchNorm2d));
    if (!bn)
        return NULL;

    if (module_init((Module*)bn, "BatchNorm2d", batchnorm2d_forward, batchnorm2d_free) != 0) {
        CM_FREE(bn);
        return NULL;
    }

    bn->num_features        = num_features;
    bn->eps                 = eps;
    bn->momentum            = momentum;
    bn->affine              = affine;
    bn->track_running_stats = track_running_stats;

    // Create learnable parameters if affine
    if (affine) {
        int param_shape[] = {num_features};

        // Weight (gamma)
        Tensor* weight = tensor_ones(param_shape, 1, dtype, device);
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

        // Bias (beta)
        Tensor* bias = tensor_zeros(param_shape, 1, dtype, device);
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

    // Create running statistics if tracking
    if (track_running_stats) {
        int stat_shape[] = {num_features};
        bn->running_mean = tensor_zeros(stat_shape, 1, dtype, device);
        bn->running_var  = tensor_ones(stat_shape, 1, dtype, device);

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

    LOG_DEBUG("Created BatchNorm2d layer: num_features=%d, eps=%.6f, momentum=%.6f", num_features,
              eps, momentum);

    return bn;
}
