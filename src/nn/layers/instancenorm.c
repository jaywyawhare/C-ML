/**
 * @file instancenorm.c
 * @brief Instance Normalization 2D layer implementation
 *
 * InstanceNorm normalizes each channel of each sample independently:
 * y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
 * where mean/var are computed per (batch, channel) pair over (H, W).
 */

#include "nn/layers/instancenorm.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* instancenorm2d_forward(Module* module, Tensor* input) {
    InstanceNorm2d* in = (InstanceNorm2d*)module;

    if (!in || !input)
        return NULL;

    // Input shape: [batch, channels, height, width]
    if (input->ndim != 4) {
        LOG_ERROR("InstanceNorm2d expects 4D input [N, C, H, W], got %dD", input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    if (channels != in->num_features) {
        LOG_ERROR("InstanceNorm2d: input channels (%d) doesn't match num_features (%d)",
                  channels, in->num_features);
        return NULL;
    }

    // Ensure input is executed
    tensor_ensure_executed(input);
    float* input_data = (float*)tensor_data_ptr(input);
    if (!input_data)
        return NULL;

    int spatial = height * width;

    // Create output tensor
    TensorConfig config = (TensorConfig){.dtype      = input->dtype,
                                         .device     = input->device,
                                         .has_dtype  = true,
                                         .has_device = true};
    Tensor* output = tensor_empty(input->shape, input->ndim, &config);
    if (!output)
        return NULL;

    tensor_ensure_executed(output);
    float* output_data = (float*)tensor_data_ptr(output);

    float* weight_data = NULL;
    float* bias_data   = NULL;

    if (in->affine && in->weight && in->weight->tensor) {
        tensor_ensure_executed(in->weight->tensor);
        weight_data = (float*)tensor_data_ptr(in->weight->tensor);
    }
    if (in->affine && in->bias && in->bias->tensor) {
        tensor_ensure_executed(in->bias->tensor);
        bias_data = (float*)tensor_data_ptr(in->bias->tensor);
    }

    // Normalize each (batch, channel) pair independently
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            int offset = (n * channels + c) * spatial;

            // Compute mean
            float mean = 0.0f;
            for (int s = 0; s < spatial; s++) {
                mean += input_data[offset + s];
            }
            mean /= (float)spatial;

            // Compute variance
            float var = 0.0f;
            for (int s = 0; s < spatial; s++) {
                float diff = input_data[offset + s] - mean;
                var += diff * diff;
            }
            var /= (float)spatial;

            float inv_std = 1.0f / sqrtf(var + in->eps);

            // Normalize and apply affine if needed
            float w = weight_data ? weight_data[c] : 1.0f;
            float b = bias_data ? bias_data[c] : 0.0f;

            for (int s = 0; s < spatial; s++) {
                output_data[offset + s] = (input_data[offset + s] - mean) * inv_std * w + b;
            }
        }
    }

    return output;
}

static void instancenorm2d_free(Module* module) {
    InstanceNorm2d* in = (InstanceNorm2d*)module;
    if (!in)
        return;

    // Parameters are freed by module system
    free(in);
}

InstanceNorm2d* nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                   DeviceType device) {
    InstanceNorm2d* in = malloc(sizeof(InstanceNorm2d));
    if (!in)
        return NULL;

    if (module_init((Module*)in, "InstanceNorm2d", instancenorm2d_forward, instancenorm2d_free) != 0) {
        free(in);
        return NULL;
    }

    in->num_features = num_features;
    in->eps          = eps > 0.0f ? eps : 1e-5f;
    in->affine       = affine;
    in->weight       = NULL;
    in->bias         = NULL;

    if (affine) {
        int param_shape[] = {num_features};
        TensorConfig config = (TensorConfig){.dtype      = dtype,
                                             .device     = device,
                                             .has_dtype  = true,
                                             .has_device = true};

        Tensor* weight = tensor_ones(param_shape, 1, &config);
        if (!weight) {
            module_free((Module*)in);
            return NULL;
        }

        if (module_add_parameter((Module*)in, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)in);
            return NULL;
        }

        in->weight = module_get_parameter((Module*)in, "weight");

        Tensor* bias = tensor_zeros(param_shape, 1, &config);
        if (!bias) {
            module_free((Module*)in);
            return NULL;
        }

        if (module_add_parameter((Module*)in, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)in);
            return NULL;
        }

        in->bias = module_get_parameter((Module*)in, "bias");
    }

    return in;
}
