#include "nn/layers/layernorm2d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* layernorm2d_forward(Module* module, Tensor* input) {
    LayerNorm2d* ln = (LayerNorm2d*)module;

    if (!ln || !input)
        return NULL;

    if (input->ndim != 4) {
        LOG_ERROR("LayerNorm2d expects 4D input [N, C, H, W], got %dD", input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    if (channels != ln->num_channels) {
        LOG_ERROR("LayerNorm2d: input channels (%d) doesn't match num_channels (%d)",
                  channels, ln->num_channels);
        return NULL;
    }

    tensor_ensure_executed(input);
    float* input_data = (float*)tensor_data_ptr(input);
    if (!input_data)
        return NULL;

    int spatial = channels * height * width;

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, input->ndim, &config);
    if (!output)
        return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)tensor_data_ptr(output);

    float* weight_data = NULL;
    float* bias_data   = NULL;
    if (ln->affine && ln->weight && ln->weight->tensor) {
        tensor_ensure_executed(ln->weight->tensor);
        weight_data = (float*)tensor_data_ptr(ln->weight->tensor);
    }
    if (ln->affine && ln->bias && ln->bias->tensor) {
        tensor_ensure_executed(ln->bias->tensor);
        bias_data = (float*)tensor_data_ptr(ln->bias->tensor);
    }
    for (int n = 0; n < batch; n++) {
        int offset = n * spatial;
        float mean = 0.0f;
        for (int i = 0; i < spatial; i++)
            mean += input_data[offset + i];
        mean /= (float)spatial;
        float var = 0.0f;
        for (int i = 0; i < spatial; i++) {
            float diff = input_data[offset + i] - mean;
            var += diff * diff;
        }
        var /= (float)spatial;

        float inv_std = 1.0f / sqrtf(var + ln->eps);
        for (int c = 0; c < channels; c++) {
            float w = weight_data ? weight_data[c] : 1.0f;
            float b = bias_data ? bias_data[c] : 0.0f;
            int ch_offset = offset + c * height * width;

            for (int hw = 0; hw < height * width; hw++) {
                out_data[ch_offset + hw] =
                    (input_data[ch_offset + hw] - mean) * inv_std * w + b;
            }
        }
    }

    return output;
}

static void layernorm2d_free(Module* module) {
    LayerNorm2d* ln = (LayerNorm2d*)module;
    if (!ln)
        return;
    free(ln);
}

LayerNorm2d* nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                             DeviceType device) {
    LayerNorm2d* ln = malloc(sizeof(LayerNorm2d));
    if (!ln)
        return NULL;

    if (module_init((Module*)ln, "LayerNorm2d", layernorm2d_forward, layernorm2d_free) != 0) {
        free(ln);
        return NULL;
    }

    ln->num_channels = num_channels;
    ln->eps          = eps > 0.0f ? eps : 1e-5f;
    ln->affine       = affine;
    ln->weight       = NULL;
    ln->bias         = NULL;

    if (affine) {
        int param_shape[] = {num_channels};
        TensorConfig config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

        Tensor* weight = tensor_ones(param_shape, 1, &config);
        if (!weight) {
            module_free((Module*)ln);
            return NULL;
        }
        if (module_add_parameter((Module*)ln, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)ln);
            return NULL;
        }
        ln->weight = module_get_parameter((Module*)ln, "weight");

        Tensor* bias = tensor_zeros(param_shape, 1, &config);
        if (!bias) {
            module_free((Module*)ln);
            return NULL;
        }
        if (module_add_parameter((Module*)ln, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)ln);
            return NULL;
        }
        ln->bias = module_get_parameter((Module*)ln, "bias");
    }

    return ln;
}
