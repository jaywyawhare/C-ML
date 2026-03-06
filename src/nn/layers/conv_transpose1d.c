/**
 * @file conv_transpose1d.c
 * @brief 1D Transposed Convolution layer implementation
 *
 * Output shape: L_out = (L_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
 */

#include "nn/layers/conv_transpose1d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* conv_transpose1d_forward(Module* module, Tensor* input) {
    ConvTranspose1d* layer = (ConvTranspose1d*)module;

    if (!layer || !input)
        return NULL;

    if (input->ndim != 3) {
        LOG_ERROR("ConvTranspose1d expects 3D input [batch, in_channels, length], got %dD",
                  input->ndim);
        return NULL;
    }

    if (!layer->weight || !layer->weight->tensor) {
        LOG_ERROR("ConvTranspose1d missing weight parameter");
        return NULL;
    }

    tensor_ensure_executed(input);
    tensor_ensure_executed(layer->weight->tensor);

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_length   = input->shape[2];

    if (in_channels != layer->in_channels) {
        LOG_ERROR("ConvTranspose1d: input channels (%d) doesn't match expected (%d)",
                  in_channels, layer->in_channels);
        return NULL;
    }

    int out_channels = layer->out_channels;
    int ks           = layer->kernel_size;
    int s            = layer->stride;
    int p            = layer->padding;
    int opad         = layer->output_padding;
    int d            = layer->dilation;

    int out_length = (in_length - 1) * s - 2 * p + d * (ks - 1) + opad + 1;

    if (out_length <= 0) {
        LOG_ERROR("ConvTranspose1d: invalid output length (%d)", out_length);
        return NULL;
    }

    int out_shape[] = {batch, out_channels, out_length};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 3, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* w_data   = (float*)layer->weight->tensor->data;
    float* out_data = (float*)output->data;

    if (!in_data || !w_data || !out_data) {
        tensor_free(output);
        return NULL;
    }

    // Weight shape: [in_channels, out_channels, kernel_size]
    // Transposed conv: scatter each input element to output
    for (int b = 0; b < batch; b++) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int il = 0; il < in_length; il++) {
                float in_val = in_data[(b * in_channels + ic) * in_length + il];

                for (int oc = 0; oc < out_channels; oc++) {
                    for (int k = 0; k < ks; k++) {
                        int ol = il * s - p + k * d;

                        if (ol >= 0 && ol < out_length) {
                            float w_val = w_data[(ic * out_channels + oc) * ks + k];
                            out_data[(b * out_channels + oc) * out_length + ol] += in_val * w_val;
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if (layer->use_bias && layer->bias && layer->bias->tensor) {
        tensor_ensure_executed(layer->bias->tensor);
        float* bias_data = (float*)layer->bias->tensor->data;
        if (bias_data) {
            for (int b = 0; b < batch; b++) {
                for (int oc = 0; oc < out_channels; oc++) {
                    float bv = bias_data[oc];
                    for (int ol = 0; ol < out_length; ol++) {
                        out_data[(b * out_channels + oc) * out_length + ol] += bv;
                    }
                }
            }
        }
    }

    return output;
}

static void conv_transpose1d_free(Module* module) {
    ConvTranspose1d* layer = (ConvTranspose1d*)module;
    if (!layer)
        return;
    free(layer);
}

ConvTranspose1d* nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device) {
    ConvTranspose1d* layer = malloc(sizeof(ConvTranspose1d));
    if (!layer)
        return NULL;

    if (module_init((Module*)layer, "ConvTranspose1d", conv_transpose1d_forward,
                    conv_transpose1d_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->in_channels    = in_channels;
    layer->out_channels   = out_channels;
    layer->kernel_size    = kernel_size;
    layer->stride         = stride;
    layer->padding        = padding;
    layer->output_padding = output_padding;
    layer->dilation       = 1;
    layer->use_bias       = use_bias;

    // Weight: [in_channels, out_channels, kernel_size]
    int weight_shape[] = {in_channels, out_channels, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 3, &config);
    if (!weight) {
        module_free((Module*)layer);
        return NULL;
    }

    // Kaiming initialization
    float scale = sqrtf(2.0f / (float)(in_channels * kernel_size));
    float* data = (float*)tensor_data_ptr(weight);
    if (data) {
        for (size_t i = 0; i < weight->numel; i++)
            data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }

    if (module_add_parameter((Module*)layer, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)layer);
        return NULL;
    }
    layer->weight = module_get_parameter((Module*)layer, "weight");

    if (use_bias) {
        int bias_shape[] = {out_channels};
        Tensor* bias = tensor_zeros(bias_shape, 1, &config);
        if (!bias) {
            module_free((Module*)layer);
            return NULL;
        }
        if (module_add_parameter((Module*)layer, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)layer);
            return NULL;
        }
        layer->bias = module_get_parameter((Module*)layer, "bias");
    } else {
        layer->bias = NULL;
    }

    return layer;
}
