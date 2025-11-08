/**
 * @file conv2d.c
 * @brief 2D Convolution layer implementation
 *
 * Note: This is a simplified implementation. Full convolution requires
 * im2col transformation or direct convolution loops.
 */

#include "nn/layers/conv2d.h"
#include "nn/module.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 2D Convolution forward pass
static Tensor* conv2d_forward(Module* module, Tensor* input) {
    Conv2d* conv2d = (Conv2d*)module;

    if (!conv2d || !input)
        return NULL;

    // Input shape: [batch, in_channels, height, width]
    if (input->ndim != 4) {
        LOG_ERROR("Conv2d expects 4D input [batch, in_channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    Parameter* weight_param = conv2d->weight;
    Parameter* bias_param   = conv2d->bias;

    if (!weight_param || !weight_param->tensor) {
        LOG_ERROR("Conv2d missing weight parameter");
        return NULL;
    }

    Tensor* weight = weight_param->tensor;

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_height   = input->shape[2];
    int in_width    = input->shape[3];

    int out_channels       = weight->shape[0];
    int weight_in_channels = weight->shape[1];

    // Validate input channels match weight channels
    if (in_channels != weight_in_channels) {
        LOG_ERROR("Conv2d: input channels (%d) doesn't match weight in_channels (%d)", in_channels,
                  weight_in_channels);
        return NULL;
    }

    // Validate weight channels match layer configuration
    if (in_channels != conv2d->in_channels || out_channels != conv2d->out_channels) {
        LOG_ERROR("Conv2d: weight dimensions don't match layer configuration");
        return NULL;
    }
    int kernel_h   = conv2d->kernel_size[0];
    int kernel_w   = conv2d->kernel_size[1];
    int stride_h   = conv2d->stride[0];
    int stride_w   = conv2d->stride[1];
    int padding_h  = conv2d->padding[0];
    int padding_w  = conv2d->padding[1];
    int dilation_h = conv2d->dilation[0];
    int dilation_w = conv2d->dilation[1];

    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create output tensor [batch, out_channels, out_h, out_w]
    int output_shape[] = {batch, out_channels, out_height, out_width};
    Tensor* output     = tensor_zeros(output_shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* input_data  = (float*)tensor_data_ptr(input);
    float* weight_data = (float*)tensor_data_ptr(weight);
    float* output_data = (float*)tensor_data_ptr(output);
    float* bias_data   = NULL;

    if (!input_data || !weight_data || !output_data) {
        tensor_free(output);
        return NULL;
    }

    if (conv2d->use_bias && bias_param && bias_param->tensor) {
        bias_data = (float*)tensor_data_ptr(bias_param->tensor);
    }

    // Perform 2D convolution
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;

                    // Convolution operation
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                // Calculate input position with padding and dilation
                                int ih = oh * stride_h - padding_h + kh * dilation_h;
                                int iw = ow * stride_w - padding_w + kw * dilation_w;

                                // Skip if out of bounds (padding)
                                if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
                                    continue;
                                }

                                // Input indices: [b, ic, ih, iw]
                                int input_indices[] = {b, ic, ih, iw};
                                size_t input_offset = tensor_compute_offset(input, input_indices);

                                // Weight indices: [oc, ic, kh, kw]
                                int weight_indices[] = {oc, ic, kh, kw};
                                size_t weight_offset =
                                    tensor_compute_offset(weight, weight_indices);

                                sum += input_data[input_offset] * weight_data[weight_offset];
                            }
                        }
                    }

                    // Add bias if present
                    if (bias_data) {
                        sum += bias_data[oc];
                    }

                    // Output indices: [b, oc, oh, ow]
                    int output_indices[]       = {b, oc, oh, ow};
                    size_t output_offset       = tensor_compute_offset(output, output_indices);
                    output_data[output_offset] = sum;
                }
            }
        }
    }

    return output;
}

static void conv2d_free(Module* module) {
    Conv2d* conv2d = (Conv2d*)module;
    if (!conv2d)
        return;

    CM_FREE(conv2d);
}

static void kaiming_init(Tensor* tensor, int in_channels, int out_channels, int kernel_size) {
    (void)out_channels;
    if (!tensor || !tensor->data)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    // He initialization: std = sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    float scale  = sqrtf(2.0f / (in_channels * kernel_size * kernel_size));
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

Conv2d* nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device) {
    Conv2d* conv2d = CM_MALLOC(sizeof(Conv2d));
    if (!conv2d)
        return NULL;

    if (module_init((Module*)conv2d, "Conv2d", conv2d_forward, conv2d_free) != 0) {
        CM_FREE(conv2d);
        return NULL;
    }

    conv2d->in_channels    = in_channels;
    conv2d->out_channels   = out_channels;
    conv2d->kernel_size[0] = kernel_size;
    conv2d->kernel_size[1] = kernel_size;
    conv2d->stride[0]      = stride;
    conv2d->stride[1]      = stride;
    conv2d->padding[0]     = padding;
    conv2d->padding[1]     = padding;
    conv2d->dilation[0]    = dilation;
    conv2d->dilation[1]    = dilation;
    conv2d->use_bias       = use_bias;

    // Create weight tensor [out_channels, in_channels, kernel_h, kernel_w]
    int weight_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    Tensor* weight     = tensor_empty(weight_shape, 4, dtype, device);
    if (!weight) {
        module_free((Module*)conv2d);
        return NULL;
    }

    // Initialize with Kaiming/He initialization
    kaiming_init(weight, in_channels, out_channels, kernel_size);

    if (module_add_parameter((Module*)conv2d, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)conv2d);
        return NULL;
    }

    conv2d->weight = module_get_parameter((Module*)conv2d, "weight");

    // Create bias if needed
    if (use_bias) {
        int bias_shape[] = {out_channels};
        Tensor* bias     = tensor_zeros(bias_shape, 1, dtype, device);
        if (!bias) {
            module_free((Module*)conv2d);
            return NULL;
        }

        if (module_add_parameter((Module*)conv2d, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)conv2d);
            return NULL;
        }

        conv2d->bias = module_get_parameter((Module*)conv2d, "bias");
    } else {
        conv2d->bias = NULL;
    }

    LOG_DEBUG("Created Conv2d layer: %d -> %d, kernel=%d, stride=%d, padding=%d", in_channels,
              out_channels, kernel_size, stride, padding);

    return conv2d;
}
