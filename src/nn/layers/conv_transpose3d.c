#include "nn/layers/conv_transpose3d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor* conv_transpose3d_forward(Module* module, Tensor* input) {
    ConvTranspose3d* layer = (ConvTranspose3d*)module;

    if (!layer || !input)
        return NULL;

    if (input->ndim != 5) {
        LOG_ERROR("ConvTranspose3d expects 5D input [batch, in_channels, depth, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    if (!layer->weight || !layer->weight->tensor) {
        LOG_ERROR("ConvTranspose3d missing weight parameter");
        return NULL;
    }

    tensor_ensure_executed(input);
    tensor_ensure_executed(layer->weight->tensor);

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_depth    = input->shape[2];
    int in_height   = input->shape[3];
    int in_width    = input->shape[4];

    if (in_channels != layer->in_channels) {
        LOG_ERROR("ConvTranspose3d: input channels (%d) doesn't match expected (%d)",
                  in_channels, layer->in_channels);
        return NULL;
    }

    int out_channels = layer->out_channels;
    int kernel_d     = layer->kernel_size[0];
    int kernel_h     = layer->kernel_size[1];
    int kernel_w     = layer->kernel_size[2];
    int stride_d     = layer->stride[0];
    int stride_h     = layer->stride[1];
    int stride_w     = layer->stride[2];
    int padding_d    = layer->padding[0];
    int padding_h    = layer->padding[1];
    int padding_w    = layer->padding[2];
    int opad_d       = layer->output_padding[0];
    int opad_h       = layer->output_padding[1];
    int opad_w       = layer->output_padding[2];
    int dilation_d   = layer->dilation[0];
    int dilation_h   = layer->dilation[1];
    int dilation_w   = layer->dilation[2];

    /* D_out = (D_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1 */
    int out_depth  = (in_depth  - 1) * stride_d - 2 * padding_d +
                     dilation_d * (kernel_d - 1) + opad_d + 1;
    int out_height = (in_height - 1) * stride_h - 2 * padding_h +
                     dilation_h * (kernel_h - 1) + opad_h + 1;
    int out_width  = (in_width  - 1) * stride_w - 2 * padding_w +
                     dilation_w * (kernel_w - 1) + opad_w + 1;

    if (out_depth <= 0 || out_height <= 0 || out_width <= 0) {
        LOG_ERROR("ConvTranspose3d: invalid output dimensions (%d x %d x %d)",
                  out_depth, out_height, out_width);
        return NULL;
    }

    int out_shape[] = {batch, out_channels, out_depth, out_height, out_width};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 5, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* w_data   = (float*)layer->weight->tensor->data;
    float* out_data = (float*)output->data;

    if (!in_data || !w_data || !out_data) {
        tensor_free(output);
        return NULL;
    }

    /* Weight shape: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
     * Transposed conv: scatter each input voxel to output */
    for (int b = 0; b < batch; b++) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int id = 0; id < in_depth; id++) {
                for (int ih = 0; ih < in_height; ih++) {
                    for (int iw = 0; iw < in_width; iw++) {
                        float in_val = in_data[((((b * in_channels + ic) * in_depth + id)
                                                  * in_height + ih) * in_width + iw)];

                        for (int oc = 0; oc < out_channels; oc++) {
                            for (int kd = 0; kd < kernel_d; kd++) {
                                int od = id * stride_d - padding_d + kd * dilation_d;
                                if (od < 0 || od >= out_depth)
                                    continue;

                                for (int kh = 0; kh < kernel_h; kh++) {
                                    int oh = ih * stride_h - padding_h + kh * dilation_h;
                                    if (oh < 0 || oh >= out_height)
                                        continue;

                                    for (int kw = 0; kw < kernel_w; kw++) {
                                        int ow = iw * stride_w - padding_w + kw * dilation_w;
                                        if (ow < 0 || ow >= out_width)
                                            continue;

                                        float w_val = w_data[((((ic * out_channels + oc)
                                                                 * kernel_d + kd)
                                                                 * kernel_h + kh)
                                                                 * kernel_w + kw)];
                                        out_data[((((b * out_channels + oc) * out_depth + od)
                                                    * out_height + oh) * out_width + ow)]
                                            += in_val * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* Add bias */
    if (layer->use_bias && layer->bias && layer->bias->tensor) {
        tensor_ensure_executed(layer->bias->tensor);
        float* bias_data = (float*)layer->bias->tensor->data;
        if (bias_data) {
            int spatial = out_depth * out_height * out_width;
            for (int b = 0; b < batch; b++) {
                for (int oc = 0; oc < out_channels; oc++) {
                    float bv = bias_data[oc];
                    int base_idx = ((b * out_channels + oc) * out_depth) * out_height * out_width;
                    for (int s = 0; s < spatial; s++) {
                        out_data[base_idx + s] += bv;
                    }
                }
            }
        }
    }

    return output;
}

static void conv_transpose3d_free(Module* module) {
    ConvTranspose3d* layer = (ConvTranspose3d*)module;
    if (!layer)
        return;
    free(layer);
}

static void kaiming_init_transpose3d(Tensor* tensor, int in_channels,
                                      int kd, int kh, int kw) {
    if (!tensor || !tensor->data)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    float fan_in = (float)(in_channels * kd * kh * kw);
    float scale  = sqrtf(2.0f / fan_in);
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

ConvTranspose3d* nn_conv_transpose3d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device) {
    ConvTranspose3d* layer = calloc(1, sizeof(ConvTranspose3d));
    if (!layer) {
        LOG_ERROR("ConvTranspose3d: failed to allocate memory");
        return NULL;
    }

    if (module_init((Module*)layer, "ConvTranspose3d", conv_transpose3d_forward,
                    conv_transpose3d_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->in_channels       = in_channels;
    layer->out_channels      = out_channels;
    layer->kernel_size[0]    = kernel_size;
    layer->kernel_size[1]    = kernel_size;
    layer->kernel_size[2]    = kernel_size;
    layer->stride[0]         = stride;
    layer->stride[1]         = stride;
    layer->stride[2]         = stride;
    layer->padding[0]        = padding;
    layer->padding[1]        = padding;
    layer->padding[2]        = padding;
    layer->output_padding[0] = output_padding;
    layer->output_padding[1] = output_padding;
    layer->output_padding[2] = output_padding;
    layer->dilation[0]       = 1;
    layer->dilation[1]       = 1;
    layer->dilation[2]       = 1;
    layer->use_bias          = use_bias;

    /* Weight: [in_channels, out_channels, kd, kh, kw] */
    int weight_shape[] = {in_channels, out_channels, kernel_size, kernel_size, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 5, &config);
    if (!weight) {
        module_free((Module*)layer);
        return NULL;
    }

    kaiming_init_transpose3d(weight, in_channels, kernel_size, kernel_size, kernel_size);

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

    LOG_DEBUG("Created ConvTranspose3d layer: %d -> %d, kernel=%d, stride=%d, padding=%d",
              in_channels, out_channels, kernel_size, stride, padding);

    return layer;
}
