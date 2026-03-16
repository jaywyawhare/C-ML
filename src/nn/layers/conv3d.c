#include "nn/layers/conv3d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

static Tensor* conv3d_forward(Module* module, Tensor* input) {
    Conv3d* conv = (Conv3d*)module;
    if (!conv || !input)
        return NULL;
    if (input->ndim != 5) {
        LOG_ERROR("Conv3d expects 5D input [batch, channels, depth, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    tensor_ensure_executed(input);
    tensor_ensure_executed(conv->weight->tensor);

    int batch = input->shape[0];
    int in_ch = input->shape[1];
    int id    = input->shape[2];
    int ih    = input->shape[3];
    int iw    = input->shape[4];

    int out_ch = conv->out_channels;
    int kd = conv->kernel_size[0], kh = conv->kernel_size[1], kw = conv->kernel_size[2];
    int sd = conv->stride[0],      sh = conv->stride[1],      sw = conv->stride[2];
    int pd = conv->padding[0],     ph = conv->padding[1],     pw = conv->padding[2];
    int dd = conv->dilation[0],    dh = conv->dilation[1],    dw = conv->dilation[2];

    int od = (id + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
    int oh = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int ow = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    int out_shape[] = {batch, out_ch, od, oh, ow};
    TensorConfig config =
        (TensorConfig){.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 5, &config);
    tensor_ensure_executed(output);

    float* in_data  = (float*)tensor_data_ptr(input);
    float* w_data   = (float*)tensor_data_ptr(conv->weight->tensor);
    float* out_data = (float*)tensor_data_ptr(output);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int d = 0; d < od; d++) {
                for (int h = 0; h < oh; h++) {
                    for (int w = 0; w < ow; w++) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < in_ch; ic++) {
                            for (int kdi = 0; kdi < kd; kdi++) {
                                for (int khi = 0; khi < kh; khi++) {
                                    for (int kwi = 0; kwi < kw; kwi++) {
                                        int iz = d * sd - pd + kdi * dd;
                                        int iy = h * sh - ph + khi * dh;
                                        int ix = w * sw - pw + kwi * dw;
                                        if (iz >= 0 && iz < id && iy >= 0 && iy < ih &&
                                            ix >= 0 && ix < iw) {
                                            int in_idx = ((b * in_ch + ic) * id + iz) * ih * iw +
                                                         iy * iw + ix;
                                            int w_idx = ((oc * in_ch + ic) * kd + kdi) * kh * kw +
                                                        khi * kw + kwi;
                                            sum += in_data[in_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        int out_idx =
                            ((b * out_ch + oc) * od + d) * oh * ow + h * ow + w;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }
    }
    if (conv->use_bias && conv->bias && conv->bias->tensor) {
        tensor_ensure_executed(conv->bias->tensor);
        float* bias_data = (float*)tensor_data_ptr(conv->bias->tensor);
        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < out_ch; oc++) {
                for (int d = 0; d < od; d++) {
                    for (int h = 0; h < oh; h++) {
                        for (int w = 0; w < ow; w++) {
                            out_data[((b * out_ch + oc) * od + d) * oh * ow + h * ow + w] +=
                                bias_data[oc];
                        }
                    }
                }
            }
        }
    }

    return output;
}

static void conv3d_free(Module* module) {
    Conv3d* conv3d = (Conv3d*)module;
    if (!conv3d)
        return;

    free(conv3d);
}

static void kaiming_init_3d(Tensor* tensor, int in_channels, int kernel_size) {
    if (!tensor || !tensor->data)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;
    float scale  = sqrtf(2.0f / (float)(in_channels * kernel_size * kernel_size * kernel_size));
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

Conv3d* nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device) {
    Conv3d* conv3d = malloc(sizeof(Conv3d));
    if (!conv3d)
        return NULL;

    if (module_init((Module*)conv3d, "Conv3d", conv3d_forward, conv3d_free) != 0) {
        free(conv3d);
        return NULL;
    }

    conv3d->in_channels    = in_channels;
    conv3d->out_channels   = out_channels;
    conv3d->kernel_size[0] = kernel_size;
    conv3d->kernel_size[1] = kernel_size;
    conv3d->kernel_size[2] = kernel_size;
    conv3d->stride[0]      = stride;
    conv3d->stride[1]      = stride;
    conv3d->stride[2]      = stride;
    conv3d->padding[0]     = padding;
    conv3d->padding[1]     = padding;
    conv3d->padding[2]     = padding;
    conv3d->dilation[0]    = dilation;
    conv3d->dilation[1]    = dilation;
    conv3d->dilation[2]    = dilation;
    conv3d->use_bias       = use_bias;
    conv3d->groups         = 1;
    int weight_shape[] = {out_channels, in_channels, kernel_size, kernel_size, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 5, &config);
    if (!weight) {
        module_free((Module*)conv3d);
        return NULL;
    }
    kaiming_init_3d(weight, in_channels, kernel_size);

    if (module_add_parameter((Module*)conv3d, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)conv3d);
        return NULL;
    }

    conv3d->weight = module_get_parameter((Module*)conv3d, "weight");
    if (use_bias) {
        int bias_shape[] = {out_channels};
        TensorConfig bias_config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        Tensor* bias = tensor_zeros(bias_shape, 1, &bias_config);
        if (!bias) {
            module_free((Module*)conv3d);
            return NULL;
        }

        if (module_add_parameter((Module*)conv3d, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)conv3d);
            return NULL;
        }

        conv3d->bias = module_get_parameter((Module*)conv3d, "bias");
    } else {
        conv3d->bias = NULL;
    }

    return conv3d;
}
