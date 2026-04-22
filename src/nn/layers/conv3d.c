#include "nn/layers/conv3d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
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

    Conv3DParams params = {
        .kernel_size = {conv->kernel_size[0], conv->kernel_size[1], conv->kernel_size[2]},
        .stride = {conv->stride[0], conv->stride[1], conv->stride[2]},
        .padding = {conv->padding[0], conv->padding[1], conv->padding[2]},
        .dilation = {conv->dilation[0], conv->dilation[1], conv->dilation[2]},
        .use_bias = conv->use_bias,
    };
    Tensor* bias = (conv->use_bias && conv->bias) ? conv->bias->tensor : NULL;
    return uop_conv3d(input, conv->weight->tensor, bias, &params);
}

static void conv3d_free(Module* module) {
    Conv3d* conv3d = (Conv3d*)module;
    if (!conv3d)
        return;

    free(conv3d);
}

static void kaiming_init_3d(Tensor* tensor, int in_channels, int kernel_size) {
    if (!tensor)
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
