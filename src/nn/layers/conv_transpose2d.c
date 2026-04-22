#include "nn/layers/conv_transpose2d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* conv_transpose2d_forward(Module* module, Tensor* input) {
    ConvTranspose2d* layer = (ConvTranspose2d*)module;

    if (!layer || !input)
        return NULL;

    if (input->ndim != 4) {
        LOG_ERROR("ConvTranspose2d expects 4D input [batch, in_channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    if (!layer->weight || !layer->weight->tensor) {
        LOG_ERROR("ConvTranspose2d missing weight parameter");
        return NULL;
    }
    int in_channels = input->shape[1];

    if (in_channels != layer->in_channels) {
        LOG_ERROR("ConvTranspose2d: input channels (%d) doesn't match expected (%d)",
                  in_channels, layer->in_channels);
        return NULL;
    }

    ConvTranspose2DParams params = {
        .kernel_size = {layer->kernel_size[0], layer->kernel_size[1]},
        .stride = {layer->stride[0], layer->stride[1]},
        .padding = {layer->padding[0], layer->padding[1]},
        .output_padding = {layer->output_padding[0], layer->output_padding[1]},
        .dilation = {layer->dilation[0], layer->dilation[1]},
        .use_bias = layer->use_bias,
    };
    Tensor* bias = (layer->use_bias && layer->bias) ? layer->bias->tensor : NULL;
    return uop_conv_transpose2d(input, layer->weight->tensor, bias, &params);
}

static void conv_transpose2d_free(Module* module) {
    ConvTranspose2d* layer = (ConvTranspose2d*)module;
    if (!layer)
        return;
    free(layer);
}

static void kaiming_init_transpose(Tensor* tensor, int in_channels, int kernel_size) {
    if (!tensor)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    float scale  = sqrtf(2.0f / (float)(in_channels * kernel_size * kernel_size));
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

ConvTranspose2d* nn_conv_transpose2d(int in_channels, int out_channels, int kernel_size,
                                      int stride, int padding, int output_padding,
                                      bool use_bias, DType dtype, DeviceType device) {
    ConvTranspose2d* layer = malloc(sizeof(ConvTranspose2d));
    if (!layer)
        return NULL;

    if (module_init((Module*)layer, "ConvTranspose2d", conv_transpose2d_forward,
                    conv_transpose2d_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->in_channels       = in_channels;
    layer->out_channels      = out_channels;
    layer->kernel_size[0]    = kernel_size;
    layer->kernel_size[1]    = kernel_size;
    layer->stride[0]         = stride;
    layer->stride[1]         = stride;
    layer->padding[0]        = padding;
    layer->padding[1]        = padding;
    layer->output_padding[0] = output_padding;
    layer->output_padding[1] = output_padding;
    layer->dilation[0]       = 1;
    layer->dilation[1]       = 1;
    layer->use_bias          = use_bias;
    int weight_shape[] = {in_channels, out_channels, kernel_size, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 4, &config);
    if (!weight) {
        module_free((Module*)layer);
        return NULL;
    }

    kaiming_init_transpose(weight, in_channels, kernel_size);

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
