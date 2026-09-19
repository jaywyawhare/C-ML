#include "nn/layers/conv_transpose1d.h"
#include "nn/init.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

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

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_length   = input->shape[2];

    if (in_channels != layer->in_channels) {
        LOG_ERROR("ConvTranspose1d: input channels (%d) doesn't match expected (%d)", in_channels,
                  layer->in_channels);
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

    /* Lazy: run as a height-1 ConvTranspose2d so it builds IR, decomposes to
     * primitives, and is graph-autodiff differentiable (was an eager loop). */
    Tensor* w        = layer->weight->tensor; /* [Cin,Cout,ks] */
    int xs[4]        = {batch, in_channels, 1, in_length};
    ReshapeParams ri = {xs, 4};
    Tensor* x4       = uop_reshape(input, &ri);
    if (!x4)
        return NULL;
    int wsz[4]       = {w->shape[0], w->shape[1], 1, ks};
    ReshapeParams rw = {wsz, 4};
    Tensor* w4       = uop_reshape(w, &rw);
    if (!w4)
        return NULL;

    Tensor* bias             = (layer->use_bias && layer->bias) ? layer->bias->tensor : NULL;
    ConvTranspose2DParams p2 = {0};
    p2.kernel_size[0]        = 1;
    p2.kernel_size[1]        = ks;
    p2.stride[0]             = 1;
    p2.stride[1]             = s;
    p2.padding[0]            = 0;
    p2.padding[1]            = p;
    p2.output_padding[0]     = 0;
    p2.output_padding[1]     = opad;
    p2.dilation[0]           = 1;
    p2.dilation[1]           = d;
    p2.use_bias              = bias != NULL;
    Tensor* y4               = uop_conv_transpose2d(x4, w4, bias, &p2); /* [N,Cout,1,OL] */
    if (!y4)
        return NULL;

    int ys[3]        = {batch, out_channels, out_length};
    ReshapeParams ro = {ys, 3};
    return uop_reshape(y4, &ro);
}

static void conv_transpose1d_free(Module* module) {
    ConvTranspose1d* layer = (ConvTranspose1d*)module;
    if (!layer)
        return;
    cml_free(layer);
}

ConvTranspose1d* nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size, int stride,
                                     int padding, int output_padding, bool use_bias, DType dtype,
                                     DeviceType device) {
    ConvTranspose1d* layer = cml_malloc(sizeof(ConvTranspose1d));
    if (!layer)
        return NULL;

    if (module_init((Module*)layer, "ConvTranspose1d", conv_transpose1d_forward,
                    conv_transpose1d_free) != 0) {
        cml_free(layer);
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
    int weight_shape[]    = {in_channels, out_channels, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 3, &config);
    if (weight)
        nn_init_kaiming(weight, in_channels, kernel_size);
    layer->weight = nn_add_weight_param((Module*)layer, weight);
    if (!layer->weight)
        return NULL;

    if (use_bias) {
        layer->bias = nn_add_bias_param((Module*)layer, out_channels, dtype, device, NULL);
        if (!layer->bias)
            return NULL;
    } else {
        layer->bias = NULL;
    }

    return layer;
}
