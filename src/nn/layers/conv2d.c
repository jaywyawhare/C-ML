#include "nn/layers/conv2d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
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

    int in_channels        = input->shape[1];
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
    int stride_h   = conv2d->stride[0];
    int stride_w   = conv2d->stride[1];
    int padding_h  = conv2d->padding[0];
    int padding_w  = conv2d->padding[1];
    int dilation_h = conv2d->dilation[0];
    int dilation_w = conv2d->dilation[1];

    // Use uop_conv2d for convolution
    Conv2DParams conv_params;
    int stride_arr[]   = {stride_h, stride_w};
    int padding_arr[]  = {padding_h, padding_w};
    int dilation_arr[] = {dilation_h, dilation_w};

    conv_params.stride   = stride_arr;
    conv_params.padding  = padding_arr;
    conv_params.dilation = dilation_arr;

    Tensor* bias = NULL;
    if (conv2d->use_bias && bias_param && bias_param->tensor) {
        bias = bias_param->tensor;
    }

    // Perform convolution using uop_conv2d
    Tensor* output = uop_conv2d(input, weight, bias, &conv_params);

    if (!output) {
        LOG_ERROR("Failed 2D convolution using uop_conv2d");
        return NULL;
    }

    return output;
}

static void conv2d_free(Module* module) {
    Conv2d* conv2d = (Conv2d*)module;
    if (!conv2d)
        return;

    free(conv2d);
}

static void kaiming_init(Tensor* tensor, int in_channels, int out_channels, int kernel_size) {
    (void)out_channels;
    if (!tensor || !tensor->data)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    // He initialization: std = sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    float scale  = sqrtf(2.0f / (float)(in_channels * kernel_size * kernel_size));
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)(float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

Conv2d* nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device) {
    Conv2d* conv2d = malloc(sizeof(Conv2d));
    if (!conv2d)
        return NULL;

    if (module_init((Module*)conv2d, "Conv2d", conv2d_forward, conv2d_free) != 0) {
        free(conv2d);
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
    conv2d->groups         = 1;

    // Create weight tensor [out_channels, in_channels, kernel_h, kernel_w]
    int weight_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 4, &config);
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
        TensorConfig bias_config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        Tensor* bias = tensor_zeros(bias_shape, 1, &bias_config);
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
