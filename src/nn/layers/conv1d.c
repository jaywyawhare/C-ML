#include "nn/layers/conv1d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

static Tensor* conv1d_forward(Module* module, Tensor* input) {
    Conv1d* conv = (Conv1d*)module;

    if (!conv || !input)
        return NULL;
    if (input->ndim != 3) {
        LOG_ERROR("Conv1d expects 3D input [batch, in_channels, length], got %dD", input->ndim);
        return NULL;
    }

    Parameter* weight_param = conv->weight;
    Parameter* bias_param   = conv->bias;

    if (!weight_param || !weight_param->tensor) {
        LOG_ERROR("Conv1d missing weight parameter");
        return NULL;
    }

    Tensor* weight = weight_param->tensor;

    int in_channels        = input->shape[1];
    int weight_in_channels = weight->shape[1];
    if (in_channels != weight_in_channels) {
        LOG_ERROR("Conv1d: input channels (%d) doesn't match weight in_channels (%d)", in_channels,
                  weight_in_channels);
        return NULL;
    }
    if (in_channels != conv->in_channels) {
        LOG_ERROR("Conv1d: input dimensions don't match layer configuration");
        return NULL;
    }

    int batch   = input->shape[0];
    int in_ch   = input->shape[1];
    int length  = input->shape[2];
    int out_ch  = conv->out_channels;
    int ks      = conv->kernel_size;
    int s       = conv->stride;
    int p       = conv->padding;
    int d       = conv->dilation;

    if (weight->shape[0] != out_ch || weight->shape[2] != ks) {
        LOG_ERROR("Conv1d: weight shape mismatch");
        return NULL;
    }

    /* Map 1D conv to 2D: [B,C,L] -> [B,C,1,L], weight [OC,IC,K] -> [OC,IC,1,K]. */
    int in4_shape[]  = {batch, in_ch, 1, length};
    int w4_shape[]   = {out_ch, in_ch, 1, ks};
    ReshapeParams rp_in = {.new_shape = in4_shape, .new_ndim = 4};
    ReshapeParams rp_w  = {.new_shape = w4_shape, .new_ndim = 4};

    Tensor* in4 = uop_reshape(input, &rp_in);
    if (!in4)
        return NULL;
    Tensor* w4 = uop_reshape(weight, &rp_w);
    if (!w4)
        return NULL;

    int stride_arr[]   = {1, s};
    int padding_arr[]  = {0, p};
    int dilation_arr[] = {1, d};
    Conv2DParams conv_params;
    conv_params.stride   = stride_arr;
    conv_params.padding  = padding_arr;
    conv_params.dilation = dilation_arr;
    conv_params.groups   = 1;

    Tensor* bias = NULL;
    if (conv->use_bias && bias_param && bias_param->tensor)
        bias = bias_param->tensor;

    Tensor* out4 = uop_conv2d(in4, w4, bias, &conv_params);
    if (!out4)
        return NULL;

    int out_len = (length + 2 * p - d * (ks - 1) - 1) / s + 1;
    int out3_shape[] = {batch, out_ch, out_len};
    ReshapeParams rp_out = {.new_shape = out3_shape, .new_ndim = 3};
    return uop_reshape(out4, &rp_out);
}

static void conv1d_free(Module* module) {
    Conv1d* conv = (Conv1d*)module;
    if (!conv)
        return;

    free(conv);
}

static void kaiming_init_1d(Tensor* tensor, int in_channels, int kernel_size) {
    if (!tensor)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;
    float scale  = sqrtf(2.0f / (float)(in_channels * kernel_size));
    size_t numel = tensor->numel;

    for (size_t i = 0; i < numel; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

Conv1d* nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                  int dilation, bool use_bias, DType dtype, DeviceType device) {
    Conv1d* conv = malloc(sizeof(Conv1d));
    if (!conv)
        return NULL;

    if (module_init((Module*)conv, "Conv1d", conv1d_forward, conv1d_free) != 0) {
        free(conv);
        return NULL;
    }

    conv->in_channels  = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size  = kernel_size;
    conv->stride       = stride;
    conv->padding      = padding;
    conv->dilation     = dilation;
    conv->use_bias     = use_bias;
    conv->groups       = 1;
    int weight_shape[] = {out_channels, in_channels, kernel_size};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_empty(weight_shape, 3, &config);
    if (!weight) {
        module_free((Module*)conv);
        return NULL;
    }
    kaiming_init_1d(weight, in_channels, kernel_size);

    if (module_add_parameter((Module*)conv, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)conv);
        return NULL;
    }

    conv->weight = module_get_parameter((Module*)conv, "weight");
    if (use_bias) {
        int bias_shape[] = {out_channels};
        TensorConfig bias_config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        Tensor* bias = tensor_zeros(bias_shape, 1, &bias_config);
        if (!bias) {
            module_free((Module*)conv);
            return NULL;
        }

        if (module_add_parameter((Module*)conv, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)conv);
            return NULL;
        }

        conv->bias = module_get_parameter((Module*)conv, "bias");
    } else {
        conv->bias = NULL;
    }

    return conv;
}
