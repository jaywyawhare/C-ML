#include "nn/layers/conv1d.h"
#include "nn.h"
#include "tensor/tensor.h"
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

    int in_channels        = input->shape[1];
    int weight_in_channels = weight_param->tensor->shape[1];
    if (in_channels != weight_in_channels) {
        LOG_ERROR("Conv1d: input channels (%d) doesn't match weight in_channels (%d)", in_channels,
                  weight_in_channels);
        return NULL;
    }
    if (in_channels != conv->in_channels) {
        LOG_ERROR("Conv1d: input dimensions don't match layer configuration");
        return NULL;
    }
    tensor_ensure_executed(input);
    tensor_ensure_executed(weight_param->tensor);
    int batch   = input->shape[0];
    int in_ch   = input->shape[1];
    int length  = input->shape[2];
    int out_ch  = conv->out_channels;
    int ks      = conv->kernel_size;
    int s       = conv->stride;
    int p       = conv->padding;
    int d       = conv->dilation;
    int out_len = (length + 2 * p - d * (ks - 1) - 1) / s + 1;

    if (out_len <= 0) {
        LOG_ERROR("Conv1d: invalid output length %d (input=%d, kernel=%d, stride=%d, padding=%d, dilation=%d)",
                  out_len, length, ks, s, p, d);
        return NULL;
    }
    int out_shape[] = {batch, out_ch, out_len};
    TensorConfig config =
        (TensorConfig){.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 3, &config);
    if (!output) {
        LOG_ERROR("Conv1d: failed to allocate output tensor");
        return NULL;
    }
    tensor_ensure_executed(output);

    float* in_data  = (float*)tensor_data_ptr(input);
    float* w_data   = (float*)tensor_data_ptr(weight_param->tensor);
    float* out_data = (float*)tensor_data_ptr(output);

    if (!in_data || !w_data || !out_data) {
        LOG_ERROR("Conv1d: failed to get data pointers");
        return NULL;
    }
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int ol = 0; ol < out_len; ol++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int k = 0; k < ks; k++) {
                        int il = ol * s - p + k * d;
                        if (il >= 0 && il < length) {
                            int in_idx = b * in_ch * length + ic * length + il;
                            int w_idx  = oc * in_ch * ks + ic * ks + k;
                            sum += in_data[in_idx] * w_data[w_idx];
                        }
                    }
                }
                int out_idx       = b * out_ch * out_len + oc * out_len + ol;
                out_data[out_idx] = sum;
            }
        }
    }
    if (conv->use_bias && bias_param && bias_param->tensor) {
        tensor_ensure_executed(bias_param->tensor);
        float* bias_data = (float*)tensor_data_ptr(bias_param->tensor);
        if (bias_data) {
            for (int b = 0; b < batch; b++) {
                for (int oc = 0; oc < out_ch; oc++) {
                    for (int ol = 0; ol < out_len; ol++) {
                        out_data[b * out_ch * out_len + oc * out_len + ol] += bias_data[oc];
                    }
                }
            }
        }
    }

    return output;
}

static void conv1d_free(Module* module) {
    Conv1d* conv = (Conv1d*)module;
    if (!conv)
        return;

    free(conv);
}

static void kaiming_init_1d(Tensor* tensor, int in_channels, int kernel_size) {
    if (!tensor || !tensor->data)
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
