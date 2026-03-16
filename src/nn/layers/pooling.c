/**
 * @file pooling.c
 * @brief Pooling layers implementation using stride-based views
 */

#include "nn/layers/pooling.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

static Tensor* create_pooling_window_view(Tensor* input, int b, int c, int oh, int ow, int kernel_h,
                                          int kernel_w, int stride_h, int stride_w, int padding_h,
                                          int padding_w, int dilation_h, int dilation_w,
                                          int in_height, int in_width) {
    int start_h = oh * stride_h - padding_h;
    int start_w = ow * stride_w - padding_w;
    int window_shape[] = {kernel_h, kernel_w};
    // The window strides should step by dilation in the input
    size_t* window_strides = malloc(2 * sizeof(size_t));
    if (!window_strides)
        return NULL;
    size_t input_h_stride = input->strides ? input->strides[2] : (size_t)(in_width);
    size_t input_w_stride = input->strides ? input->strides[3] : 1;
    window_strides[0] = (size_t)input_h_stride * (size_t)dilation_h;
    window_strides[1] = (size_t)input_w_stride * (size_t)dilation_w;
    size_t storage_offset = 0;
    if (input->strides) {
        int indices[]      = {b, c, start_h, start_w};
        size_t elem_offset = tensor_compute_offset(input, indices);
        storage_offset     = elem_offset * cml_dtype_size(input->dtype);
    } else {
        size_t elem_offset =
            ((size_t)b * (size_t)input->shape[1] * (size_t)in_height * (size_t)in_width +
             (size_t)c * (size_t)in_height * (size_t)in_width + (size_t)start_h * (size_t)in_width +
             (size_t)start_w);
        storage_offset = elem_offset * cml_dtype_size(input->dtype);
    }
    Tensor* window_view = tensor_as_strided(input, window_shape, 2, window_strides, storage_offset);

    free(window_strides);
    return window_view;
}

static Tensor* maxpool2d_forward(Module* module, Tensor* input) {
    MaxPool2d* pool = (MaxPool2d*)module;

    if (!pool || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("MaxPool2d expects 4D input [batch, channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    int batch     = input->shape[0];
    int channels  = input->shape[1];
    int in_height = input->shape[2];
    int in_width  = input->shape[3];

    int kernel_h   = pool->kernel_size[0];
    int kernel_w   = pool->kernel_size[1];
    int stride_h   = pool->stride[0];
    int stride_w   = pool->stride[1];
    int padding_h  = pool->padding[0];
    int padding_w  = pool->padding[1];
    int dilation_h = pool->dilation[0];
    int dilation_w = pool->dilation[1];
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (pool->ceil_mode) {
        int actual_h = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int actual_w = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        if ((in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) % stride_h != 0) {
            out_height = actual_h + 1;
        }
        if ((in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) % stride_w != 0) {
            out_width = actual_w + 1;
        }
    }
    int output_shape[]  = {batch, channels, out_height, out_width};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(output_shape, 4, &config);
    if (!output)
        return NULL;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    // Create strided view for this pooling window
                    Tensor* window = create_pooling_window_view(
                        input, b, c, oh, ow, kernel_h, kernel_w, stride_h, stride_w, padding_h,
                        padding_w, dilation_h, dilation_w, in_height, in_width);
                    if (!window) {
                        tensor_free(output);
                        return NULL;
                    }
                    ReduceParams reduce_params;
                    int dims[]             = {0, 1}; // Reduce over both spatial dimensions
                    reduce_params.dims     = dims;
                    reduce_params.num_dims = 2;
                    reduce_params.keepdim  = false;

                    Tensor* reduced = uop_max_reduce(window, &reduce_params);
                    tensor_free(window);

                    if (!reduced) {
                        tensor_free(output);
                        return NULL;
                    }
                    float* reduced_data = (float*)tensor_data_ptr(reduced);
                    if (reduced_data) {
                        int output_indices[] = {b, c, oh, ow};
                        size_t output_offset = tensor_compute_offset(output, output_indices);
                        float* output_data   = (float*)tensor_data_ptr(output);
                        if (output_data) {
                            output_data[output_offset] = reduced_data[0];
                        }
                    }
                    tensor_free(reduced);
                }
            }
        }
    }

    return output;
}

static void maxpool2d_free(Module* module) { free(module); }

MaxPool2d* nn_maxpool2d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool2d* pool = malloc(sizeof(MaxPool2d));
    if (!pool)
        return NULL;

    if (module_init((Module*)pool, "MaxPool2d", maxpool2d_forward, maxpool2d_free) != 0) {
        free(pool);
        return NULL;
    }

    pool->kernel_size[0] = kernel_size;
    pool->kernel_size[1] = kernel_size;
    pool->stride[0]      = stride > 0 ? stride : kernel_size;
    pool->stride[1]      = stride > 0 ? stride : kernel_size;
    pool->padding[0]     = padding;
    pool->padding[1]     = padding;
    pool->dilation[0]    = dilation;
    pool->dilation[1]    = dilation;
    pool->ceil_mode      = ceil_mode;

    return pool;
}

static Tensor* avgpool2d_forward(Module* module, Tensor* input) {
    AvgPool2d* pool = (AvgPool2d*)module;

    if (!pool || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("AvgPool2d expects 4D input [batch, channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    int batch     = input->shape[0];
    int channels  = input->shape[1];
    int in_height = input->shape[2];
    int in_width  = input->shape[3];

    int kernel_h  = pool->kernel_size[0];
    int kernel_w  = pool->kernel_size[1];
    int stride_h  = pool->stride[0];
    int stride_w  = pool->stride[1];
    int padding_h = pool->padding[0];
    int padding_w = pool->padding[1];
    int out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;
    if (pool->ceil_mode) {
        int actual_h = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
        int actual_w = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;
        if ((in_height + 2 * padding_h - kernel_h) % stride_h != 0) {
            out_height = actual_h + 1;
        }
        if ((in_width + 2 * padding_w - kernel_w) % stride_w != 0) {
            out_width = actual_w + 1;
        }
    }
    int output_shape[]  = {batch, channels, out_height, out_width};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(output_shape, 4, &config);
    if (!output)
        return NULL;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    // Create strided view for this pooling window
                    Tensor* window = create_pooling_window_view(
                        input, b, c, oh, ow, kernel_h, kernel_w, stride_h, stride_w, padding_h,
                        padding_w, 1, 1, // No dilation for avg pool
                        in_height, in_width);
                    if (!window) {
                        tensor_free(output);
                        return NULL;
                    }
                    ReduceParams reduce_params;
                    int dims[]             = {0, 1}; // Reduce over both spatial dimensions
                    reduce_params.dims     = dims;
                    reduce_params.num_dims = 2;
                    reduce_params.keepdim  = false;

                    Tensor* reduced = uop_mean(window, &reduce_params);
                    tensor_free(window);

                    if (!reduced) {
                        tensor_free(output);
                        return NULL;
                    }
                    float* reduced_data = (float*)tensor_data_ptr(reduced);
                    if (reduced_data) {
                        int output_indices[] = {b, c, oh, ow};
                        size_t output_offset = tensor_compute_offset(output, output_indices);
                        float* output_data   = (float*)tensor_data_ptr(output);
                        if (output_data) {
                            output_data[output_offset] = reduced_data[0];
                        }
                    }
                    tensor_free(reduced);
                }
            }
        }
    }

    return output;
}

static void avgpool2d_free(Module* module) { free(module); }

AvgPool2d* nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad) {
    AvgPool2d* pool = malloc(sizeof(AvgPool2d));
    if (!pool)
        return NULL;

    if (module_init((Module*)pool, "AvgPool2d", avgpool2d_forward, avgpool2d_free) != 0) {
        free(pool);
        return NULL;
    }

    pool->kernel_size[0]    = kernel_size;
    pool->kernel_size[1]    = kernel_size;
    pool->stride[0]         = stride > 0 ? stride : kernel_size;
    pool->stride[1]         = stride > 0 ? stride : kernel_size;
    pool->padding[0]        = padding;
    pool->padding[1]        = padding;
    pool->ceil_mode         = ceil_mode;
    pool->count_include_pad = count_include_pad;

    return pool;
}

static Tensor* maxpool3d_forward(Module* module, Tensor* input) {
    MaxPool3d* pool = (MaxPool3d*)module;
    if (!input || input->ndim != 5) return NULL;  // [N, C, D, H, W]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1];
    int D = input->shape[2], H = input->shape[3], W = input->shape[4];

    int out_d = (D + 2 * pool->padding[0] - pool->dilation[0] * (pool->kernel_size[0] - 1) - 1) / pool->stride[0] + 1;
    int out_h = (H + 2 * pool->padding[1] - pool->dilation[1] * (pool->kernel_size[1] - 1) - 1) / pool->stride[1] + 1;
    int out_w = (W + 2 * pool->padding[2] - pool->dilation[2] * (pool->kernel_size[2] - 1) - 1) / pool->stride[2] + 1;

    if (pool->ceil_mode) {
        if ((D + 2 * pool->padding[0] - pool->dilation[0] * (pool->kernel_size[0] - 1) - 1) % pool->stride[0] != 0) out_d++;
        if ((H + 2 * pool->padding[1] - pool->dilation[1] * (pool->kernel_size[1] - 1) - 1) % pool->stride[1] != 0) out_h++;
        if ((W + 2 * pool->padding[2] - pool->dilation[2] * (pool->kernel_size[2] - 1) - 1) % pool->stride[2] != 0) out_w++;
    }

    int out_shape[] = {N, C, out_d, out_h, out_w};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 5, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int od = 0; od < out_d; od++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float max_val = -FLT_MAX;
                        for (int kd = 0; kd < pool->kernel_size[0]; kd++) {
                            int id = od * pool->stride[0] - pool->padding[0] + kd * pool->dilation[0];
                            if (id < 0 || id >= D) continue;
                            for (int kh = 0; kh < pool->kernel_size[1]; kh++) {
                                int ih = oh * pool->stride[1] - pool->padding[1] + kh * pool->dilation[1];
                                if (ih < 0 || ih >= H) continue;
                                for (int kw = 0; kw < pool->kernel_size[2]; kw++) {
                                    int iw = ow * pool->stride[2] - pool->padding[2] + kw * pool->dilation[2];
                                    if (iw < 0 || iw >= W) continue;
                                    float v = in_data[n*C*D*H*W + c*D*H*W + id*H*W + ih*W + iw];
                                    if (v > max_val) max_val = v;
                                }
                            }
                        }
                        out_data[n*C*out_d*out_h*out_w + c*out_d*out_h*out_w + od*out_h*out_w + oh*out_w + ow] = max_val;
                    }
                }
            }
        }
    }
    return output;
}

static void maxpool3d_free(Module* module) { free(module); }

MaxPool3d* nn_maxpool3d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool3d* pool = malloc(sizeof(MaxPool3d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "MaxPool3d", maxpool3d_forward, maxpool3d_free) != 0) {
        free(pool); return NULL;
    }
    for (int i = 0; i < 3; i++) {
        pool->kernel_size[i] = kernel_size;
        pool->stride[i] = stride > 0 ? stride : kernel_size;
        pool->padding[i] = padding;
        pool->dilation[i] = dilation > 0 ? dilation : 1;
    }
    pool->ceil_mode = ceil_mode;
    return pool;
}

static Tensor* avgpool3d_forward(Module* module, Tensor* input) {
    AvgPool3d* pool = (AvgPool3d*)module;
    if (!input || input->ndim != 5) return NULL;  // [N, C, D, H, W]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1];
    int D = input->shape[2], H = input->shape[3], W = input->shape[4];

    int out_d = (D + 2 * pool->padding[0] - pool->kernel_size[0]) / pool->stride[0] + 1;
    int out_h = (H + 2 * pool->padding[1] - pool->kernel_size[1]) / pool->stride[1] + 1;
    int out_w = (W + 2 * pool->padding[2] - pool->kernel_size[2]) / pool->stride[2] + 1;

    if (pool->ceil_mode) {
        if ((D + 2 * pool->padding[0] - pool->kernel_size[0]) % pool->stride[0] != 0) out_d++;
        if ((H + 2 * pool->padding[1] - pool->kernel_size[1]) % pool->stride[1] != 0) out_h++;
        if ((W + 2 * pool->padding[2] - pool->kernel_size[2]) % pool->stride[2] != 0) out_w++;
    }

    int out_shape[] = {N, C, out_d, out_h, out_w};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 5, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int od = 0; od < out_d; od++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float sum = 0.0f;
                        int count = 0;
                        for (int kd = 0; kd < pool->kernel_size[0]; kd++) {
                            int id = od * pool->stride[0] - pool->padding[0] + kd;
                            if (id < 0 || id >= D) { if (pool->count_include_pad) count++; continue; }
                            for (int kh = 0; kh < pool->kernel_size[1]; kh++) {
                                int ih = oh * pool->stride[1] - pool->padding[1] + kh;
                                if (ih < 0 || ih >= H) { if (pool->count_include_pad) count++; continue; }
                                for (int kw = 0; kw < pool->kernel_size[2]; kw++) {
                                    int iw = ow * pool->stride[2] - pool->padding[2] + kw;
                                    if (iw < 0 || iw >= W) { if (pool->count_include_pad) count++; continue; }
                                    sum += in_data[n*C*D*H*W + c*D*H*W + id*H*W + ih*W + iw];
                                    count++;
                                }
                            }
                        }
                        out_data[n*C*out_d*out_h*out_w + c*out_d*out_h*out_w + od*out_h*out_w + oh*out_w + ow] =
                            count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }
    }
    return output;
}

static void avgpool3d_free(Module* module) { free(module); }

AvgPool3d* nn_avgpool3d(int kernel_size, int stride, int padding, bool ceil_mode, bool count_include_pad) {
    AvgPool3d* pool = malloc(sizeof(AvgPool3d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "AvgPool3d", avgpool3d_forward, avgpool3d_free) != 0) {
        free(pool); return NULL;
    }
    for (int i = 0; i < 3; i++) {
        pool->kernel_size[i] = kernel_size;
        pool->stride[i] = stride > 0 ? stride : kernel_size;
        pool->padding[i] = padding;
    }
    pool->ceil_mode = ceil_mode;
    pool->count_include_pad = count_include_pad;
    return pool;
}

static Tensor* maxpool1d_forward(Module* module, Tensor* input) {
    MaxPool1d* pool = (MaxPool1d*)module;
    if (!input || input->ndim != 3) return NULL;  // [N, C, L]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1], L = input->shape[2];
    int out_l;
    if (pool->ceil_mode)
        out_l = (int)ceilf((float)(L + 2 * pool->padding - pool->dilation * (pool->kernel_size - 1) - 1) / pool->stride + 1);
    else
        out_l = (L + 2 * pool->padding - pool->dilation * (pool->kernel_size - 1) - 1) / pool->stride + 1;

    int out_shape[] = {N, C, out_l};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 3, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < out_l; ol++) {
                float max_val = -FLT_MAX;
                for (int k = 0; k < pool->kernel_size; k++) {
                    int il = ol * pool->stride - pool->padding + k * pool->dilation;
                    if (il >= 0 && il < L) {
                        float v = in_data[n * C * L + c * L + il];
                        if (v > max_val) max_val = v;
                    }
                }
                out_data[n * C * out_l + c * out_l + ol] = max_val;
            }
        }
    }
    return output;
}

static void maxpool1d_free(Module* module) { free(module); }

MaxPool1d* nn_maxpool1d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool1d* pool = malloc(sizeof(MaxPool1d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "MaxPool1d", maxpool1d_forward, maxpool1d_free) != 0) {
        free(pool); return NULL;
    }
    pool->kernel_size = kernel_size;
    pool->stride = stride > 0 ? stride : kernel_size;
    pool->padding = padding;
    pool->dilation = dilation > 0 ? dilation : 1;
    pool->ceil_mode = ceil_mode;
    return pool;
}

static Tensor* avgpool1d_forward(Module* module, Tensor* input) {
    AvgPool1d* pool = (AvgPool1d*)module;
    if (!input || input->ndim != 3) return NULL;  // [N, C, L]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1], L = input->shape[2];
    int out_l;
    if (pool->ceil_mode)
        out_l = (int)ceilf((float)(L + 2 * pool->padding - pool->kernel_size) / pool->stride + 1);
    else
        out_l = (L + 2 * pool->padding - pool->kernel_size) / pool->stride + 1;

    int out_shape[] = {N, C, out_l};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 3, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < out_l; ol++) {
                float sum = 0.0f;
                int count = 0;
                for (int k = 0; k < pool->kernel_size; k++) {
                    int il = ol * pool->stride - pool->padding + k;
                    if (il >= 0 && il < L) {
                        sum += in_data[n * C * L + c * L + il];
                        count++;
                    } else if (pool->count_include_pad) {
                        count++;
                    }
                }
                out_data[n * C * out_l + c * out_l + ol] = count > 0 ? sum / count : 0.0f;
            }
        }
    }
    return output;
}

static void avgpool1d_free(Module* module) { free(module); }

AvgPool1d* nn_avgpool1d(int kernel_size, int stride, int padding, bool ceil_mode, bool count_include_pad) {
    AvgPool1d* pool = malloc(sizeof(AvgPool1d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "AvgPool1d", avgpool1d_forward, avgpool1d_free) != 0) {
        free(pool); return NULL;
    }
    pool->kernel_size = kernel_size;
    pool->stride = stride > 0 ? stride : kernel_size;
    pool->padding = padding;
    pool->ceil_mode = ceil_mode;
    pool->count_include_pad = count_include_pad;
    return pool;
}

static Tensor* adaptive_avgpool2d_forward(Module* module, Tensor* input) {
    AdaptiveAvgPool2d* pool = (AdaptiveAvgPool2d*)module;
    if (!input || input->ndim != 4) return NULL;  // [N, C, H, W]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1];
    int in_h = input->shape[2], in_w = input->shape[3];
    int out_h = pool->output_size[0], out_w = pool->output_size[1];

    int out_shape[] = {N, C, out_h, out_w};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 4, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                int h_start = (int)floorf((float)oh * in_h / out_h);
                int h_end = (int)ceilf((float)(oh + 1) * in_h / out_h);
                for (int ow = 0; ow < out_w; ow++) {
                    int w_start = (int)floorf((float)ow * in_w / out_w);
                    int w_end = (int)ceilf((float)(ow + 1) * in_w / out_w);
                    float sum = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            sum += in_data[n * C * in_h * in_w + c * in_h * in_w + h * in_w + w];
                            count++;
                        }
                    }
                    out_data[n * C * out_h * out_w + c * out_h * out_w + oh * out_w + ow] =
                        count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }
    return output;
}

static void adaptive_avgpool2d_free(Module* module) { free(module); }

AdaptiveAvgPool2d* nn_adaptive_avgpool2d(int output_h, int output_w) {
    AdaptiveAvgPool2d* pool = malloc(sizeof(AdaptiveAvgPool2d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "AdaptiveAvgPool2d", adaptive_avgpool2d_forward, adaptive_avgpool2d_free) != 0) {
        free(pool); return NULL;
    }
    pool->output_size[0] = output_h;
    pool->output_size[1] = output_w;
    return pool;
}

static Tensor* adaptive_avgpool1d_forward(Module* module, Tensor* input) {
    AdaptiveAvgPool1d* pool = (AdaptiveAvgPool1d*)module;
    if (!input || input->ndim != 3) return NULL;  // [N, C, L]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1], L = input->shape[2];
    int out_l = pool->output_size;

    int out_shape[] = {N, C, out_l};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 3, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < out_l; ol++) {
                int start = (int)floorf((float)ol * L / out_l);
                int end = (int)ceilf((float)(ol + 1) * L / out_l);
                float sum = 0.0f;
                for (int i = start; i < end; i++)
                    sum += in_data[n * C * L + c * L + i];
                out_data[n * C * out_l + c * out_l + ol] = sum / (end - start);
            }
        }
    }
    return output;
}

static void adaptive_avgpool1d_free(Module* module) { free(module); }

AdaptiveAvgPool1d* nn_adaptive_avgpool1d(int output_size) {
    AdaptiveAvgPool1d* pool = malloc(sizeof(AdaptiveAvgPool1d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "AdaptiveAvgPool1d", adaptive_avgpool1d_forward, adaptive_avgpool1d_free) != 0) {
        free(pool); return NULL;
    }
    pool->output_size = output_size;
    return pool;
}

static Tensor* adaptive_maxpool2d_forward(Module* module, Tensor* input) {
    AdaptiveMaxPool2d* pool = (AdaptiveMaxPool2d*)module;
    if (!input || input->ndim != 4) return NULL;  // [N, C, H, W]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1];
    int in_h = input->shape[2], in_w = input->shape[3];
    int out_h = pool->output_size[0], out_w = pool->output_size[1];

    int out_shape[] = {N, C, out_h, out_w};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 4, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                int h_start = (int)floorf((float)oh * in_h / out_h);
                int h_end = (int)ceilf((float)(oh + 1) * in_h / out_h);
                for (int ow = 0; ow < out_w; ow++) {
                    int w_start = (int)floorf((float)ow * in_w / out_w);
                    int w_end = (int)ceilf((float)(ow + 1) * in_w / out_w);
                    float max_val = -FLT_MAX;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            float v = in_data[n*C*in_h*in_w + c*in_h*in_w + h*in_w + w];
                            if (v > max_val) max_val = v;
                        }
                    }
                    out_data[n*C*out_h*out_w + c*out_h*out_w + oh*out_w + ow] = max_val;
                }
            }
        }
    }
    return output;
}

static void adaptive_maxpool2d_free(Module* module) { free(module); }

AdaptiveMaxPool2d* nn_adaptive_maxpool2d(int output_h, int output_w) {
    AdaptiveMaxPool2d* pool = malloc(sizeof(AdaptiveMaxPool2d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "AdaptiveMaxPool2d", adaptive_maxpool2d_forward, adaptive_maxpool2d_free) != 0) {
        free(pool); return NULL;
    }
    pool->output_size[0] = output_h;
    pool->output_size[1] = output_w;
    return pool;
}

static Tensor* adaptive_maxpool1d_forward(Module* module, Tensor* input) {
    AdaptiveMaxPool1d* pool = (AdaptiveMaxPool1d*)module;
    if (!input || input->ndim != 3) return NULL;  // [N, C, L]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data) return NULL;

    int N = input->shape[0], C = input->shape[1], L = input->shape[2];
    int out_l = pool->output_size;

    int out_shape[] = {N, C, out_l};
    TensorConfig config = {.dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 3, &config);
    if (!output) return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < out_l; ol++) {
                int start = (int)floorf((float)ol * L / out_l);
                int end = (int)ceilf((float)(ol + 1) * L / out_l);
                float max_val = -FLT_MAX;
                for (int i = start; i < end; i++) {
                    float v = in_data[n*C*L + c*L + i];
                    if (v > max_val) max_val = v;
                }
                out_data[n*C*out_l + c*out_l + ol] = max_val;
            }
        }
    }
    return output;
}

static void adaptive_maxpool1d_free(Module* module) { free(module); }

AdaptiveMaxPool1d* nn_adaptive_maxpool1d(int output_size) {
    AdaptiveMaxPool1d* pool = malloc(sizeof(AdaptiveMaxPool1d));
    if (!pool) return NULL;
    if (module_init((Module*)pool, "AdaptiveMaxPool1d", adaptive_maxpool1d_forward, adaptive_maxpool1d_free) != 0) {
        free(pool); return NULL;
    }
    pool->output_size = output_size;
    return pool;
}
