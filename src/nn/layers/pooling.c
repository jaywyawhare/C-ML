/**
 * @file pooling.c
 * @brief Pooling layers implementation
 */

#include "nn/layers/pooling.h"
#include "nn/module.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

static Tensor* maxpool2d_forward(Module* module, Tensor* input) {
    MaxPool2d* pool = (MaxPool2d*)module;

    if (!pool || !input)
        return NULL;

    // Input shape: [batch, channels, height, width]
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

    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Handle ceil_mode
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

    // Create output tensor [batch, channels, out_h, out_w]
    int output_shape[] = {batch, channels, out_height, out_width};
    Tensor* output     = tensor_empty(output_shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* input_data  = (float*)tensor_data_ptr(input);
    float* output_data = (float*)tensor_data_ptr(output);

    if (!input_data || !output_data) {
        tensor_free(output);
        return NULL;
    }

    // Perform max pooling
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float max_val = -INFINITY;

                    // Find max value in pooling window
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // Calculate input position with padding and dilation
                            int ih = oh * stride_h - padding_h + kh * dilation_h;
                            int iw = ow * stride_w - padding_w + kw * dilation_w;

                            // Skip if out of bounds (padding)
                            if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
                                continue;
                            }

                            // Input indices: [b, c, ih, iw]
                            int input_indices[] = {b, c, ih, iw};
                            size_t input_offset = tensor_compute_offset(input, input_indices);
                            float val           = input_data[input_offset];

                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }

                    // Output indices: [b, c, oh, ow]
                    int output_indices[] = {b, c, oh, ow};
                    size_t output_offset = tensor_compute_offset(output, output_indices);

                    // If no valid values found (all padding), use 0
                    if (max_val == -INFINITY) {
                        max_val = 0.0f;
                    }

                    output_data[output_offset] = max_val;
                }
            }
        }
    }

    return output;
}

static void maxpool2d_free(Module* module) { CM_FREE(module); }

MaxPool2d* nn_maxpool2d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool2d* pool = CM_MALLOC(sizeof(MaxPool2d));
    if (!pool)
        return NULL;

    if (module_init((Module*)pool, "MaxPool2d", maxpool2d_forward, maxpool2d_free) != 0) {
        CM_FREE(pool);
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

    LOG_DEBUG("Created MaxPool2d layer: kernel=%d, stride=%d, padding=%d", kernel_size,
              pool->stride[0], padding);

    return pool;
}

static Tensor* avgpool2d_forward(Module* module, Tensor* input) {
    AvgPool2d* pool = (AvgPool2d*)module;

    if (!pool || !input)
        return NULL;

    // Input shape: [batch, channels, height, width]
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

    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    // Handle ceil_mode
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

    // Create output tensor [batch, channels, out_h, out_w]
    int output_shape[] = {batch, channels, out_height, out_width};
    Tensor* output     = tensor_empty(output_shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* input_data  = (float*)tensor_data_ptr(input);
    float* output_data = (float*)tensor_data_ptr(output);

    if (!input_data || !output_data) {
        tensor_free(output);
        return NULL;
    }

    // Perform average pooling
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    int count = 0;

                    // Find average value in pooling window
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // Calculate input position with padding
                            int ih = oh * stride_h - padding_h + kh;
                            int iw = ow * stride_w - padding_w + kw;

                            // Skip if out of bounds (padding)
                            if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
                                if (pool->count_include_pad) {
                                    count++;
                                }
                                continue;
                            }

                            // Input indices: [b, c, ih, iw]
                            int input_indices[] = {b, c, ih, iw};
                            size_t input_offset = tensor_compute_offset(input, input_indices);
                            sum += input_data[input_offset];
                            count++;
                        }
                    }

                    // Output indices: [b, c, oh, ow]
                    int output_indices[] = {b, c, oh, ow};
                    size_t output_offset = tensor_compute_offset(output, output_indices);

                    // Compute average
                    float avg                  = count > 0 ? sum / count : 0.0f;
                    output_data[output_offset] = avg;
                }
            }
        }
    }

    return output;
}

static void avgpool2d_free(Module* module) { CM_FREE(module); }

AvgPool2d* nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad) {
    AvgPool2d* pool = CM_MALLOC(sizeof(AvgPool2d));
    if (!pool)
        return NULL;

    if (module_init((Module*)pool, "AvgPool2d", avgpool2d_forward, avgpool2d_free) != 0) {
        CM_FREE(pool);
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

    LOG_DEBUG("Created AvgPool2d layer: kernel=%d, stride=%d, padding=%d", kernel_size,
              pool->stride[0], padding);

    return pool;
}
