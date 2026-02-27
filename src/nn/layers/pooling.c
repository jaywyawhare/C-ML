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

// Helper: Create strided view for a pooling window
static Tensor* create_pooling_window_view(Tensor* input, int b, int c, int oh, int ow, int kernel_h,
                                          int kernel_w, int stride_h, int stride_w, int padding_h,
                                          int padding_w, int dilation_h, int dilation_w,
                                          int in_height, int in_width) {
    // Calculate window start position
    int start_h = oh * stride_h - padding_h;
    int start_w = ow * stride_w - padding_w;

    // Create shape for window: [kernel_h, kernel_w]
    int window_shape[] = {kernel_h, kernel_w};

    // Calculate strides for the window view
    // The window strides should step by dilation in the input
    size_t* window_strides = malloc(2 * sizeof(size_t));
    if (!window_strides)
        return NULL;

    // Get input strides
    size_t input_h_stride = input->strides ? input->strides[2] : (size_t)(in_width);
    size_t input_w_stride = input->strides ? input->strides[3] : 1;

    // Window strides account for dilation
    window_strides[0] = (size_t)input_h_stride * (size_t)dilation_h;
    window_strides[1] = (size_t)input_w_stride * (size_t)dilation_w;

    // Calculate storage offset for this window (in bytes)
    size_t storage_offset = 0;
    if (input->strides) {
        // Use input's stride structure - tensor_compute_offset returns element offset
        int indices[]      = {b, c, start_h, start_w};
        size_t elem_offset = tensor_compute_offset(input, indices);
        storage_offset     = elem_offset * cml_dtype_size(input->dtype);
    } else {
        // Contiguous tensor
        size_t elem_offset =
            ((size_t)b * (size_t)input->shape[1] * (size_t)in_height * (size_t)in_width +
             (size_t)c * (size_t)in_height * (size_t)in_width + (size_t)start_h * (size_t)in_width +
             (size_t)start_w);
        storage_offset = elem_offset * cml_dtype_size(input->dtype);
    }

    // Create strided view
    Tensor* window_view = tensor_as_strided(input, window_shape, 2, window_strides, storage_offset);

    free(window_strides);
    return window_view;
}

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
    int output_shape[]  = {batch, channels, out_height, out_width};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(output_shape, 4, &config);
    if (!output)
        return NULL;

    // Use stride-based approach: create views for each pooling window and apply max reduction
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

                    // Apply max reduction over the window using uop_max_reduce
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

                    // Copy result to output
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
    int output_shape[]  = {batch, channels, out_height, out_width};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(output_shape, 4, &config);
    if (!output)
        return NULL;

    // Use stride-based approach: create views for each pooling window and apply mean reduction
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

                    // Apply mean reduction over the window using uop_mean
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

                    // Copy result to output
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

    LOG_DEBUG("Created AvgPool2d layer: kernel=%d, stride=%d, padding=%d", kernel_size,
              pool->stride[0], padding);

    return pool;
}
