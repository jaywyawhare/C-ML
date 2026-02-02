/**
 * @file tensor_manipulation.c
 * @brief Tensor manipulation operations: concat, stack, split, gather, scatter
 */

#include "tensor/tensor.h"
#include "tensor/tensor_views.h"
#include "tensor/tensor_manipulation.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Concatenate tensors along specified dimension
Tensor* tensor_concat(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) {
        LOG_ERROR("tensor_concat: invalid input");
        return NULL;
    }

    if (num_tensors == 1) {
        return tensor_clone(tensors[0]);
    }

    // Normalize dimension
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = tensors[0]->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= tensors[0]->ndim) {
        LOG_ERROR("tensor_concat: dimension %d out of range", dim);
        return NULL;
    }

    // Check all tensors have compatible shapes
    for (int i = 1; i < num_tensors; i++) {
        if (!tensors[i] || tensors[i]->ndim != tensors[0]->ndim) {
            LOG_ERROR("tensor_concat: all tensors must have same number of dimensions");
            return NULL;
        }
        for (int d = 0; d < tensors[0]->ndim; d++) {
            if (d != normalized_dim && tensors[i]->shape[d] != tensors[0]->shape[d]) {
                LOG_ERROR(
                    "tensor_concat: all tensors must have same shape except in concat dimension");
                return NULL;
            }
        }
    }

    // Calculate output shape
    int* out_shape = tensor_shape_copy(tensors[0]->shape, tensors[0]->ndim);
    if (!out_shape)
        return NULL;

    int total_size = 0;
    for (int i = 0; i < num_tensors; i++) {
        total_size += tensors[i]->shape[normalized_dim];
    }
    out_shape[normalized_dim] = total_size;

    // Create output tensor
    TensorConfig config = (TensorConfig){.dtype      = tensors[0]->dtype,
                                         .device     = tensors[0]->device,
                                         .has_dtype  = true,
                                         .has_device = true};
    Tensor* output      = tensor_empty(out_shape, tensors[0]->ndim, &config);
    free(out_shape);
    if (!output)
        return NULL;

    // Copy data from each tensor
    float* out_data = (float*)tensor_data_ptr(output);
    int offset      = 0;

    for (int t = 0; t < num_tensors; t++) {
        Tensor* tensor = tensors[t];
        float* in_data = (float*)tensor_data_ptr(tensor);

        // Calculate size of each slice along concat dimension
        size_t slice_size = 1;
        for (int d = normalized_dim + 1; d < tensor->ndim; d++) {
            slice_size *= (size_t)tensor->shape[d];
        }

        size_t num_slices = 1;
        for (int d = 0; d < normalized_dim; d++) {
            num_slices *= (size_t)tensor->shape[d];
        }

        size_t slice_elements = slice_size * (size_t)tensor->shape[normalized_dim];

        // Copy each slice
        for (size_t s = 0; s < num_slices; s++) {
            size_t src_offset = s * slice_elements;
            size_t dst_offset = s * (slice_size * (size_t)total_size) + (size_t)offset * slice_size;
            memcpy(out_data + dst_offset, in_data + src_offset, slice_elements * sizeof(float));
        }

        offset += tensor->shape[normalized_dim];
    }

    return output;
}

// Stack tensors along new dimension
Tensor* tensor_stack(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) {
        LOG_ERROR("tensor_stack: invalid input");
        return NULL;
    }

    // Normalize dimension
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = tensors[0]->ndim + 1 + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim > tensors[0]->ndim) {
        LOG_ERROR("tensor_stack: dimension %d out of range", dim);
        return NULL;
    }

    // Check all tensors have same shape
    for (int i = 1; i < num_tensors; i++) {
        if (!tensors[i] || tensors[i]->ndim != tensors[0]->ndim) {
            LOG_ERROR("tensor_stack: all tensors must have same number of dimensions");
            return NULL;
        }
        for (int d = 0; d < tensors[0]->ndim; d++) {
            if (tensors[i]->shape[d] != tensors[0]->shape[d]) {
                LOG_ERROR("tensor_stack: all tensors must have same shape");
                return NULL;
            }
        }
    }

    // Calculate output shape (insert new dimension)
    int out_ndim   = tensors[0]->ndim + 1;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    for (int d = 0; d < normalized_dim; d++) {
        out_shape[d] = tensors[0]->shape[d];
    }
    out_shape[normalized_dim] = num_tensors;
    for (int d = normalized_dim; d < tensors[0]->ndim; d++) {
        out_shape[d + 1] = tensors[0]->shape[d];
    }

    // Create output tensor
    TensorConfig config = (TensorConfig){.dtype      = tensors[0]->dtype,
                                         .device     = tensors[0]->device,
                                         .has_dtype  = true,
                                         .has_device = true};
    Tensor* output      = tensor_empty(out_shape, out_ndim, &config);
    free(out_shape);
    if (!output)
        return NULL;

    // Copy data from each tensor
    float* out_data    = (float*)tensor_data_ptr(output);
    size_t tensor_size = tensors[0]->numel;

    for (int t = 0; t < num_tensors; t++) {
        float* in_data = (float*)tensor_data_ptr(tensors[t]);
        size_t offset  = (size_t)t * tensor_size;
        memcpy(out_data + offset, in_data, tensor_size * sizeof(float));
    }

    return output;
}

// Split tensor into multiple tensors
Tensor** tensor_split(Tensor* tensor, int num_splits, int dim, int* split_sizes) {
    if (!tensor || num_splits <= 0) {
        LOG_ERROR("tensor_split: invalid input");
        return NULL;
    }

    // Normalize dimension
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = tensor->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= tensor->ndim) {
        LOG_ERROR("tensor_split: dimension %d out of range", dim);
        return NULL;
    }

    int dim_size = tensor->shape[normalized_dim];

    // Allocate result array
    Tensor** results = malloc((size_t)num_splits * sizeof(Tensor*));
    if (!results)
        return NULL;

    // Calculate split sizes
    int* sizes = NULL;
    if (split_sizes) {
        sizes = split_sizes;
    } else {
        // Equal splits
        int base_size = dim_size / num_splits;
        int remainder = dim_size % num_splits;
        sizes         = malloc((size_t)num_splits * sizeof(int));
        if (!sizes) {
            free(results);
            return NULL;
        }
        for (int i = 0; i < num_splits; i++) {
            sizes[i] = base_size + (i < remainder ? 1 : 0);
        }
    }

    // Create each split tensor
    int offset = 0;
    for (int i = 0; i < num_splits; i++) {
        // Calculate output shape
        int* out_shape = tensor_shape_copy(tensor->shape, tensor->ndim);
        if (!out_shape) {
            for (int j = 0; j < i; j++)
                tensor_free(results[j]);
            free(results);
            if (!split_sizes)
                free(sizes);
            return NULL;
        }
        out_shape[normalized_dim] = sizes[i];

        TensorConfig config = (TensorConfig){.dtype      = tensor->dtype,
                                             .device     = tensor->device,
                                             .has_dtype  = true,
                                             .has_device = true};
        Tensor* split       = tensor_empty(out_shape, tensor->ndim, &config);
        free(out_shape);
        if (!split) {
            for (int j = 0; j < i; j++)
                tensor_free(results[j]);
            free(results);
            if (!split_sizes)
                free(sizes);
            return NULL;
        }

        // Copy data
        float* split_data  = (float*)tensor_data_ptr(split);
        float* tensor_data = (float*)tensor_data_ptr(tensor);

        // Calculate slice size
        size_t slice_size = 1;
        for (int d = normalized_dim + 1; d < tensor->ndim; d++) {
            slice_size *= (size_t)tensor->shape[d];
        }

        size_t num_slices = 1;
        for (int d = 0; d < normalized_dim; d++) {
            num_slices *= (size_t)tensor->shape[d];
        }

        size_t slice_elements = slice_size * (size_t)sizes[i];
        size_t src_offset     = (size_t)offset * slice_size;

        for (size_t s = 0; s < num_slices; s++) {
            memcpy(split_data + s * slice_elements,
                   tensor_data + s * (slice_size * (size_t)dim_size) + src_offset,
                   slice_elements * sizeof(float));
        }

        results[i] = split;
        offset += sizes[i];
    }

    if (!split_sizes)
        free(sizes);
    return results;
}

// Gather values from tensor using indices
Tensor* tensor_gather(Tensor* input, Tensor* indices, int dim) {
    if (!input || !indices) {
        LOG_ERROR("tensor_gather: invalid input");
        return NULL;
    }

    // Normalize dimension
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = input->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= input->ndim) {
        LOG_ERROR("tensor_gather: dimension %d out of range", dim);
        return NULL;
    }

    // Check indices shape matches input shape except in gather dimension
    if (indices->ndim != input->ndim) {
        LOG_ERROR("tensor_gather: indices must have same number of dimensions as input");
        return NULL;
    }

    for (int d = 0; d < input->ndim; d++) {
        if (d != normalized_dim && indices->shape[d] != input->shape[d]) {
            LOG_ERROR(
                "tensor_gather: indices shape must match input shape except in gather dimension");
            return NULL;
        }
    }

    // Create output tensor with indices shape
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(indices->shape, indices->ndim, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);
    int* idx_data   = (int*)tensor_data_ptr(indices);

    // Calculate strides
    size_t* input_strides  = compute_contiguous_strides(input->shape, input->ndim);
    size_t* output_strides = compute_contiguous_strides(output->shape, output->ndim);
    if (!input_strides || !output_strides) {
        tensor_free(output);
        if (input_strides)
            free(input_strides);
        if (output_strides)
            free(output_strides);
        return NULL;
    }

    // Gather values
    for (size_t i = 0; i < output->numel; i++) {
        // Compute multi-dimensional indices for output
        int* out_indices = malloc((size_t)output->ndim * sizeof(int));
        size_t idx       = i;
        for (int d = output->ndim - 1; d >= 0; d--) {
            out_indices[d] = (int)(idx % (size_t)output->shape[d]);
            idx /= (size_t)output->shape[d];
        }

        // Get index value from indices tensor
        size_t idx_offset = 0;
        for (int d = 0; d < output->ndim; d++) {
            idx_offset += (size_t)out_indices[d] * output_strides[d];
        }
        int gather_idx = idx_data[idx_offset];

        // Bounds check
        if (gather_idx < 0 || gather_idx >= input->shape[normalized_dim]) {
            free(out_indices);
            free(input_strides);
            free(output_strides);
            tensor_free(output);
            LOG_ERROR("tensor_gather: index %d out of range [0, %d)", gather_idx,
                      input->shape[normalized_dim]);
            return NULL;
        }

        // Compute input indices
        int* in_indices = malloc((size_t)input->ndim * sizeof(int));
        for (int d = 0; d < input->ndim; d++) {
            in_indices[d] = (d == normalized_dim) ? gather_idx : out_indices[d];
        }

        // Compute input offset
        size_t in_offset = 0;
        for (int d = 0; d < input->ndim; d++) {
            in_offset += (size_t)in_indices[d] * input_strides[d];
        }

        out_data[i] = in_data[in_offset];

        free(out_indices);
        free(in_indices);
    }

    free(input_strides);
    free(output_strides);
    return output;
}

// Scatter values into tensor at specified indices
Tensor* tensor_scatter(Tensor* input, int dim, Tensor* index, Tensor* src) {
    if (!input || !index || !src) {
        LOG_ERROR("tensor_scatter: invalid input");
        return NULL;
    }

    // Normalize dimension
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = input->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= input->ndim) {
        LOG_ERROR("tensor_scatter: dimension %d out of range", dim);
        return NULL;
    }

    // Check shapes match
    if (index->ndim != src->ndim || index->ndim != input->ndim) {
        LOG_ERROR("tensor_scatter: index, src, and input must have same number of dimensions");
        return NULL;
    }

    for (int d = 0; d < input->ndim; d++) {
        if (d != normalized_dim &&
            (index->shape[d] != input->shape[d] || src->shape[d] != input->shape[d])) {
            LOG_ERROR("tensor_scatter: shapes must match except in scatter dimension");
            return NULL;
        }
    }

    // Clone input tensor
    Tensor* output = tensor_clone(input);
    if (!output)
        return NULL;

    float* out_data = (float*)tensor_data_ptr(output);
    float* src_data = (float*)tensor_data_ptr(src);
    int* idx_data   = (int*)tensor_data_ptr(index);

    // Calculate strides
    size_t* output_strides = compute_contiguous_strides(output->shape, output->ndim);
    size_t* src_strides    = compute_contiguous_strides(src->shape, src->ndim);
    if (!output_strides || !src_strides) {
        tensor_free(output);
        if (output_strides)
            free(output_strides);
        if (src_strides)
            free(src_strides);
        return NULL;
    }

    // Scatter values
    for (size_t i = 0; i < src->numel; i++) {
        // Compute multi-dimensional indices
        int* src_indices = malloc((size_t)src->ndim * sizeof(int));
        size_t idx       = i;
        for (int d = src->ndim - 1; d >= 0; d--) {
            src_indices[d] = (int)(idx % (size_t)src->shape[d]);
            idx /= (size_t)src->shape[d];
        }

        // Get index value
        size_t idx_offset = 0;
        for (int d = 0; d < src->ndim; d++) {
            idx_offset += (size_t)src_indices[d] * src_strides[d];
        }
        int scatter_idx = idx_data[idx_offset];

        // Bounds check
        if (scatter_idx < 0 || scatter_idx >= output->shape[normalized_dim]) {
            free(src_indices);
            free(output_strides);
            free(src_strides);
            tensor_free(output);
            LOG_ERROR("tensor_scatter: index %d out of range [0, %d)", scatter_idx,
                      output->shape[normalized_dim]);
            return NULL;
        }

        // Compute output indices
        int* out_indices = malloc((size_t)output->ndim * sizeof(int));
        for (int d = 0; d < output->ndim; d++) {
            out_indices[d] = (d == normalized_dim) ? scatter_idx : src_indices[d];
        }

        // Compute output offset
        size_t out_offset = 0;
        for (int d = 0; d < output->ndim; d++) {
            out_offset += (size_t)out_indices[d] * output_strides[d];
        }

        out_data[out_offset] = src_data[i];

        free(src_indices);
        free(out_indices);
    }

    free(output_strides);
    free(src_strides);
    return output;
}
