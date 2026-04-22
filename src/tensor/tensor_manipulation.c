#include "tensor/tensor.h"
#include "tensor/tensor_views.h"
#include "tensor/tensor_manipulation.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor* tensor_concat(Tensor** tensors, int num_tensors, int dim) {
    return uop_cat(tensors, num_tensors, dim);
}

Tensor* tensor_stack(Tensor** tensors, int num_tensors, int dim) {
    return uop_stack(tensors, num_tensors, dim);
}

Tensor** tensor_split(Tensor* tensor, int num_splits, int dim, int* split_sizes) {
    if (!tensor || num_splits <= 0) {
        LOG_ERROR("tensor_split: invalid input");
        return NULL;
    }

    int normalized_dim = dim;
    if (normalized_dim < 0)
        normalized_dim = tensor->ndim + normalized_dim;
    if (normalized_dim < 0 || normalized_dim >= tensor->ndim) {
        LOG_ERROR("tensor_split: dimension %d out of range", dim);
        return NULL;
    }

    int dim_size = tensor->shape[normalized_dim];

    int* sizes = NULL;
    int need_free = 0;
    if (split_sizes) {
        sizes = split_sizes;
    } else {
        int base_size = dim_size / num_splits;
        int remainder = dim_size % num_splits;
        sizes = malloc((size_t)num_splits * sizeof(int));
        if (!sizes) return NULL;
        need_free = 1;
        for (int i = 0; i < num_splits; i++)
            sizes[i] = base_size + (i < remainder ? 1 : 0);
    }

    Tensor** results = malloc((size_t)num_splits * sizeof(Tensor*));
    if (!results) {
        if (need_free) free(sizes);
        return NULL;
    }

    int offset = 0;
    for (int i = 0; i < num_splits; i++) {
        int* starts = calloc((size_t)tensor->ndim, sizeof(int));
        int* ends   = malloc((size_t)tensor->ndim * sizeof(int));
        int* steps  = malloc((size_t)tensor->ndim * sizeof(int));
        if (!starts || !ends || !steps) {
            free(starts); free(ends); free(steps);
            for (int j = 0; j < i; j++) tensor_free(results[j]);
            free(results);
            if (need_free) free(sizes);
            return NULL;
        }
        for (int d = 0; d < tensor->ndim; d++) {
            starts[d] = (d == normalized_dim) ? offset : 0;
            ends[d]   = (d == normalized_dim) ? offset + sizes[i] : tensor->shape[d];
            steps[d]  = 1;
        }

        SliceParams* sp = malloc(sizeof(SliceParams));
        if (!sp) {
            free(starts); free(ends); free(steps);
            for (int j = 0; j < i; j++) tensor_free(results[j]);
            free(results);
            if (need_free) free(sizes);
            return NULL;
        }
        sp->start    = starts;
        sp->end      = ends;
        sp->step     = steps;
        sp->num_dims = tensor->ndim;

        results[i] = uop_slice(tensor, sp);
        offset += sizes[i];
    }

    if (need_free) free(sizes);
    return results;
}

Tensor* tensor_gather(Tensor* input, Tensor* indices, int dim) {
    if (!input || !indices) {
        LOG_ERROR("tensor_gather: invalid input");
        return NULL;
    }

    int normalized_dim = dim;
    if (normalized_dim < 0)
        normalized_dim = input->ndim + normalized_dim;
    if (normalized_dim < 0 || normalized_dim >= input->ndim) {
        LOG_ERROR("tensor_gather: dimension %d out of range", dim);
        return NULL;
    }

    /* uop_gather requires 1D indices; for multi-dim indices use eager fallback */
    if (indices->ndim == 1) {
        return uop_gather(input, indices, normalized_dim);
    }

    if (indices->ndim != input->ndim) {
        LOG_ERROR("tensor_gather: indices must have same number of dimensions as input");
        return NULL;
    }
    for (int d = 0; d < input->ndim; d++) {
        if (d != normalized_dim && indices->shape[d] != input->shape[d]) {
            LOG_ERROR("tensor_gather: indices shape must match input shape except in gather dimension");
            return NULL;
        }
    }

    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(indices->shape, indices->ndim, &config);
    if (!output) return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);
    int* idx_data   = (int*)tensor_data_ptr(indices);

    size_t* input_strides  = compute_contiguous_strides(input->shape, input->ndim);
    size_t* output_strides = compute_contiguous_strides(output->shape, output->ndim);
    if (!input_strides || !output_strides) {
        tensor_free(output);
        free(input_strides);
        free(output_strides);
        return NULL;
    }

    for (size_t i = 0; i < output->numel; i++) {
        int* out_indices = malloc((size_t)output->ndim * sizeof(int));
        size_t idx       = i;
        for (int d = output->ndim - 1; d >= 0; d--) {
            out_indices[d] = (int)(idx % (size_t)output->shape[d]);
            idx /= (size_t)output->shape[d];
        }

        size_t idx_offset = 0;
        for (int d = 0; d < output->ndim; d++)
            idx_offset += (size_t)out_indices[d] * output_strides[d];
        int gather_idx = idx_data[idx_offset];

        if (gather_idx < 0 || gather_idx >= input->shape[normalized_dim]) {
            free(out_indices);
            free(input_strides);
            free(output_strides);
            tensor_free(output);
            LOG_ERROR("tensor_gather: index %d out of range [0, %d)", gather_idx,
                      input->shape[normalized_dim]);
            return NULL;
        }

        int* in_indices = malloc((size_t)input->ndim * sizeof(int));
        for (int d = 0; d < input->ndim; d++)
            in_indices[d] = (d == normalized_dim) ? gather_idx : out_indices[d];

        size_t in_offset = 0;
        for (int d = 0; d < input->ndim; d++)
            in_offset += (size_t)in_indices[d] * input_strides[d];

        out_data[i] = in_data[in_offset];

        free(out_indices);
        free(in_indices);
    }

    free(input_strides);
    free(output_strides);
    return output;
}

Tensor* tensor_scatter(Tensor* input, int dim, Tensor* index, Tensor* src) {
    return uop_scatter(input, dim, index, src);
}
