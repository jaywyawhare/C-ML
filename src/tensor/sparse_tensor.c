#include "tensor/sparse_tensor.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

SparseCOOData* sparse_coo_tensor(Tensor* indices, Tensor* values,
                                  const int* dense_shape, int dense_ndim) {
    if (!indices || !values || !dense_shape || dense_ndim <= 0) {
        LOG_ERROR("sparse_coo_tensor: invalid arguments");
        return NULL;
    }

    tensor_ensure_executed(indices);
    tensor_ensure_executed(values);

    if (indices->ndim != 2) {
        LOG_ERROR("sparse_coo_tensor: indices must be 2D [nnz, ndim], got %dD", indices->ndim);
        return NULL;
    }

    if (values->ndim != 1) {
        LOG_ERROR("sparse_coo_tensor: values must be 1D [nnz], got %dD", values->ndim);
        return NULL;
    }

    int nnz  = indices->shape[0];
    int ndim = indices->shape[1];

    if (nnz != values->shape[0]) {
        LOG_ERROR("sparse_coo_tensor: indices nnz (%d) != values nnz (%d)",
                  nnz, values->shape[0]);
        return NULL;
    }

    if (ndim != dense_ndim) {
        LOG_ERROR("sparse_coo_tensor: indices ndim (%d) != dense_ndim (%d)", ndim, dense_ndim);
        return NULL;
    }

    SparseCOOData* sparse = calloc(1, sizeof(SparseCOOData));
    if (!sparse) {
        LOG_ERROR("sparse_coo_tensor: failed to allocate SparseCOOData");
        return NULL;
    }

    /* Clone indices and values tensors so the sparse tensor owns its own copies */
    sparse->indices = tensor_clone(indices);
    sparse->values  = tensor_clone(values);

    if (!sparse->indices || !sparse->values) {
        LOG_ERROR("sparse_coo_tensor: failed to clone indices or values");
        if (sparse->indices) tensor_free(sparse->indices);
        if (sparse->values) tensor_free(sparse->values);
        free(sparse);
        return NULL;
    }

    sparse->nnz        = nnz;
    sparse->dense_ndim = dense_ndim;

    sparse->dense_shape = calloc(dense_ndim, sizeof(int));
    if (!sparse->dense_shape) {
        LOG_ERROR("sparse_coo_tensor: failed to allocate dense_shape");
        tensor_free(sparse->indices);
        tensor_free(sparse->values);
        free(sparse);
        return NULL;
    }
    memcpy(sparse->dense_shape, dense_shape, dense_ndim * sizeof(int));

    LOG_DEBUG("Created sparse COO tensor: nnz=%d, dense_ndim=%d", nnz, dense_ndim);
    return sparse;
}

SparseCOOData* sparse_from_dense(Tensor* dense) {
    if (!dense) {
        LOG_ERROR("sparse_from_dense: NULL input");
        return NULL;
    }

    tensor_ensure_executed(dense);

    float* data = (float*)dense->data;
    if (!data) {
        LOG_ERROR("sparse_from_dense: NULL data pointer");
        return NULL;
    }

    /* First pass: count non-zeros */
    int nnz = 0;
    for (size_t i = 0; i < dense->numel; i++) {
        if (data[i] != 0.0f) {
            nnz++;
        }
    }

    if (nnz == 0) {
        LOG_DEBUG("sparse_from_dense: input is all zeros, creating empty sparse tensor");
    }

    int ndim = dense->ndim;

    /* Allocate indices [nnz, ndim] as INT32 and values [nnz] as same dtype */
    int idx_shape[] = {nnz, ndim};
    TensorConfig idx_config = (TensorConfig){
        .dtype = DTYPE_INT32, .device = dense->device, .has_dtype = true, .has_device = true};

    Tensor* indices = NULL;
    if (nnz > 0) {
        indices = tensor_empty(idx_shape, 2, &idx_config);
    } else {
        /* For zero nnz, create a minimal tensor */
        int zero_shape[] = {0, ndim};
        indices = tensor_empty(zero_shape, 2, &idx_config);
    }

    int val_shape[] = {nnz};
    TensorConfig val_config = (TensorConfig){
        .dtype = dense->dtype, .device = dense->device, .has_dtype = true, .has_device = true};
    Tensor* values = NULL;
    if (nnz > 0) {
        values = tensor_empty(val_shape, 1, &val_config);
    } else {
        int zero_shape[] = {0};
        values = tensor_empty(zero_shape, 1, &val_config);
    }

    if (!indices || !values) {
        if (indices) tensor_free(indices);
        if (values) tensor_free(values);
        LOG_ERROR("sparse_from_dense: failed to allocate index/value tensors");
        return NULL;
    }

    if (nnz > 0) {
        int32_t* idx_data = (int32_t*)indices->data;
        float* val_data   = (float*)values->data;

        if (!idx_data || !val_data) {
            tensor_free(indices);
            tensor_free(values);
            LOG_ERROR("sparse_from_dense: NULL data after allocation");
            return NULL;
        }

        /* Second pass: fill indices and values */
        int count = 0;
        for (size_t flat = 0; flat < dense->numel; flat++) {
            if (data[flat] != 0.0f) {
                /* Convert flat index to multi-dimensional coordinates */
                size_t remainder = flat;
                for (int d = ndim - 1; d >= 0; d--) {
                    idx_data[count * ndim + d] = (int32_t)(remainder % dense->shape[d]);
                    remainder /= dense->shape[d];
                }
                val_data[count] = data[flat];
                count++;
            }
        }
    }

    SparseCOOData* sparse = calloc(1, sizeof(SparseCOOData));
    if (!sparse) {
        tensor_free(indices);
        tensor_free(values);
        LOG_ERROR("sparse_from_dense: failed to allocate SparseCOOData");
        return NULL;
    }

    sparse->indices    = indices;
    sparse->values     = values;
    sparse->nnz        = nnz;
    sparse->dense_ndim = ndim;

    sparse->dense_shape = calloc(ndim, sizeof(int));
    if (!sparse->dense_shape) {
        tensor_free(indices);
        tensor_free(values);
        free(sparse);
        LOG_ERROR("sparse_from_dense: failed to allocate dense_shape");
        return NULL;
    }
    memcpy(sparse->dense_shape, dense->shape, ndim * sizeof(int));

    LOG_DEBUG("sparse_from_dense: converted to sparse with nnz=%d out of %zu elements",
              nnz, dense->numel);
    return sparse;
}

Tensor* sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config) {
    if (!sparse) {
        LOG_ERROR("sparse_to_dense: NULL input");
        return NULL;
    }

    /* Create a zero-filled tensor with the dense shape */
    Tensor* output = tensor_zeros(sparse->dense_shape, sparse->dense_ndim, config);
    if (!output) {
        LOG_ERROR("sparse_to_dense: failed to create output tensor");
        return NULL;
    }

    if (sparse->nnz == 0) {
        return output;
    }

    tensor_ensure_executed(sparse->indices);
    tensor_ensure_executed(sparse->values);

    int32_t* idx_data = (int32_t*)sparse->indices->data;
    float* val_data   = (float*)sparse->values->data;
    float* out_data   = (float*)output->data;

    if (!idx_data || !val_data || !out_data) {
        LOG_ERROR("sparse_to_dense: NULL data pointers");
        tensor_free(output);
        return NULL;
    }

    int ndim = sparse->dense_ndim;

    /* Place each non-zero value at its coordinate in the dense tensor */
    for (int i = 0; i < sparse->nnz; i++) {
        /* Compute flat index from multi-dimensional coordinates */
        size_t flat = 0;
        size_t stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            flat += (size_t)idx_data[i * ndim + d] * stride;
            stride *= sparse->dense_shape[d];
        }

        if (flat < output->numel) {
            out_data[flat] += val_data[i];
        }
    }

    return output;
}

Tensor* sparse_matmul(SparseCOOData* sparse, Tensor* dense) {
    if (!sparse || !dense) {
        LOG_ERROR("sparse_matmul: NULL argument");
        return NULL;
    }

    /* Validate sparse is 2D */
    if (sparse->dense_ndim != 2) {
        LOG_ERROR("sparse_matmul: sparse tensor must be 2D, got %dD", sparse->dense_ndim);
        return NULL;
    }

    tensor_ensure_executed(dense);

    if (dense->ndim != 2) {
        LOG_ERROR("sparse_matmul: dense tensor must be 2D, got %dD", dense->ndim);
        return NULL;
    }

    int M = sparse->dense_shape[0];
    int K = sparse->dense_shape[1];
    int N = dense->shape[1];

    if (K != dense->shape[0]) {
        LOG_ERROR("sparse_matmul: dimension mismatch - sparse cols (%d) != dense rows (%d)",
                  K, dense->shape[0]);
        return NULL;
    }

    /* Create output [M, N] zero tensor */
    int out_shape[] = {M, N};
    TensorConfig config = (TensorConfig){
        .dtype = dense->dtype, .device = dense->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 2, &config);
    if (!output) {
        LOG_ERROR("sparse_matmul: failed to create output tensor");
        return NULL;
    }

    if (sparse->nnz == 0) {
        return output;
    }

    tensor_ensure_executed(sparse->indices);
    tensor_ensure_executed(sparse->values);

    int32_t* idx_data   = (int32_t*)sparse->indices->data;
    float* val_data     = (float*)sparse->values->data;
    float* dense_data   = (float*)dense->data;
    float* out_data     = (float*)output->data;

    if (!idx_data || !val_data || !dense_data || !out_data) {
        LOG_ERROR("sparse_matmul: NULL data pointers");
        tensor_free(output);
        return NULL;
    }

    /* SpMM: for each non-zero A[row, col] = v, accumulate v * B[col, :] into C[row, :] */
    for (int i = 0; i < sparse->nnz; i++) {
        int row = idx_data[i * 2 + 0];
        int col = idx_data[i * 2 + 1];
        float v = val_data[i];

        if (row < 0 || row >= M || col < 0 || col >= K)
            continue;

        for (int n = 0; n < N; n++) {
            out_data[row * N + n] += v * dense_data[col * N + n];
        }
    }

    LOG_DEBUG("sparse_matmul: [%d,%d] sparse x [%d,%d] dense -> [%d,%d]",
              M, K, K, N, M, N);
    return output;
}

SparseCOOData* sparse_coalesce(SparseCOOData* sparse) {
    if (!sparse) {
        LOG_ERROR("sparse_coalesce: NULL input");
        return NULL;
    }

    LOG_DEBUG("sparse_coalesce: coalescing %d entries", sparse->nnz);

    /*
     * Simple approach: convert to dense via sparse_to_dense, then convert
     * back with sparse_from_dense. The dense round-trip naturally sums
     * duplicate indices and produces sorted, unique entries.
     */
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* dense_tmp = sparse_to_dense(sparse, &config);
    if (!dense_tmp) {
        LOG_ERROR("sparse_coalesce: failed to convert to dense");
        return NULL;
    }

    SparseCOOData* coalesced = sparse_from_dense(dense_tmp);
    tensor_free(dense_tmp);

    if (!coalesced) {
        LOG_ERROR("sparse_coalesce: failed to convert back to sparse");
        return NULL;
    }

    LOG_DEBUG("sparse_coalesce: %d entries -> %d unique entries", sparse->nnz, coalesced->nnz);
    return coalesced;
}

void sparse_free(SparseCOOData* sparse) {
    if (!sparse)
        return;

    if (sparse->indices) {
        tensor_free(sparse->indices);
        sparse->indices = NULL;
    }

    if (sparse->values) {
        tensor_free(sparse->values);
        sparse->values = NULL;
    }

    if (sparse->dense_shape) {
        free(sparse->dense_shape);
        sparse->dense_shape = NULL;
    }

    free(sparse);
}
