/**
 * @file tensor_parallel.c
 * @brief Tensor parallelism implementation
 *
 * Column-parallel and row-parallel linear layers following the Megatron-LM
 * partitioning strategy.  Forward passes perform a manual matmul
 * (output = input @ weight^T) on CPU using float32.  The all-reduce is
 * simulated via a simple element-wise sum (single-process mode).
 */

#include "distributed/tensor_parallel.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Internal helpers
 * ======================================================================== */

/**
 * Perform output[M, N] = input[M, K] @ weight[N, K]^T
 *
 * weight is stored row-major as [N, K], so weight^T column j is weight row j.
 * output[i][j] = sum_k input[i][k] * weight[j][k]
 */
static void matmul_weight_transposed(const float* input, int M, int K,
                                     const float* weight, int N,
                                     float* output)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += input[i * K + k] * weight[j * K + k];
            }
            output[i * N + j] = sum;
        }
    }
}

/* ========================================================================
 * Weight sharding utility
 * ======================================================================== */

Tensor* cml_tp_shard_weight(Tensor* weight, int dim, int tp_size, int tp_rank)
{
    if (!weight) {
        LOG_ERROR("cml_tp_shard_weight: weight is NULL");
        return NULL;
    }
    if (weight->ndim != 2) {
        LOG_ERROR("cml_tp_shard_weight: expected 2-D weight, got ndim=%d", weight->ndim);
        return NULL;
    }
    if (dim != 0 && dim != 1) {
        LOG_ERROR("cml_tp_shard_weight: dim must be 0 or 1, got %d", dim);
        return NULL;
    }
    if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size) {
        LOG_ERROR("cml_tp_shard_weight: invalid tp_size=%d or tp_rank=%d", tp_size, tp_rank);
        return NULL;
    }

    int rows = weight->shape[0];
    int cols = weight->shape[1];

    if (dim == 0 && rows % tp_size != 0) {
        LOG_ERROR("cml_tp_shard_weight: rows (%d) not divisible by tp_size (%d)", rows, tp_size);
        return NULL;
    }
    if (dim == 1 && cols % tp_size != 0) {
        LOG_ERROR("cml_tp_shard_weight: cols (%d) not divisible by tp_size (%d)", cols, tp_size);
        return NULL;
    }

    tensor_ensure_executed(weight);
    const float* src = (const float*)tensor_data_ptr(weight);
    if (!src) {
        LOG_ERROR("cml_tp_shard_weight: failed to get weight data");
        return NULL;
    }

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    if (dim == 0) {
        /* Row sharding: each rank gets shard_rows consecutive rows */
        int shard_rows = rows / tp_size;
        int shard_shape[2] = {shard_rows, cols};
        size_t shard_elems = (size_t)shard_rows * cols;
        float* shard_data = (float*)malloc(shard_elems * sizeof(float));
        if (!shard_data) {
            LOG_ERROR("cml_tp_shard_weight: allocation failed");
            return NULL;
        }
        size_t offset = (size_t)tp_rank * shard_rows * cols;
        memcpy(shard_data, src + offset, shard_elems * sizeof(float));

        Tensor* shard = tensor_from_data(shard_data, shard_shape, 2, &cfg);
        free(shard_data);
        return shard;
    } else {
        /* Column sharding: each rank gets shard_cols consecutive columns */
        int shard_cols = cols / tp_size;
        int shard_shape[2] = {rows, shard_cols};
        size_t shard_elems = (size_t)rows * shard_cols;
        float* shard_data = (float*)malloc(shard_elems * sizeof(float));
        if (!shard_data) {
            LOG_ERROR("cml_tp_shard_weight: allocation failed");
            return NULL;
        }
        int col_start = tp_rank * shard_cols;
        for (int r = 0; r < rows; r++) {
            memcpy(shard_data + (size_t)r * shard_cols,
                   src + (size_t)r * cols + col_start,
                   (size_t)shard_cols * sizeof(float));
        }

        Tensor* shard = tensor_from_data(shard_data, shard_shape, 2, &cfg);
        free(shard_data);
        return shard;
    }
}

/* ========================================================================
 * Column-parallel linear
 * ======================================================================== */

CMLColumnParallelLinear* cml_column_parallel_create(Tensor* full_weight,
                                                     Tensor* full_bias,
                                                     int tp_size,
                                                     int tp_rank)
{
    if (!full_weight) {
        LOG_ERROR("cml_column_parallel_create: full_weight is NULL");
        return NULL;
    }
    if (full_weight->ndim != 2) {
        LOG_ERROR("cml_column_parallel_create: expected 2-D weight, got ndim=%d",
                  full_weight->ndim);
        return NULL;
    }
    if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size) {
        LOG_ERROR("cml_column_parallel_create: invalid tp_size=%d or tp_rank=%d",
                  tp_size, tp_rank);
        return NULL;
    }

    int out_features = full_weight->shape[0];
    int in_features  = full_weight->shape[1];

    if (out_features % tp_size != 0) {
        LOG_ERROR("cml_column_parallel_create: out_features (%d) not divisible by tp_size (%d)",
                  out_features, tp_size);
        return NULL;
    }

    CMLColumnParallelLinear* cp = (CMLColumnParallelLinear*)calloc(1, sizeof(*cp));
    if (!cp) {
        LOG_ERROR("cml_column_parallel_create: allocation failed");
        return NULL;
    }

    cp->in_features  = in_features;
    cp->out_features = out_features;
    cp->tp_size      = tp_size;
    cp->tp_rank      = tp_rank;

    /* Shard weight along dim 0 (output dimension) */
    cp->weight = cml_tp_shard_weight(full_weight, 0, tp_size, tp_rank);
    if (!cp->weight) {
        LOG_ERROR("cml_column_parallel_create: failed to shard weight");
        free(cp);
        return NULL;
    }

    /* Shard bias along dim 0 if present (bias is 1-D [out_features]) */
    if (full_bias) {
        tensor_ensure_executed(full_bias);
        const float* bias_src = (const float*)tensor_data_ptr(full_bias);
        if (!bias_src) {
            LOG_ERROR("cml_column_parallel_create: failed to get bias data");
            tensor_free(cp->weight);
            free(cp);
            return NULL;
        }

        int shard_out = out_features / tp_size;
        int bias_shape[1] = {shard_out};
        float* bias_data = (float*)malloc((size_t)shard_out * sizeof(float));
        if (!bias_data) {
            LOG_ERROR("cml_column_parallel_create: bias allocation failed");
            tensor_free(cp->weight);
            free(cp);
            return NULL;
        }
        memcpy(bias_data, bias_src + tp_rank * shard_out, (size_t)shard_out * sizeof(float));

        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        cp->bias = tensor_from_data(bias_data, bias_shape, 1, &cfg);
        free(bias_data);
        if (!cp->bias) {
            LOG_ERROR("cml_column_parallel_create: failed to create bias tensor");
            tensor_free(cp->weight);
            free(cp);
            return NULL;
        }
    } else {
        cp->bias = NULL;
    }

    LOG_DEBUG("Column-parallel created: rank %d/%d, local weight [%d, %d]",
              tp_rank, tp_size, cp->weight->shape[0], cp->weight->shape[1]);

    return cp;
}

void cml_column_parallel_free(CMLColumnParallelLinear* cp)
{
    if (!cp) return;
    if (cp->weight) tensor_free(cp->weight);
    if (cp->bias)   tensor_free(cp->bias);
    free(cp);
}

Tensor* cml_column_parallel_forward(CMLColumnParallelLinear* cp, Tensor* input)
{
    if (!cp || !input) {
        LOG_ERROR("cml_column_parallel_forward: NULL argument");
        return NULL;
    }
    if (input->ndim != 2) {
        LOG_ERROR("cml_column_parallel_forward: expected 2-D input, got ndim=%d", input->ndim);
        return NULL;
    }

    int batch       = input->shape[0];
    int in_features = input->shape[1];

    if (in_features != cp->in_features) {
        LOG_ERROR("cml_column_parallel_forward: input in_features (%d) != expected (%d)",
                  in_features, cp->in_features);
        return NULL;
    }

    int local_out = cp->weight->shape[0]; /* out_features / tp_size */

    tensor_ensure_executed(input);
    tensor_ensure_executed(cp->weight);
    const float* input_data  = (const float*)tensor_data_ptr(input);
    const float* weight_data = (const float*)tensor_data_ptr(cp->weight);
    if (!input_data || !weight_data) {
        LOG_ERROR("cml_column_parallel_forward: failed to get data pointers");
        return NULL;
    }

    size_t out_elems = (size_t)batch * local_out;
    float* out_data = (float*)malloc(out_elems * sizeof(float));
    if (!out_data) {
        LOG_ERROR("cml_column_parallel_forward: allocation failed");
        return NULL;
    }

    /* output[i, j] = sum_k input[i, k] * weight[j, k] */
    matmul_weight_transposed(input_data, batch, in_features,
                             weight_data, local_out, out_data);

    /* Add bias if present */
    if (cp->bias) {
        tensor_ensure_executed(cp->bias);
        const float* bias_data = (const float*)tensor_data_ptr(cp->bias);
        if (bias_data) {
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < local_out; j++) {
                    out_data[i * local_out + j] += bias_data[j];
                }
            }
        }
    }

    int out_shape[2] = {batch, local_out};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* output = tensor_from_data(out_data, out_shape, 2, &cfg);
    free(out_data);
    return output;
}

/* ========================================================================
 * Row-parallel linear
 * ======================================================================== */

CMLRowParallelLinear* cml_row_parallel_create(Tensor* full_weight,
                                               Tensor* full_bias,
                                               int tp_size,
                                               int tp_rank)
{
    if (!full_weight) {
        LOG_ERROR("cml_row_parallel_create: full_weight is NULL");
        return NULL;
    }
    if (full_weight->ndim != 2) {
        LOG_ERROR("cml_row_parallel_create: expected 2-D weight, got ndim=%d",
                  full_weight->ndim);
        return NULL;
    }
    if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size) {
        LOG_ERROR("cml_row_parallel_create: invalid tp_size=%d or tp_rank=%d",
                  tp_size, tp_rank);
        return NULL;
    }

    int out_features = full_weight->shape[0];
    int in_features  = full_weight->shape[1];

    if (in_features % tp_size != 0) {
        LOG_ERROR("cml_row_parallel_create: in_features (%d) not divisible by tp_size (%d)",
                  in_features, tp_size);
        return NULL;
    }

    CMLRowParallelLinear* rp = (CMLRowParallelLinear*)calloc(1, sizeof(*rp));
    if (!rp) {
        LOG_ERROR("cml_row_parallel_create: allocation failed");
        return NULL;
    }

    rp->in_features  = in_features;
    rp->out_features = out_features;
    rp->tp_size      = tp_size;
    rp->tp_rank      = tp_rank;

    /* Shard weight along dim 1 (input dimension) */
    rp->weight = cml_tp_shard_weight(full_weight, 1, tp_size, tp_rank);
    if (!rp->weight) {
        LOG_ERROR("cml_row_parallel_create: failed to shard weight");
        free(rp);
        return NULL;
    }

    /* Only rank 0 gets the bias; other ranks set bias = NULL */
    if (full_bias && tp_rank == 0) {
        tensor_ensure_executed(full_bias);
        const float* bias_src = (const float*)tensor_data_ptr(full_bias);
        if (!bias_src) {
            LOG_ERROR("cml_row_parallel_create: failed to get bias data");
            tensor_free(rp->weight);
            free(rp);
            return NULL;
        }

        int bias_shape[1] = {out_features};
        float* bias_data = (float*)malloc((size_t)out_features * sizeof(float));
        if (!bias_data) {
            LOG_ERROR("cml_row_parallel_create: bias allocation failed");
            tensor_free(rp->weight);
            free(rp);
            return NULL;
        }
        memcpy(bias_data, bias_src, (size_t)out_features * sizeof(float));

        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        rp->bias = tensor_from_data(bias_data, bias_shape, 1, &cfg);
        free(bias_data);
        if (!rp->bias) {
            LOG_ERROR("cml_row_parallel_create: failed to create bias tensor");
            tensor_free(rp->weight);
            free(rp);
            return NULL;
        }
    } else {
        rp->bias = NULL;
    }

    LOG_DEBUG("Row-parallel created: rank %d/%d, local weight [%d, %d]",
              tp_rank, tp_size, rp->weight->shape[0], rp->weight->shape[1]);

    return rp;
}

void cml_row_parallel_free(CMLRowParallelLinear* rp)
{
    if (!rp) return;
    if (rp->weight) tensor_free(rp->weight);
    if (rp->bias)   tensor_free(rp->bias);
    free(rp);
}

Tensor* cml_row_parallel_forward(CMLRowParallelLinear* rp, Tensor* input)
{
    if (!rp || !input) {
        LOG_ERROR("cml_row_parallel_forward: NULL argument");
        return NULL;
    }
    if (input->ndim != 2) {
        LOG_ERROR("cml_row_parallel_forward: expected 2-D input, got ndim=%d", input->ndim);
        return NULL;
    }

    int batch      = input->shape[0];
    int local_in   = input->shape[1];
    int expected_in = rp->in_features / rp->tp_size;

    if (local_in != expected_in) {
        LOG_ERROR("cml_row_parallel_forward: input in_features (%d) != expected (%d)",
                  local_in, expected_in);
        return NULL;
    }

    int out_features = rp->out_features;

    tensor_ensure_executed(input);
    tensor_ensure_executed(rp->weight);
    const float* input_data  = (const float*)tensor_data_ptr(input);
    const float* weight_data = (const float*)tensor_data_ptr(rp->weight);
    if (!input_data || !weight_data) {
        LOG_ERROR("cml_row_parallel_forward: failed to get data pointers");
        return NULL;
    }

    size_t out_elems = (size_t)batch * out_features;
    float* out_data = (float*)malloc(out_elems * sizeof(float));
    if (!out_data) {
        LOG_ERROR("cml_row_parallel_forward: allocation failed");
        return NULL;
    }

    /* output[i, j] = sum_k input[i, k] * weight[j, k]
     * weight is [out_features, in_features / tp_size]
     * input  is [batch, in_features / tp_size]
     */
    matmul_weight_transposed(input_data, batch, local_in,
                             weight_data, out_features, out_data);

    /* Add bias (only rank 0 has bias) */
    if (rp->bias) {
        tensor_ensure_executed(rp->bias);
        const float* bias_data = (const float*)tensor_data_ptr(rp->bias);
        if (bias_data) {
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < out_features; j++) {
                    out_data[i * out_features + j] += bias_data[j];
                }
            }
        }
    }

    int out_shape[2] = {batch, out_features};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* output = tensor_from_data(out_data, out_shape, 2, &cfg);
    free(out_data);
    return output;
}

/* ========================================================================
 * All-reduce (simulated)
 * ======================================================================== */

Tensor* cml_tp_all_reduce_sum(Tensor** partials, int num_parts)
{
    if (!partials || num_parts <= 0) {
        LOG_ERROR("cml_tp_all_reduce_sum: invalid arguments");
        return NULL;
    }
    if (!partials[0]) {
        LOG_ERROR("cml_tp_all_reduce_sum: partials[0] is NULL");
        return NULL;
    }

    int ndim = partials[0]->ndim;
    size_t numel = partials[0]->numel;

    /* Verify all partials have the same shape */
    for (int p = 1; p < num_parts; p++) {
        if (!partials[p]) {
            LOG_ERROR("cml_tp_all_reduce_sum: partials[%d] is NULL", p);
            return NULL;
        }
        if (partials[p]->ndim != ndim) {
            LOG_ERROR("cml_tp_all_reduce_sum: shape mismatch at partials[%d]", p);
            return NULL;
        }
        for (int d = 0; d < ndim; d++) {
            if (partials[p]->shape[d] != partials[0]->shape[d]) {
                LOG_ERROR("cml_tp_all_reduce_sum: shape mismatch at dim %d of partials[%d]", d, p);
                return NULL;
            }
        }
    }

    /* Allocate output and sum all partials */
    float* sum_data = (float*)calloc(numel, sizeof(float));
    if (!sum_data) {
        LOG_ERROR("cml_tp_all_reduce_sum: allocation failed");
        return NULL;
    }

    for (int p = 0; p < num_parts; p++) {
        tensor_ensure_executed(partials[p]);
        const float* pdata = (const float*)tensor_data_ptr(partials[p]);
        if (!pdata) {
            LOG_ERROR("cml_tp_all_reduce_sum: failed to get data for partials[%d]", p);
            free(sum_data);
            return NULL;
        }
        for (size_t i = 0; i < numel; i++) {
            sum_data[i] += pdata[i];
        }
    }

    /* Copy shape from partials[0] */
    int* out_shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!out_shape) {
        LOG_ERROR("cml_tp_all_reduce_sum: shape allocation failed");
        free(sum_data);
        return NULL;
    }
    memcpy(out_shape, partials[0]->shape, (size_t)ndim * sizeof(int));

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(sum_data, out_shape, ndim, &cfg);
    free(sum_data);
    free(out_shape);
    return result;
}
