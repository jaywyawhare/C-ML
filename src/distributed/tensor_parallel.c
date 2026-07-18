#include "distributed/tensor_parallel.h"
#include "distributed/distributed.h"
#include "ops/uops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

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
        float* shard_data = (float*)cml_malloc(shard_elems * sizeof(float));
        if (!shard_data) {
            LOG_ERROR("cml_tp_shard_weight: allocation failed");
            return NULL;
        }
        size_t offset = (size_t)tp_rank * shard_rows * cols;
        memcpy(shard_data, src + offset, shard_elems * sizeof(float));

        Tensor* shard = tensor_from_data(shard_data, shard_shape, 2, &cfg);
        cml_free(shard_data);
        return shard;
    } else {
        /* Column sharding: each rank gets shard_cols consecutive columns */
        int shard_cols = cols / tp_size;
        int shard_shape[2] = {rows, shard_cols};
        size_t shard_elems = (size_t)rows * shard_cols;
        float* shard_data = (float*)cml_malloc(shard_elems * sizeof(float));
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
        cml_free(shard_data);
        return shard;
    }
}

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

    CMLColumnParallelLinear* cp = (CMLColumnParallelLinear*)cml_calloc(1, sizeof(*cp));
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
        cml_free(cp);
        return NULL;
    }

    /* Shard bias along dim 0 if present (bias is 1-D [out_features]) */
    if (full_bias) {
        tensor_ensure_executed(full_bias);
        const float* bias_src = (const float*)tensor_data_ptr(full_bias);
        if (!bias_src) {
            LOG_ERROR("cml_column_parallel_create: failed to get bias data");
            tensor_free(cp->weight);
            cml_free(cp);
            return NULL;
        }

        int shard_out = out_features / tp_size;
        int bias_shape[1] = {shard_out};
        float* bias_data = (float*)cml_malloc((size_t)shard_out * sizeof(float));
        if (!bias_data) {
            LOG_ERROR("cml_column_parallel_create: bias allocation failed");
            tensor_free(cp->weight);
            cml_free(cp);
            return NULL;
        }
        memcpy(bias_data, bias_src + tp_rank * shard_out, (size_t)shard_out * sizeof(float));

        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        cp->bias = tensor_from_data(bias_data, bias_shape, 1, &cfg);
        cml_free(bias_data);
        if (!cp->bias) {
            LOG_ERROR("cml_column_parallel_create: failed to create bias tensor");
            tensor_free(cp->weight);
            cml_free(cp);
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
    cml_free(cp);
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

    (void)batch;

    /* Single autograd-tracked fused node: output = input @ weight^T + bias
     * ([batch, out/tp]). Building it through uop_linear (instead of a raw
     * matmul into a fresh buffer) links the output into the IR graph, so
     * tensor_backward produces a gradient for the sharded `weight`/`bias`.
     * The previous raw-matmul path had no autograd edge → TP was inference-only. */
    Tensor* output = uop_linear(input, cp->weight, cp->bias);
    if (!output) {
        LOG_ERROR("cml_column_parallel_forward: uop_linear failed");
        return NULL;
    }
    /* Materialize at the layer boundary (a TP layer is a natural
     * communication/realization point). This keeps the autograd node intact for
     * backward while giving callers a concrete activation to chain — matching
     * the materialized-input contract the rest of the TP path expects. */
    tensor_ensure_executed(output);
    return output;
}

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

    CMLRowParallelLinear* rp = (CMLRowParallelLinear*)cml_calloc(1, sizeof(*rp));
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
        cml_free(rp);
        return NULL;
    }

    /* Only rank 0 gets the bias; other ranks set bias = NULL */
    if (full_bias && tp_rank == 0) {
        tensor_ensure_executed(full_bias);
        const float* bias_src = (const float*)tensor_data_ptr(full_bias);
        if (!bias_src) {
            LOG_ERROR("cml_row_parallel_create: failed to get bias data");
            tensor_free(rp->weight);
            cml_free(rp);
            return NULL;
        }

        int bias_shape[1] = {out_features};
        float* bias_data = (float*)cml_malloc((size_t)out_features * sizeof(float));
        if (!bias_data) {
            LOG_ERROR("cml_row_parallel_create: bias allocation failed");
            tensor_free(rp->weight);
            cml_free(rp);
            return NULL;
        }
        memcpy(bias_data, bias_src, (size_t)out_features * sizeof(float));

        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        rp->bias = tensor_from_data(bias_data, bias_shape, 1, &cfg);
        cml_free(bias_data);
        if (!rp->bias) {
            LOG_ERROR("cml_row_parallel_create: failed to create bias tensor");
            tensor_free(rp->weight);
            cml_free(rp);
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
    cml_free(rp);
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

    (void)batch;
    (void)local_in;

    /* Autograd-tracked partial: output = input_shard @ weight^T (+ bias on
     * rank 0), shape [batch, out_features]. This is each rank's PARTIAL — the
     * caller must all-reduce-sum the partials across ranks (see
     * cml_row_parallel_all_reduce) for the final result. Built via uop_linear
     * so gradients flow back to the sharded `weight`. */
    Tensor* output = uop_linear(input, rp->weight, rp->bias);
    if (!output) {
        LOG_ERROR("cml_row_parallel_forward: uop_linear failed");
        return NULL;
    }
    /* Materialize at the layer boundary (see cml_column_parallel_forward). */
    tensor_ensure_executed(output);
    return output;
}

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
    float* sum_data = (float*)cml_calloc(numel, sizeof(float));
    if (!sum_data) {
        LOG_ERROR("cml_tp_all_reduce_sum: allocation failed");
        return NULL;
    }

    for (int p = 0; p < num_parts; p++) {
        tensor_ensure_executed(partials[p]);
        const float* pdata = (const float*)tensor_data_ptr(partials[p]);
        if (!pdata) {
            LOG_ERROR("cml_tp_all_reduce_sum: failed to get data for partials[%d]", p);
            cml_free(sum_data);
            return NULL;
        }
        for (size_t i = 0; i < numel; i++) {
            sum_data[i] += pdata[i];
        }
    }

    /* Copy shape from partials[0] */
    int* out_shape = (int*)cml_malloc((size_t)ndim * sizeof(int));
    if (!out_shape) {
        LOG_ERROR("cml_tp_all_reduce_sum: shape allocation failed");
        cml_free(sum_data);
        return NULL;
    }
    memcpy(out_shape, partials[0]->shape, (size_t)ndim * sizeof(int));

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(sum_data, out_shape, ndim, &cfg);
    cml_free(sum_data);
    cml_free(out_shape);
    return result;
}

Tensor* cml_tp_all_gather(Tensor** partials, int num_parts, int dim)
{
    if (!partials || num_parts <= 0) {
        LOG_ERROR("cml_tp_all_gather: invalid arguments");
        return NULL;
    }
    for (int p = 0; p < num_parts; p++) {
        if (!partials[p]) {
            LOG_ERROR("cml_tp_all_gather: partials[%d] is NULL", p);
            return NULL;
        }
        tensor_ensure_executed(partials[p]);
    }
    /* Column-parallel: each rank holds [batch, out_features/tp]; the full
     * output is the rank-order concatenation along the feature dim. */
    return tensor_concat(partials, num_parts, dim);
}

int cml_row_parallel_all_reduce(Tensor* partial)
{
    if (!partial) {
        LOG_ERROR("cml_row_parallel_all_reduce: partial is NULL");
        return -1;
    }
    /* Sum this rank's row-parallel partial with every other rank's, in place,
     * using the real process-group collective (a no-op at world_size==1).
     *
     * The gradient of an all-reduce-sum is the identity to each rank's input,
     * so overwriting the materialized partial in place keeps per-rank weight
     * gradients correct: dL/d(partial_r) == dL/d(reduced) for every rank r. */
    tensor_ensure_executed(partial);
    return cml_dist_allreduce(partial, DIST_REDUCE_SUM);
}
