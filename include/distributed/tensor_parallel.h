/**
 * @file tensor_parallel.h
 * @brief Tensor parallelism for model inference / training
 *
 * Implements Megatron-LM style column-parallel and row-parallel linear layers.
 * Column parallel shards the output dimension (used for QKV / gate / up projections).
 * Row parallel shards the input dimension (used for O / down projections), requiring
 * an all-reduce after the forward pass.
 *
 * For single-process simulation the all-reduce is a simple element-wise sum of
 * the partial results from each rank.
 */

#ifndef CML_DISTRIBUTED_TENSOR_PARALLEL_H
#define CML_DISTRIBUTED_TENSOR_PARALLEL_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Structs
 * ======================================================================== */

/**
 * @brief Column-parallel linear layer.
 *
 * Shards weight along the output dimension so that each rank holds
 * a [out_features / tp_size, in_features] slice.  The forward pass
 * computes a local matmul producing [batch, out_features / tp_size].
 */
typedef struct CMLColumnParallelLinear {
    Tensor* weight;     /* [out_features / tp_size, in_features] - local shard */
    Tensor* bias;       /* [out_features / tp_size] or NULL */
    int in_features;
    int out_features;   /* global out_features */
    int tp_size;
    int tp_rank;
} CMLColumnParallelLinear;

/**
 * @brief Row-parallel linear layer.
 *
 * Shards weight along the input dimension so that each rank holds
 * a [out_features, in_features / tp_size] slice.  The forward pass
 * computes a local matmul producing [batch, out_features]; the caller
 * is responsible for performing an all-reduce sum across ranks.
 */
typedef struct CMLRowParallelLinear {
    Tensor* weight;     /* [out_features, in_features / tp_size] - local shard */
    Tensor* bias;       /* [out_features] or NULL (only rank 0 has bias) */
    int in_features;    /* global in_features */
    int out_features;
    int tp_size;
    int tp_rank;
} CMLRowParallelLinear;

/**
 * @brief Lightweight config carried around for convenience.
 */
typedef struct CMLTensorParallelConfig {
    int tp_size;
    int tp_rank;
} CMLTensorParallelConfig;

/* ========================================================================
 * Column-parallel API
 * ======================================================================== */

/**
 * @brief Create a column-parallel linear layer by sharding a full weight.
 *
 * The full weight has shape [out_features, in_features].  Each rank receives
 * rows [rank * shard_size .. (rank+1) * shard_size) where
 * shard_size = out_features / tp_size.
 *
 * @param full_weight  Full weight tensor [out_features, in_features]
 * @param full_bias    Full bias tensor [out_features] or NULL
 * @param tp_size      Number of tensor-parallel ranks
 * @param tp_rank      This rank's index (0-based)
 * @return Allocated CMLColumnParallelLinear, or NULL on error
 */
CMLColumnParallelLinear* cml_column_parallel_create(Tensor* full_weight,
                                                     Tensor* full_bias,
                                                     int tp_size,
                                                     int tp_rank);

/**
 * @brief Free a column-parallel linear layer and its local shard tensors.
 */
void cml_column_parallel_free(CMLColumnParallelLinear* cp);

/**
 * @brief Forward pass through a column-parallel linear layer.
 *
 * Computes output = input @ weight^T  (local shard).
 * Input shape:  [batch, in_features]
 * Output shape: [batch, out_features / tp_size]
 *
 * @param cp     Column-parallel layer
 * @param input  Input tensor [batch, in_features]
 * @return Output tensor [batch, out_features / tp_size], or NULL on error
 */
Tensor* cml_column_parallel_forward(CMLColumnParallelLinear* cp, Tensor* input);

/* ========================================================================
 * Row-parallel API
 * ======================================================================== */

/**
 * @brief Create a row-parallel linear layer by sharding a full weight.
 *
 * The full weight has shape [out_features, in_features].  Each rank receives
 * columns [rank * shard_size .. (rank+1) * shard_size) where
 * shard_size = in_features / tp_size.
 *
 * @param full_weight  Full weight tensor [out_features, in_features]
 * @param full_bias    Full bias tensor [out_features] or NULL
 * @param tp_size      Number of tensor-parallel ranks
 * @param tp_rank      This rank's index (0-based)
 * @return Allocated CMLRowParallelLinear, or NULL on error
 */
CMLRowParallelLinear* cml_row_parallel_create(Tensor* full_weight,
                                               Tensor* full_bias,
                                               int tp_size,
                                               int tp_rank);

/**
 * @brief Free a row-parallel linear layer and its local shard tensors.
 */
void cml_row_parallel_free(CMLRowParallelLinear* rp);

/**
 * @brief Forward pass through a row-parallel linear layer.
 *
 * Computes output = input @ weight^T  (local shard).
 * Input shape:  [batch, in_features / tp_size]
 * Output shape: [batch, out_features]
 *
 * The caller must perform an all-reduce sum of the outputs across ranks
 * to obtain the final result.
 *
 * @param rp     Row-parallel layer
 * @param input  Input tensor [batch, in_features / tp_size]
 * @return Output tensor [batch, out_features], or NULL on error
 */
Tensor* cml_row_parallel_forward(CMLRowParallelLinear* rp, Tensor* input);

/* ========================================================================
 * All-reduce & utilities
 * ======================================================================== */

/**
 * @brief Simulate an all-reduce sum across tensor-parallel ranks.
 *
 * In a real distributed setting each rank would contribute its partial
 * result and the communication library would produce the sum.  For
 * single-process simulation we simply sum the arrays element-wise.
 *
 * All tensors in @p partials must have the same shape and dtype.
 *
 * @param partials   Array of partial-result tensors (one per rank)
 * @param num_parts  Number of partials (== tp_size)
 * @return New tensor holding the element-wise sum, or NULL on error
 */
Tensor* cml_tp_all_reduce_sum(Tensor** partials, int num_parts);

/**
 * @brief Extract a shard from a weight tensor along a given dimension.
 *
 * For dim == 0 (row sharding):  returns rows
 *   [tp_rank * shard_rows .. (tp_rank+1) * shard_rows)
 * For dim == 1 (column sharding): returns columns
 *   [tp_rank * shard_cols .. (tp_rank+1) * shard_cols)
 *
 * The returned tensor owns a freshly allocated copy of the data.
 *
 * @param weight   Full weight tensor (2-D)
 * @param dim      Dimension to shard (0 or 1)
 * @param tp_size  Number of tensor-parallel ranks
 * @param tp_rank  This rank's index
 * @return New tensor holding the shard, or NULL on error
 */
Tensor* cml_tp_shard_weight(Tensor* weight, int dim, int tp_size, int tp_rank);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_TENSOR_PARALLEL_H */
