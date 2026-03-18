#ifndef CML_DISTRIBUTED_TENSOR_PARALLEL_H
#define CML_DISTRIBUTED_TENSOR_PARALLEL_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLColumnParallelLinear {
    Tensor* weight;     /* [out_features / tp_size, in_features] - local shard */
    Tensor* bias;       /* [out_features / tp_size] or NULL */
    int in_features;
    int out_features;   /* global out_features */
    int tp_size;
    int tp_rank;
} CMLColumnParallelLinear;

/* Caller must perform an all-reduce sum across ranks after forward. */
typedef struct CMLRowParallelLinear {
    Tensor* weight;     /* [out_features, in_features / tp_size] - local shard */
    Tensor* bias;       /* [out_features] or NULL (only rank 0 has bias) */
    int in_features;    /* global in_features */
    int out_features;
    int tp_size;
    int tp_rank;
} CMLRowParallelLinear;

typedef struct CMLTensorParallelConfig {
    int tp_size;
    int tp_rank;
} CMLTensorParallelConfig;

CMLColumnParallelLinear* cml_column_parallel_create(Tensor* full_weight,
                                                     Tensor* full_bias,
                                                     int tp_size,
                                                     int tp_rank);

void cml_column_parallel_free(CMLColumnParallelLinear* cp);

Tensor* cml_column_parallel_forward(CMLColumnParallelLinear* cp, Tensor* input);

CMLRowParallelLinear* cml_row_parallel_create(Tensor* full_weight,
                                               Tensor* full_bias,
                                               int tp_size,
                                               int tp_rank);

void cml_row_parallel_free(CMLRowParallelLinear* rp);

/* Caller must all-reduce sum outputs across ranks for the final result. */
Tensor* cml_row_parallel_forward(CMLRowParallelLinear* rp, Tensor* input);

/* All tensors in partials must have the same shape and dtype. */
Tensor* cml_tp_all_reduce_sum(Tensor** partials, int num_parts);

/* Returns a freshly allocated copy of the shard data. */
Tensor* cml_tp_shard_weight(Tensor* weight, int dim, int tp_size, int tp_rank);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_TENSOR_PARALLEL_H */
