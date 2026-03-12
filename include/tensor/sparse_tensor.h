#ifndef CML_TENSOR_SPARSE_TENSOR_H
#define CML_TENSOR_SPARSE_TENSOR_H

#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TENSOR_FORMAT_DENSE,
    TENSOR_FORMAT_SPARSE_COO
} TensorFormat;

/* Sparse COO: stores non-zero entries as (indices, values) pairs */
typedef struct SparseCOOData {
    Tensor* indices;    /* [nnz, ndim] - integer coordinates of non-zeros */
    Tensor* values;     /* [nnz] - values at those coordinates */
    int* dense_shape;
    int dense_ndim;
    int nnz;
} SparseCOOData;

SparseCOOData* sparse_coo_tensor(Tensor* indices, Tensor* values,
                                  const int* dense_shape, int dense_ndim);
SparseCOOData* sparse_from_dense(Tensor* dense);
Tensor* sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config);

/* C = A * B where A is sparse [M, K] and B is dense [K, N] */
Tensor* sparse_matmul(SparseCOOData* sparse, Tensor* dense);

/* Sort indices and sum duplicate entries */
SparseCOOData* sparse_coalesce(SparseCOOData* sparse);

void sparse_free(SparseCOOData* sparse);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_SPARSE_TENSOR_H
