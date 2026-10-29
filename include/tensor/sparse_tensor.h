#ifndef CML_TENSOR_SPARSE_TENSOR_H
#define CML_TENSOR_SPARSE_TENSOR_H

#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TENSOR_FORMAT_DENSE, TENSOR_FORMAT_SPARSE_COO } TensorFormat;

/* Sparse COO: stores non-zero entries as (indices, values) pairs */
typedef struct SparseCOOData {
    Tensor* indices; /* [nnz, ndim] - integer coordinates of non-zeros */
    Tensor* values;  /* [nnz] - values at those coordinates */
    int* dense_shape;
    int dense_ndim;
    int nnz;
} SparseCOOData;

SparseCOOData* sparse_coo_tensor(Tensor* indices, Tensor* values, const int* dense_shape,
                                 int dense_ndim);
SparseCOOData* sparse_from_dense(Tensor* dense);
Tensor* sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config);

/**
 * C = A * B where A is sparse [M, K] and B is dense [K, N]
 * @param sparse (SparseCOOData*)
 * @param dense (Tensor*)
 * @return (Tensor*)
 */
Tensor* sparse_matmul(SparseCOOData* sparse, Tensor* dense);

/**
 * Sort indices and sum duplicate entries
 * @param sparse (SparseCOOData*)
 * @return (SparseCOOData*)
 */
SparseCOOData* sparse_coalesce(SparseCOOData* sparse);

void sparse_free(SparseCOOData* sparse);

/* CSR: crow_indices [rows+1] INT32 row pointers, col_indices [nnz] INT32 column
 * coordinates, values [nnz]; entries grouped by row and sorted by column. */
typedef struct SparseCSRData {
    Tensor* crow_indices;
    Tensor* col_indices;
    Tensor* values;
    int rows;
    int cols;
    int nnz;
} SparseCSRData;

/* CSC: ccol_indices [cols+1] INT32 column pointers, row_indices [nnz] INT32 row
 * coordinates, values [nnz]. */
typedef struct SparseCSCData {
    Tensor* ccol_indices;
    Tensor* row_indices;
    Tensor* values;
    int rows;
    int cols;
    int nnz;
} SparseCSCData;

/**
 * Build a CSR tensor, cloning the supplied arrays.
 * @param crow_indices (Tensor*)
 * @param col_indices (Tensor*)
 * @param values (Tensor*)
 * @param rows (int)
 * @param cols (int)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_csr_create(Tensor* crow_indices, Tensor* col_indices, Tensor* values,
                                 int rows, int cols);

/**
 * Build a CSC tensor, cloning the supplied arrays.
 * @param ccol_indices (Tensor*)
 * @param row_indices (Tensor*)
 * @param values (Tensor*)
 * @param rows (int)
 * @param cols (int)
 * @return (SparseCSCData*)
 */
SparseCSCData* sparse_csc_create(Tensor* ccol_indices, Tensor* row_indices, Tensor* values,
                                 int rows, int cols);

SparseCSRData* sparse_coo_to_csr(SparseCOOData* coo);
SparseCOOData* sparse_csr_to_coo(SparseCSRData* csr);
SparseCSCData* sparse_csr_to_csc(SparseCSRData* csr);
Tensor* sparse_csr_to_dense(SparseCSRData* csr, const TensorConfig* config);
SparseCSRData* dense_to_sparse_csr(Tensor* dense);

/**
 * CSR x dense SpMM: A [M, K] times B [K, N] -> dense [M, N].
 * Differentiable wrt B and the CSR values via the shared COO SpMM VJP.
 * @param csr (SparseCSRData*)
 * @param dense (Tensor*)
 * @return (Tensor*)
 */
Tensor* sparse_csr_matmul(SparseCSRData* csr, Tensor* dense);

/**
 * CSR matrix times dense vector: A [M, N] times v [N] -> dense [M] (eager).
 * @param csr (SparseCSRData*)
 * @param vec (Tensor*)
 * @return (Tensor*)
 */
Tensor* sparse_mv(SparseCSRData* csr, Tensor* vec);

/**
 * Element-wise sum of two equally-shaped CSR matrices.
 * @param a (SparseCSRData*)
 * @param b (SparseCSRData*)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_add(SparseCSRData* a, SparseCSRData* b);

/**
 * Scale every stored value by a scalar, returning a new CSR.
 * @param csr (SparseCSRData*)
 * @param scalar (float)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_mul_scalar(SparseCSRData* csr, float scalar);

/**
 * Transpose a CSR [M, N] into the CSC of its [N, M] transpose (O(1), structural).
 * @param csr (SparseCSRData*)
 * @return (SparseCSCData*)
 */
SparseCSCData* sparse_transpose(SparseCSRData* csr);

/**
 * Sum CSR entries: dim < 0 scalar [1], dim 0 column sums [cols], dim 1 row sums [rows].
 * @param csr (SparseCSRData*)
 * @param dim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* sparse_sum(SparseCSRData* csr, int dim, const TensorConfig* config);

int sparse_csr_nnz(const SparseCSRData* csr);
bool sparse_csr_is_sparse(const SparseCSRData* csr);
double sparse_csr_density(const SparseCSRData* csr);

void sparse_csr_free(SparseCSRData* csr);
void sparse_csc_free(SparseCSCData* csc);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_SPARSE_TENSOR_H
