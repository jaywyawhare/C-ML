#include "tensor/sparse_tensor.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include <stdlib.h>
#include <string.h>

/**
 * INT32 1-D tensor of length n on the given device.
 * @param n (int)
 * @param device (DeviceType)
 * @return (Tensor*)
 */
static Tensor* make_i32_1d(int n, DeviceType device) {
    int shape[]      = {n};
    TensorConfig cfg = (TensorConfig){
        .dtype = DTYPE_INT32, .device = device, .has_dtype = true, .has_device = true};
    return tensor_empty(shape, 1, &cfg);
}

/**
 * FLOAT32 1-D tensor of length n on the given device.
 * @param n (int)
 * @param device (DeviceType)
 * @return (Tensor*)
 */
static Tensor* make_f32_1d(int n, DeviceType device) {
    int shape[]      = {n};
    TensorConfig cfg = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = device, .has_dtype = true, .has_device = true};
    return tensor_empty(shape, 1, &cfg);
}

/**
 * Force execution and fetch the CSR row-pointer, column-index and value arrays.
 * @param csr (const SparseCSRData*)
 * @param crow (const int32_t**)
 * @param cidx (const int32_t**)
 * @param vals (const float**)
 */
static void csr_arrays(const SparseCSRData* csr, const int32_t** crow, const int32_t** cidx,
                       const float** vals) {
    tensor_ensure_executed(csr->crow_indices);
    tensor_ensure_executed(csr->col_indices);
    tensor_ensure_executed(csr->values);
    *crow = (const int32_t*)csr->crow_indices->data;
    *cidx = (const int32_t*)csr->col_indices->data;
    *vals = (const float*)csr->values->data;
}

/**
 * Allocate a compressed triplet (pointer array, index array, value array),
 * freeing any partial allocation on failure.
 * @param ptr_len (int)
 * @param nnz (int)
 * @param idx_dev (DeviceType)
 * @param val_dev (DeviceType)
 * @param ptr (Tensor**)
 * @param idx (Tensor**)
 * @param vals (Tensor**)
 * @return (bool)
 */
static bool alloc_csr_triplet(int ptr_len, int nnz, DeviceType idx_dev, DeviceType val_dev,
                              Tensor** ptr, Tensor** idx, Tensor** vals) {
    *ptr  = make_i32_1d(ptr_len, idx_dev);
    *idx  = make_i32_1d(nnz, idx_dev);
    *vals = make_f32_1d(nnz, val_dev);
    if (*ptr && *idx && *vals)
        return true;
    if (*ptr)
        tensor_free(*ptr);
    if (*idx)
        tensor_free(*idx);
    if (*vals)
        tensor_free(*vals);
    *ptr = *idx = *vals = NULL;
    return false;
}

/**
 * Build a CSR tensor from cloned row pointers, column indices and values.
 * @param crow_indices (Tensor*)
 * @param col_indices (Tensor*)
 * @param values (Tensor*)
 * @param rows (int)
 * @param cols (int)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_csr_create(Tensor* crow_indices, Tensor* col_indices, Tensor* values,
                                 int rows, int cols) {
    if (!crow_indices || !col_indices || !values || rows < 0 || cols < 0) {
        LOG_ERROR("sparse_csr_create: invalid arguments");
        return NULL;
    }
    tensor_ensure_executed(crow_indices);
    tensor_ensure_executed(col_indices);
    tensor_ensure_executed(values);

    if (crow_indices->ndim != 1 || crow_indices->shape[0] != rows + 1) {
        LOG_ERROR("sparse_csr_create: crow_indices must be 1D [rows+1]");
        return NULL;
    }
    if (col_indices->ndim != 1 || values->ndim != 1 || col_indices->shape[0] != values->shape[0]) {
        LOG_ERROR("sparse_csr_create: col_indices and values must be 1D [nnz] of equal length");
        return NULL;
    }

    SparseCSRData* csr = cml_calloc(1, sizeof(SparseCSRData));
    if (!csr) {
        LOG_ERROR("sparse_csr_create: failed to allocate SparseCSRData");
        return NULL;
    }
    csr->crow_indices = tensor_clone(crow_indices);
    csr->col_indices  = tensor_clone(col_indices);
    csr->values       = tensor_clone(values);
    if (!csr->crow_indices || !csr->col_indices || !csr->values) {
        sparse_csr_free(csr);
        LOG_ERROR("sparse_csr_create: failed to clone inputs");
        return NULL;
    }
    csr->rows = rows;
    csr->cols = cols;
    csr->nnz  = col_indices->shape[0];
    return csr;
}

/**
 * Build a CSC tensor from cloned column pointers, row indices and values.
 * @param ccol_indices (Tensor*)
 * @param row_indices (Tensor*)
 * @param values (Tensor*)
 * @param rows (int)
 * @param cols (int)
 * @return (SparseCSCData*)
 */
SparseCSCData* sparse_csc_create(Tensor* ccol_indices, Tensor* row_indices, Tensor* values,
                                 int rows, int cols) {
    if (!ccol_indices || !row_indices || !values || rows < 0 || cols < 0) {
        LOG_ERROR("sparse_csc_create: invalid arguments");
        return NULL;
    }
    tensor_ensure_executed(ccol_indices);
    tensor_ensure_executed(row_indices);
    tensor_ensure_executed(values);

    if (ccol_indices->ndim != 1 || ccol_indices->shape[0] != cols + 1) {
        LOG_ERROR("sparse_csc_create: ccol_indices must be 1D [cols+1]");
        return NULL;
    }
    if (row_indices->ndim != 1 || values->ndim != 1 || row_indices->shape[0] != values->shape[0]) {
        LOG_ERROR("sparse_csc_create: row_indices and values must be 1D [nnz] of equal length");
        return NULL;
    }

    SparseCSCData* csc = cml_calloc(1, sizeof(SparseCSCData));
    if (!csc) {
        LOG_ERROR("sparse_csc_create: failed to allocate SparseCSCData");
        return NULL;
    }
    csc->ccol_indices = tensor_clone(ccol_indices);
    csc->row_indices  = tensor_clone(row_indices);
    csc->values       = tensor_clone(values);
    if (!csc->ccol_indices || !csc->row_indices || !csc->values) {
        sparse_csc_free(csc);
        LOG_ERROR("sparse_csc_create: failed to clone inputs");
        return NULL;
    }
    csc->rows = rows;
    csc->cols = cols;
    csc->nnz  = row_indices->shape[0];
    return csc;
}

/**
 * Convert a 2-D COO tensor to CSR; coalesces first so columns are sorted per row.
 * @param coo (SparseCOOData*)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_coo_to_csr(SparseCOOData* coo) {
    if (!coo || coo->dense_ndim != 2) {
        LOG_ERROR("sparse_coo_to_csr: COO must be 2D");
        return NULL;
    }

    SparseCOOData* c = sparse_coalesce(coo);
    if (!c) {
        LOG_ERROR("sparse_coo_to_csr: coalesce failed");
        return NULL;
    }

    int rows = c->dense_shape[0];
    int cols = c->dense_shape[1];
    int nnz  = c->nnz;

    Tensor *crow, *cidx, *vals;
    if (!alloc_csr_triplet(rows + 1, nnz, c->indices->device, c->values->device, &crow, &cidx,
                           &vals)) {
        sparse_free(c);
        LOG_ERROR("sparse_coo_to_csr: allocation failed");
        return NULL;
    }

    int32_t* crow_d = (int32_t*)crow->data;
    int32_t* cidx_d = (int32_t*)cidx->data;
    float* vals_d   = (float*)vals->data;
    memset(crow_d, 0, (size_t)(rows + 1) * sizeof(int32_t));

    if (nnz > 0) {
        tensor_ensure_executed(c->indices);
        tensor_ensure_executed(c->values);
        const int32_t* coo_idx = (const int32_t*)c->indices->data;
        const float* coo_val   = (const float*)c->values->data;
        for (int i = 0; i < nnz; i++)
            crow_d[coo_idx[i * 2 + 0] + 1]++;
        for (int r = 0; r < rows; r++)
            crow_d[r + 1] += crow_d[r];
        for (int i = 0; i < nnz; i++) {
            cidx_d[i] = coo_idx[i * 2 + 1];
            vals_d[i] = coo_val[i];
        }
    }

    SparseCSRData* csr = sparse_csr_create(crow, cidx, vals, rows, cols);
    tensor_free(crow);
    tensor_free(cidx);
    tensor_free(vals);
    sparse_free(c);
    return csr;
}

/**
 * Convert a CSR tensor to COO, expanding row pointers into row coordinates.
 * @param csr (SparseCSRData*)
 * @return (SparseCOOData*)
 */
SparseCOOData* sparse_csr_to_coo(SparseCSRData* csr) {
    if (!csr) {
        LOG_ERROR("sparse_csr_to_coo: NULL input");
        return NULL;
    }
    int nnz = csr->nnz;

    int idx_shape[]  = {nnz, 2};
    TensorConfig icf = (TensorConfig){.dtype      = DTYPE_INT32,
                                      .device     = csr->col_indices->device,
                                      .has_dtype  = true,
                                      .has_device = true};
    Tensor* indices  = tensor_empty(idx_shape, 2, &icf);
    Tensor* values   = make_f32_1d(nnz, csr->values->device);
    if (!indices || !values) {
        if (indices)
            tensor_free(indices);
        if (values)
            tensor_free(values);
        LOG_ERROR("sparse_csr_to_coo: allocation failed");
        return NULL;
    }

    if (nnz > 0) {
        const int32_t *crow_d, *cidx_d;
        const float* vals_d;
        csr_arrays(csr, &crow_d, &cidx_d, &vals_d);
        int32_t* idx_d = (int32_t*)indices->data;
        float* val_d   = (float*)values->data;
        for (int r = 0; r < csr->rows; r++) {
            for (int p = crow_d[r]; p < crow_d[r + 1]; p++) {
                idx_d[p * 2 + 0] = r;
                idx_d[p * 2 + 1] = cidx_d[p];
                val_d[p]         = vals_d[p];
            }
        }
    }

    int dense_shape[]  = {csr->rows, csr->cols};
    SparseCOOData* coo = sparse_coo_tensor(indices, values, dense_shape, 2);
    tensor_free(indices);
    tensor_free(values);
    return coo;
}

/**
 * Convert a CSR tensor to CSC via a counting transpose.
 * @param csr (SparseCSRData*)
 * @return (SparseCSCData*)
 */
SparseCSCData* sparse_csr_to_csc(SparseCSRData* csr) {
    if (!csr) {
        LOG_ERROR("sparse_csr_to_csc: NULL input");
        return NULL;
    }
    int rows = csr->rows, cols = csr->cols, nnz = csr->nnz;

    Tensor *ccol, *ridx, *vals;
    if (!alloc_csr_triplet(cols + 1, nnz, csr->col_indices->device, csr->values->device, &ccol,
                           &ridx, &vals)) {
        LOG_ERROR("sparse_csr_to_csc: allocation failed");
        return NULL;
    }

    int32_t* ccol_d = (int32_t*)ccol->data;
    int32_t* ridx_d = (int32_t*)ridx->data;
    float* vals_d   = (float*)vals->data;
    memset(ccol_d, 0, (size_t)(cols + 1) * sizeof(int32_t));

    if (nnz > 0) {
        const int32_t *crow_s, *cidx_s;
        const float* vals_s;
        csr_arrays(csr, &crow_s, &cidx_s, &vals_s);

        for (int i = 0; i < nnz; i++)
            ccol_d[cidx_s[i] + 1]++;
        for (int c = 0; c < cols; c++)
            ccol_d[c + 1] += ccol_d[c];

        int* next = cml_calloc(cols, sizeof(int));
        if (!next) {
            tensor_free(ccol);
            tensor_free(ridx);
            tensor_free(vals);
            LOG_ERROR("sparse_csr_to_csc: allocation failed");
            return NULL;
        }
        for (int c = 0; c < cols; c++)
            next[c] = ccol_d[c];
        for (int r = 0; r < rows; r++) {
            for (int p = crow_s[r]; p < crow_s[r + 1]; p++) {
                int col     = cidx_s[p];
                int dst     = next[col]++;
                ridx_d[dst] = r;
                vals_d[dst] = vals_s[p];
            }
        }
        cml_free(next);
    }

    SparseCSCData* csc = sparse_csc_create(ccol, ridx, vals, rows, cols);
    tensor_free(ccol);
    tensor_free(ridx);
    tensor_free(vals);
    return csc;
}

/**
 * Scatter CSR entries into a freshly-zeroed dense tensor.
 * @param csr (SparseCSRData*)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* sparse_csr_to_dense(SparseCSRData* csr, const TensorConfig* config) {
    if (!csr) {
        LOG_ERROR("sparse_csr_to_dense: NULL input");
        return NULL;
    }
    int shape[]    = {csr->rows, csr->cols};
    Tensor* output = tensor_zeros(shape, 2, config);
    if (!output) {
        LOG_ERROR("sparse_csr_to_dense: failed to create output");
        return NULL;
    }
    if (csr->nnz == 0)
        return output;

    const int32_t *crow_d, *cidx_d;
    const float* vals_d;
    csr_arrays(csr, &crow_d, &cidx_d, &vals_d);
    tensor_ensure_executed(output);
    float* out_d = (float*)output->data;
    for (int r = 0; r < csr->rows; r++)
        for (int p = crow_d[r]; p < crow_d[r + 1]; p++)
            out_d[r * csr->cols + cidx_d[p]] += vals_d[p];
    return output;
}

/**
 * Convert a 2-D dense tensor to CSR via COO (torch's to_sparse_csr).
 * @param dense (Tensor*)
 * @return (SparseCSRData*)
 */
SparseCSRData* dense_to_sparse_csr(Tensor* dense) {
    if (!dense || dense->ndim != 2) {
        LOG_ERROR("dense_to_sparse_csr: input must be 2D");
        return NULL;
    }
    SparseCOOData* coo = sparse_from_dense(dense);
    if (!coo) {
        LOG_ERROR("dense_to_sparse_csr: dense -> COO failed");
        return NULL;
    }
    SparseCSRData* csr = sparse_coo_to_csr(coo);
    sparse_free(coo);
    return csr;
}

/**
 * COO view backed by csr->values (shared, not cloned) so grad flows to the CSR values.
 * @param csr (SparseCSRData*)
 * @return (SparseCOOData*)
 */
static SparseCOOData* csr_to_coo_view(SparseCSRData* csr) {
    int idx_shape[]  = {csr->nnz, 2};
    TensorConfig icf = (TensorConfig){.dtype      = DTYPE_INT32,
                                      .device     = csr->col_indices->device,
                                      .has_dtype  = true,
                                      .has_device = true};
    Tensor* indices  = tensor_empty(idx_shape, 2, &icf);
    if (!indices)
        return NULL;

    if (csr->nnz > 0) {
        tensor_ensure_executed(csr->crow_indices);
        tensor_ensure_executed(csr->col_indices);
        const int32_t* crow_d = (const int32_t*)csr->crow_indices->data;
        const int32_t* cidx_d = (const int32_t*)csr->col_indices->data;
        int32_t* idx_d        = (int32_t*)indices->data;
        for (int r = 0; r < csr->rows; r++)
            for (int p = crow_d[r]; p < crow_d[r + 1]; p++) {
                idx_d[p * 2 + 0] = r;
                idx_d[p * 2 + 1] = cidx_d[p];
            }
    }

    SparseCOOData* coo = cml_calloc(1, sizeof(SparseCOOData));
    if (!coo) {
        tensor_free(indices);
        return NULL;
    }
    coo->indices     = indices;
    coo->values      = csr->values; /* borrowed, not cloned */
    coo->nnz         = csr->nnz;
    coo->dense_ndim  = 2;
    coo->dense_shape = cml_calloc(2, sizeof(int));
    if (!coo->dense_shape) {
        tensor_free(indices);
        cml_free(coo);
        return NULL;
    }
    coo->dense_shape[0] = csr->rows;
    coo->dense_shape[1] = csr->cols;
    return coo;
}

/**
 * Free a csr_to_coo_view: its generated indices and struct, never the borrowed values.
 * @param coo (SparseCOOData*)
 */
static void csr_coo_view_free(SparseCOOData* coo) {
    if (!coo)
        return;
    if (coo->indices)
        tensor_free(coo->indices);
    if (coo->dense_shape)
        cml_free(coo->dense_shape);
    cml_free(coo);
}

/**
 * CSR x dense SpMM; the autograd path delegates to the COO SpMM VJP.
 * @param csr (SparseCSRData*)
 * @param dense (Tensor*)
 * @return (Tensor*)
 */
Tensor* sparse_csr_matmul(SparseCSRData* csr, Tensor* dense) {
    if (!csr || !dense) {
        LOG_ERROR("sparse_csr_matmul: NULL argument");
        return NULL;
    }
    tensor_ensure_executed(dense);
    if (dense->ndim != 2) {
        LOG_ERROR("sparse_csr_matmul: dense must be 2D");
        return NULL;
    }
    if (csr->cols != dense->shape[0]) {
        LOG_ERROR("sparse_csr_matmul: dimension mismatch - CSR cols (%d) != dense rows (%d)",
                  csr->cols, dense->shape[0]);
        return NULL;
    }

    bool any_grad = dense->requires_grad || (csr->values && csr->values->requires_grad);
    if (any_grad) {
        /* the view shares csr->values, so the COO SpMM VJP lands grad on csr->values->grad */
        SparseCOOData* view = csr_to_coo_view(csr);
        if (!view) {
            LOG_ERROR("sparse_csr_matmul: failed to build COO view for autograd");
            return NULL;
        }
        Tensor* out = sparse_matmul(view, dense);
        csr_coo_view_free(view);
        return out;
    }

    int M = csr->rows, N = dense->shape[1];
    int out_shape[]     = {M, N};
    TensorConfig config = (TensorConfig){
        .dtype = dense->dtype, .device = dense->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 2, &config);
    if (!output) {
        LOG_ERROR("sparse_csr_matmul: failed to create output");
        return NULL;
    }
    if (csr->nnz == 0)
        return output;

    const int32_t *crow_d, *cidx_d;
    const float* vals_d;
    csr_arrays(csr, &crow_d, &cidx_d, &vals_d);
    tensor_ensure_executed(output);
    const float* dense_d = (const float*)dense->data;
    float* out_d         = (float*)output->data;

    for (int r = 0; r < M; r++) {
        for (int p = crow_d[r]; p < crow_d[r + 1]; p++) {
            int col = cidx_d[p];
            float v = vals_d[p];
            for (int n = 0; n < N; n++)
                out_d[r * N + n] += v * dense_d[col * N + n];
        }
    }
    return output;
}

/**
 * CSR matrix times dense vector (eager).
 * @param csr (SparseCSRData*)
 * @param vec (Tensor*)
 * @return (Tensor*)
 */
Tensor* sparse_mv(SparseCSRData* csr, Tensor* vec) {
    if (!csr || !vec) {
        LOG_ERROR("sparse_mv: NULL argument");
        return NULL;
    }
    tensor_ensure_executed(vec);
    if (vec->ndim != 1 || vec->shape[0] != csr->cols) {
        LOG_ERROR("sparse_mv: vec must be 1D [cols]");
        return NULL;
    }

    int out_shape[]     = {csr->rows};
    TensorConfig config = (TensorConfig){
        .dtype = vec->dtype, .device = vec->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 1, &config);
    if (!output)
        return NULL;
    if (csr->nnz == 0)
        return output;

    const int32_t *crow_d, *cidx_d;
    const float* vals_d;
    csr_arrays(csr, &crow_d, &cidx_d, &vals_d);
    tensor_ensure_executed(output);
    const float* vec_d = (const float*)vec->data;
    float* out_d       = (float*)output->data;
    for (int r = 0; r < csr->rows; r++) {
        float acc = 0.0f;
        for (int p = crow_d[r]; p < crow_d[r + 1]; p++)
            acc += vals_d[p] * vec_d[cidx_d[p]];
        out_d[r] = acc;
    }
    return output;
}

/**
 * Element-wise sum of two equally-shaped CSR matrices (per-row two-pointer merge).
 * @param a (SparseCSRData*)
 * @param b (SparseCSRData*)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_add(SparseCSRData* a, SparseCSRData* b) {
    if (!a || !b) {
        LOG_ERROR("sparse_add: NULL argument");
        return NULL;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        LOG_ERROR("sparse_add: shape mismatch");
        return NULL;
    }
    int rows = a->rows, cols = a->cols;

    const int32_t *a_crow, *a_cidx, *b_crow, *b_cidx;
    const float *a_val, *b_val;
    csr_arrays(a, &a_crow, &a_cidx, &a_val);
    csr_arrays(b, &b_crow, &b_cidx, &b_val);

    int cap         = a->nnz + b->nnz;
    int32_t* crow_d = cml_calloc(rows + 1, sizeof(int32_t));
    int32_t* cidx_d = cap > 0 ? cml_malloc((size_t)cap * sizeof(int32_t)) : NULL;
    float* vals_d   = cap > 0 ? cml_malloc((size_t)cap * sizeof(float)) : NULL;
    if (!crow_d || (cap > 0 && (!cidx_d || !vals_d))) {
        cml_free(crow_d);
        cml_free(cidx_d);
        cml_free(vals_d);
        LOG_ERROR("sparse_add: allocation failed");
        return NULL;
    }

    int out = 0;
    for (int r = 0; r < rows; r++) {
        int i = a_crow[r], ie = a_crow[r + 1];
        int j = b_crow[r], je = b_crow[r + 1];
        while (i < ie && j < je) {
            if (a_cidx[i] < b_cidx[j]) {
                cidx_d[out] = a_cidx[i];
                vals_d[out] = a_val[i];
                out++;
                i++;
            } else if (a_cidx[i] > b_cidx[j]) {
                cidx_d[out] = b_cidx[j];
                vals_d[out] = b_val[j];
                out++;
                j++;
            } else {
                float s = a_val[i] + b_val[j];
                if (s != 0.0f) {
                    cidx_d[out] = a_cidx[i];
                    vals_d[out] = s;
                    out++;
                }
                i++;
                j++;
            }
        }
        while (i < ie) {
            cidx_d[out] = a_cidx[i];
            vals_d[out] = a_val[i];
            out++;
            i++;
        }
        while (j < je) {
            cidx_d[out] = b_cidx[j];
            vals_d[out] = b_val[j];
            out++;
            j++;
        }
        crow_d[r + 1] = out;
    }

    Tensor* crow          = make_i32_1d(rows + 1, a->crow_indices->device);
    Tensor* cidx          = make_i32_1d(out, a->col_indices->device);
    Tensor* vals          = make_f32_1d(out, a->values->device);
    SparseCSRData* result = NULL;
    if (crow && cidx && vals) {
        memcpy(crow->data, crow_d, (size_t)(rows + 1) * sizeof(int32_t));
        if (out > 0) {
            memcpy(cidx->data, cidx_d, (size_t)out * sizeof(int32_t));
            memcpy(vals->data, vals_d, (size_t)out * sizeof(float));
        }
        result = sparse_csr_create(crow, cidx, vals, rows, cols);
    }
    if (crow)
        tensor_free(crow);
    if (cidx)
        tensor_free(cidx);
    if (vals)
        tensor_free(vals);
    cml_free(crow_d);
    cml_free(cidx_d);
    cml_free(vals_d);
    if (!result)
        LOG_ERROR("sparse_add: failed to build result");
    return result;
}

/**
 * Scale all stored values by a scalar, returning a new CSR with identical structure.
 * @param csr (SparseCSRData*)
 * @param scalar (float)
 * @return (SparseCSRData*)
 */
SparseCSRData* sparse_mul_scalar(SparseCSRData* csr, float scalar) {
    if (!csr) {
        LOG_ERROR("sparse_mul_scalar: NULL input");
        return NULL;
    }
    Tensor* vals = make_f32_1d(csr->nnz, csr->values->device);
    if (!vals) {
        LOG_ERROR("sparse_mul_scalar: allocation failed");
        return NULL;
    }
    if (csr->nnz > 0) {
        tensor_ensure_executed(csr->values);
        const float* src = (const float*)csr->values->data;
        float* dst       = (float*)vals->data;
        for (int i = 0; i < csr->nnz; i++)
            dst[i] = src[i] * scalar;
    }
    SparseCSRData* result =
        sparse_csr_create(csr->crow_indices, csr->col_indices, vals, csr->rows, csr->cols);
    tensor_free(vals);
    return result;
}

/**
 * Transpose a CSR [M, N] into the CSC of its [N, M] transpose (O(1), structural).
 * @param csr (SparseCSRData*)
 * @return (SparseCSCData*)
 */
SparseCSCData* sparse_transpose(SparseCSRData* csr) {
    if (!csr) {
        LOG_ERROR("sparse_transpose: NULL input");
        return NULL;
    }
    /* A^T's CSC column pointers are exactly A's crow_indices; its rows are A's columns */
    return sparse_csc_create(csr->crow_indices, csr->col_indices, csr->values, csr->cols,
                             csr->rows);
}

/**
 * Sum CSR entries, optionally over a dimension (see header).
 * @param csr (SparseCSRData*)
 * @param dim (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* sparse_sum(SparseCSRData* csr, int dim, const TensorConfig* config) {
    if (!csr) {
        LOG_ERROR("sparse_sum: NULL input");
        return NULL;
    }
    int out_len    = dim < 0 ? 1 : (dim == 0 ? csr->cols : csr->rows);
    int shape[]    = {out_len};
    Tensor* output = tensor_zeros(shape, 1, config);
    if (!output) {
        LOG_ERROR("sparse_sum: failed to create output");
        return NULL;
    }
    if (csr->nnz == 0)
        return output;

    const int32_t *crow_d, *cidx_d;
    const float* vals_d;
    csr_arrays(csr, &crow_d, &cidx_d, &vals_d);
    tensor_ensure_executed(output);
    float* out_d = (float*)output->data;

    for (int r = 0; r < csr->rows; r++) {
        for (int p = crow_d[r]; p < crow_d[r + 1]; p++) {
            if (dim < 0)
                out_d[0] += vals_d[p];
            else if (dim == 0)
                out_d[cidx_d[p]] += vals_d[p];
            else
                out_d[r] += vals_d[p];
        }
    }
    return output;
}

/**
 * Number of stored entries.
 * @param csr (const SparseCSRData*)
 * @return (int)
 */
int sparse_csr_nnz(const SparseCSRData* csr) { return csr ? csr->nnz : 0; }

/**
 * Whether the handle is a (non-NULL) sparse tensor.
 * @param csr (const SparseCSRData*)
 * @return (bool)
 */
bool sparse_csr_is_sparse(const SparseCSRData* csr) { return csr != NULL; }

/**
 * Fraction of entries that are stored (nnz / rows*cols).
 * @param csr (const SparseCSRData*)
 * @return (double)
 */
double sparse_csr_density(const SparseCSRData* csr) {
    if (!csr)
        return 0.0;
    double total = (double)csr->rows * (double)csr->cols;
    return total > 0.0 ? (double)csr->nnz / total : 0.0;
}

/**
 * Free a CSR tensor: its owned index/value tensors and the struct.
 * @param csr (SparseCSRData*)
 */
void sparse_csr_free(SparseCSRData* csr) {
    if (!csr)
        return;
    if (csr->crow_indices)
        tensor_free(csr->crow_indices);
    if (csr->col_indices)
        tensor_free(csr->col_indices);
    if (csr->values)
        tensor_free(csr->values);
    cml_free(csr);
}

/**
 * Free a CSC tensor: its owned index/value tensors and the struct.
 * @param csc (SparseCSCData*)
 */
void sparse_csc_free(SparseCSCData* csc) {
    if (!csc)
        return;
    if (csc->ccol_indices)
        tensor_free(csc->ccol_indices);
    if (csc->row_indices)
        tensor_free(csc->row_indices);
    if (csc->values)
        tensor_free(csc->values);
    cml_free(csc);
}
