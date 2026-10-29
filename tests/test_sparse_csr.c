/* CSR / CSC sparse format coverage.
 *
 * Builds a known 4x3 sparse matrix and checks that CSR <-> COO <-> dense
 * round-trips are exact, that sparse_csr_matmul matches the equivalent dense
 * matmul, and that sparse_add / sparse_transpose / sparse_sum / sparse_mv
 * behave. SpMM's gradient wrt the dense operand is cross-checked against a
 * central finite difference of the library forward.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "ops/uops.h"
#include "tensor/sparse_tensor.h"
#include "test_harness.h"
#include "test_require.h"

static TensorConfig f32cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
static TensorConfig i32cfg = {
    .dtype = DTYPE_INT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* A = [[0, 2, 0],
 *      [1, 0, 0],
 *      [0, 0,-3],
 *      [4, 5, 0]]  -> 4x3, nnz=5 */
#define ROWS 4
#define COLS 3
#define NNZ 5
static const int32_t g_crow[ROWS + 1] = {0, 1, 2, 3, 5};
static const int32_t g_col[NNZ]       = {1, 0, 2, 0, 1};
static const float g_val[NNZ]         = {2.0f, 1.0f, -3.0f, 4.0f, 5.0f};
static const float g_dense[ROWS * COLS] = {
    0.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -3.0f, 4.0f, 5.0f, 0.0f};

static Tensor* i32_vec(const int32_t* xs, int n) {
    int shape[] = {n};
    Tensor* t   = tensor_zeros(shape, 1, &i32cfg);
    REQUIRE(t != NULL);
    memcpy(tensor_data_ptr(t), xs, sizeof(int32_t) * n);
    return t;
}

static Tensor* f32_vec(const float* xs, int n) {
    int shape[] = {n};
    Tensor* t   = tensor_zeros(shape, 1, &f32cfg);
    REQUIRE(t != NULL);
    memcpy(tensor_data_ptr(t), xs, sizeof(float) * n);
    return t;
}

static SparseCSRData* make_csr(void) {
    Tensor* crow    = i32_vec(g_crow, ROWS + 1);
    Tensor* col     = i32_vec(g_col, NNZ);
    Tensor* val     = f32_vec(g_val, NNZ);
    SparseCSRData* c = sparse_csr_create(crow, col, val, ROWS, COLS);
    tensor_free(crow);
    tensor_free(col);
    tensor_free(val);
    return c;
}

static int dense_eq(const float* got, const float* want, int n) {
    for (int i = 0; i < n; i++)
        if (fabsf(got[i] - want[i]) > 1e-5f) {
            printf("\n    mismatch at %d: got %.5f want %.5f", i, got[i], want[i]);
            return 0;
        }
    return 1;
}

static int test_csr_to_dense_exact(void) {
    SparseCSRData* c = make_csr();
    REQUIRE(c != NULL);
    REQUIRE(sparse_csr_nnz(c) == NNZ);
    Tensor* d = sparse_csr_to_dense(c, &f32cfg);
    REQUIRE(d != NULL);
    int ok = d->ndim == 2 && d->shape[0] == ROWS && d->shape[1] == COLS &&
             dense_eq((float*)tensor_data_ptr(d), g_dense, ROWS * COLS);
    tensor_free(d);
    sparse_csr_free(c);
    return ok;
}

static int test_dense_to_csr_roundtrip(void) {
    int shape[] = {ROWS, COLS};
    Tensor* d   = tensor_zeros(shape, 2, &f32cfg);
    REQUIRE(d != NULL);
    memcpy(tensor_data_ptr(d), g_dense, sizeof(g_dense));
    SparseCSRData* c = dense_to_sparse_csr(d);
    REQUIRE(c != NULL);
    REQUIRE(sparse_csr_nnz(c) == NNZ);
    Tensor* back = sparse_csr_to_dense(c, &f32cfg);
    int ok = dense_eq((float*)tensor_data_ptr(back), g_dense, ROWS * COLS);
    tensor_free(back);
    sparse_csr_free(c);
    tensor_free(d);
    return ok;
}

static int test_csr_coo_roundtrip(void) {
    SparseCSRData* c = make_csr();
    SparseCOOData* coo = sparse_csr_to_coo(c);
    REQUIRE(coo != NULL && coo->nnz == NNZ);
    Tensor* d = sparse_to_dense(coo, &f32cfg);
    int ok1 = dense_eq((float*)tensor_data_ptr(d), g_dense, ROWS * COLS);
    /* COO -> CSR back */
    SparseCSRData* c2 = sparse_coo_to_csr(coo);
    REQUIRE(c2 != NULL);
    Tensor* d2 = sparse_csr_to_dense(c2, &f32cfg);
    int ok2 = dense_eq((float*)tensor_data_ptr(d2), g_dense, ROWS * COLS);
    tensor_free(d);
    tensor_free(d2);
    sparse_free(coo);
    sparse_csr_free(c2);
    sparse_csr_free(c);
    return ok1 && ok2;
}

static int test_csr_to_csc(void) {
    SparseCSRData* c   = make_csr();
    SparseCSCData* csc = sparse_csr_to_csc(c);
    REQUIRE(csc != NULL && csc->rows == ROWS && csc->cols == COLS && csc->nnz == NNZ);
    /* Reconstruct dense from CSC and compare. */
    float got[ROWS * COLS];
    memset(got, 0, sizeof(got));
    tensor_ensure_executed(csc->ccol_indices);
    tensor_ensure_executed(csc->row_indices);
    tensor_ensure_executed(csc->values);
    const int32_t* ccol = (const int32_t*)tensor_data_ptr(csc->ccol_indices);
    const int32_t* ridx = (const int32_t*)tensor_data_ptr(csc->row_indices);
    const float* vals   = (const float*)tensor_data_ptr(csc->values);
    for (int col = 0; col < COLS; col++)
        for (int p = ccol[col]; p < ccol[col + 1]; p++)
            got[ridx[p] * COLS + col] += vals[p];
    int ok = dense_eq(got, g_dense, ROWS * COLS);
    sparse_csc_free(csc);
    sparse_csr_free(c);
    return ok;
}

static int test_csr_matmul_vs_dense(void) {
    cml_reset_ir_context();
    SparseCSRData* c = make_csr();
    int b_shape[]    = {COLS, 2};
    float bs[COLS * 2] = {1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 4.0f};
    Tensor* B        = tensor_zeros(b_shape, 2, &f32cfg);
    memcpy(tensor_data_ptr(B), bs, sizeof(bs));

    Tensor* C = sparse_csr_matmul(c, B);
    REQUIRE(C != NULL && C->shape[0] == ROWS && C->shape[1] == 2);

    /* equivalent dense matmul */
    Tensor* Adense = sparse_csr_to_dense(c, &f32cfg);
    Tensor* ref    = uop_matmul(Adense, B);
    REQUIRE(ref != NULL);
    int ok = dense_eq((float*)tensor_data_ptr(C), (float*)tensor_data_ptr(ref), ROWS * 2);

    tensor_free(C);
    tensor_free(ref);
    tensor_free(Adense);
    tensor_free(B);
    sparse_csr_free(c);
    cml_reset_ir_context();
    return ok;
}

static int test_sparse_add(void) {
    SparseCSRData* a   = make_csr();
    SparseCSRData* b   = make_csr();
    SparseCSRData* sum = sparse_add(a, b);
    REQUIRE(sum != NULL);
    Tensor* d = sparse_csr_to_dense(sum, &f32cfg);
    float want[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++)
        want[i] = 2.0f * g_dense[i];
    int ok = dense_eq((float*)tensor_data_ptr(d), want, ROWS * COLS);
    tensor_free(d);
    sparse_csr_free(sum);
    sparse_csr_free(b);
    sparse_csr_free(a);
    return ok;
}

static int test_mul_scalar(void) {
    SparseCSRData* a = make_csr();
    SparseCSRData* s = sparse_mul_scalar(a, -0.5f);
    REQUIRE(s != NULL);
    Tensor* d = sparse_csr_to_dense(s, &f32cfg);
    float want[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++)
        want[i] = -0.5f * g_dense[i];
    int ok = dense_eq((float*)tensor_data_ptr(d), want, ROWS * COLS);
    tensor_free(d);
    sparse_csr_free(s);
    sparse_csr_free(a);
    return ok;
}

static int test_transpose(void) {
    SparseCSRData* c   = make_csr();
    SparseCSCData* csc = sparse_transpose(c);
    REQUIRE(csc != NULL && csc->rows == COLS && csc->cols == ROWS);
    /* The CSC-of-transpose uses A's row pointers as its column pointers, so
     * reconstructing with [col over A-rows, row = A-col] yields A^T. */
    float got[COLS * ROWS];
    memset(got, 0, sizeof(got));
    const int32_t* ccol = (const int32_t*)tensor_data_ptr(csc->ccol_indices);
    const int32_t* ridx = (const int32_t*)tensor_data_ptr(csc->row_indices);
    const float* vals   = (const float*)tensor_data_ptr(csc->values);
    for (int col = 0; col < ROWS; col++)
        for (int p = ccol[col]; p < ccol[col + 1]; p++)
            got[ridx[p] * ROWS + col] += vals[p]; /* AT[ridx, col] */
    float want[COLS * ROWS];
    for (int r = 0; r < ROWS; r++)
        for (int cc = 0; cc < COLS; cc++)
            want[cc * ROWS + r] = g_dense[r * COLS + cc];
    int ok = dense_eq(got, want, COLS * ROWS);
    sparse_csc_free(csc);
    sparse_csr_free(c);
    return ok;
}

static int test_sum(void) {
    SparseCSRData* c = make_csr();
    Tensor* total    = sparse_sum(c, -1, &f32cfg);
    Tensor* col_sum  = sparse_sum(c, 0, &f32cfg);
    Tensor* row_sum  = sparse_sum(c, 1, &f32cfg);
    REQUIRE(total && col_sum && row_sum);

    float want_total = 0.0f, want_col[COLS] = {0}, want_row[ROWS] = {0};
    for (int r = 0; r < ROWS; r++)
        for (int cc = 0; cc < COLS; cc++) {
            float v = g_dense[r * COLS + cc];
            want_total += v;
            want_col[cc] += v;
            want_row[r] += v;
        }
    int ok = total->shape[0] == 1 && col_sum->shape[0] == COLS && row_sum->shape[0] == ROWS &&
             dense_eq((float*)tensor_data_ptr(total), &want_total, 1) &&
             dense_eq((float*)tensor_data_ptr(col_sum), want_col, COLS) &&
             dense_eq((float*)tensor_data_ptr(row_sum), want_row, ROWS);
    tensor_free(total);
    tensor_free(col_sum);
    tensor_free(row_sum);
    sparse_csr_free(c);
    return ok;
}

static int test_mv(void) {
    SparseCSRData* c = make_csr();
    float vs[COLS]   = {2.0f, -1.0f, 0.5f};
    Tensor* v        = f32_vec(vs, COLS);
    Tensor* out      = sparse_mv(c, v);
    REQUIRE(out != NULL && out->shape[0] == ROWS);
    float want[ROWS] = {0};
    for (int r = 0; r < ROWS; r++)
        for (int cc = 0; cc < COLS; cc++)
            want[r] += g_dense[r * COLS + cc] * vs[cc];
    int ok = dense_eq((float*)tensor_data_ptr(out), want, ROWS);
    tensor_free(out);
    tensor_free(v);
    sparse_csr_free(c);
    return ok;
}

/* reference forward: sum(CSR @ B) */
static float ref_sum_spmm(const float* bs, int bn) {
    float total = 0.0f;
    for (int r = 0; r < ROWS; r++)
        for (int p = g_crow[r]; p < g_crow[r + 1]; p++)
            for (int n = 0; n < bn; n++)
                total += g_val[p] * bs[g_col[p] * bn + n];
    return total;
}

static int test_spmm_grad_dense_fd(void) {
    const int BN = 2;
    cml_reset_ir_context();
    SparseCSRData* c = make_csr();
    int b_shape[]    = {COLS, BN};
    float bs[COLS * 2];
    for (int i = 0; i < COLS * BN; i++)
        bs[i] = 0.3f + 0.17f * i;
    Tensor* B = tensor_zeros(b_shape, 2, &f32cfg);
    memcpy(tensor_data_ptr(B), bs, sizeof(float) * COLS * BN);
    B->requires_grad = true;

    Tensor* C = sparse_csr_matmul(c, B);
    REQUIRE(C != NULL);
    ReduceParams rp = {0};
    Tensor* loss    = uop_sum(C, &rp);
    REQUIRE(loss != NULL);
    tensor_backward(loss, NULL, false, false);
    tensor_ensure_executed(B->grad);
    REQUIRE(B->grad != NULL && B->grad->shape[0] == COLS && B->grad->shape[1] == BN);

    /* analytic grad_B[k,n] = sum over entries with col==k of value */
    float want[COLS * 2];
    memset(want, 0, sizeof(want));
    for (int r = 0; r < ROWS; r++)
        for (int p = g_crow[r]; p < g_crow[r + 1]; p++)
            for (int n = 0; n < BN; n++)
                want[g_col[p] * BN + n] += g_val[p];
    int ok = dense_eq((float*)tensor_data_ptr(B->grad), want, COLS * BN);

    /* finite-difference cross-check on two entries */
    const float eps = 1e-3f;
    int probes[2]   = {1, COLS * BN - 1};
    for (int pi = 0; pi < 2 && ok; pi++) {
        int e       = probes[pi];
        float saved = bs[e];
        bs[e]       = saved + eps;
        float fp    = ref_sum_spmm(bs, BN);
        bs[e]       = saved - eps;
        float fm    = ref_sum_spmm(bs, BN);
        bs[e]       = saved;
        float num   = (fp - fm) / (2.0f * eps);
        float ana   = ((float*)tensor_data_ptr(B->grad))[e];
        float den   = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana)));
        if (fabsf(num - ana) / den > 0.05f) {
            printf("\n    FD mismatch at B[%d]: num=%.5f ana=%.5f", e, num, ana);
            ok = 0;
        }
    }
    tensor_free(B);
    sparse_csr_free(c);
    cml_reset_ir_context();
    return ok;
}

int main(void) {
    printf("sparse CSR/CSC:\n");
    TEST(csr_to_dense_exact);
    TEST(dense_to_csr_roundtrip);
    TEST(csr_coo_roundtrip);
    TEST(csr_to_csc);
    TEST(csr_matmul_vs_dense);
    TEST(sparse_add);
    TEST(mul_scalar);
    TEST(transpose);
    TEST(sum);
    TEST(mv);
    TEST(spmm_grad_dense_fd);
    return TEST_SUMMARY();
}
