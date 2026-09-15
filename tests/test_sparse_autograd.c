/* Sparse matmul (spMM) autograd coverage.
 *
 * C = A_coo @ B: gradients must flow to BOTH the dense operand B and the
 * sparse COO values (dense-grad semantics: values accumulates an [nnz]
 * dense gradient). Every analytic gradient is cross-checked against a
 * central finite difference of the full library forward, so the backward
 * kernels cannot drift from the forward kernel they adjoint.
 *
 * The default engine is the graph-level lazy autodiff; the eager
 * cpu_backward_node engine (GRAD_MODE=eager) is verified in a forked child
 * because the mode selector caches on first use.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cml.h"
#include "ops/ir/autodiff.h"
#include "ops/uops.h"
#include "tensor/sparse_tensor.h"
#include "test_harness.h"

static TensorConfig f32cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                              .has_dtype = true, .has_device = true};
static TensorConfig i32cfg = {.dtype = DTYPE_INT32, .device = DEVICE_CPU,
                              .has_dtype = true, .has_device = true};

/* ── fixtures ──────────────────────────────────────────────────────────── */
#define M 4
#define K 3
#define N 2
#define NNZ 6

/* A = [[0,2,0],[1,0,0],[0,0,-3],[1,1,0]] — entry 3 duplicates row 0's col 1 */
static const int32_t g_rows[NNZ] = {0, 1, 2, 3, 3, 0};
static const int32_t g_cols[NNZ] = {1, 0, 2, 0, 1, 1};

static void fill_f32(Tensor* t, const float* xs, size_t n) {
    memcpy(tensor_data_ptr(t), xs, sizeof(float) * n);
}

typedef struct SpmmFixture {
    Tensor* indices;
    Tensor* values;
    Tensor* B;
    SparseCOOData* sp;
    Tensor* C;
    Tensor* loss;
} SpmmFixture;

static void fixture_free(SpmmFixture* fx) {
    if (fx->sp) sparse_free(fx->sp);
    if (fx->indices) tensor_free(fx->indices);
    if (fx->values) tensor_free(fx->values);
    if (fx->B) tensor_free(fx->B);
    memset(fx, 0, sizeof(*fx));
}

/* Build A(indices, values) @ B and run backward on sum(C). Values and B are
 * marked requires_grad. Returns 0 on any failure. Grads are left LAZY under
 * the graph engine — realize them with tensor_ensure_executed before reading. */
static int run_spmm_backward(SpmmFixture* fx, const float* xs, const float* bs,
                             bool create_graph) {
    memset(fx, 0, sizeof(*fx));
    int idx_shape[2] = {NNZ, 2};
    fx->indices = tensor_zeros(idx_shape, 2, &i32cfg);
    int val_shape[1] = {NNZ};
    fx->values = tensor_zeros(val_shape, 1, &f32cfg);
    int b_shape[2] = {K, N};
    fx->B = tensor_zeros(b_shape, 2, &f32cfg);
    if (!fx->indices || !fx->values || !fx->B) return 0;

    int32_t* idx_data = (int32_t*)tensor_data_ptr(fx->indices);
    for (int m = 0; m < NNZ; m++) {
        idx_data[m * 2 + 0] = g_rows[m];
        idx_data[m * 2 + 1] = g_cols[m];
    }
    fill_f32(fx->values, xs, NNZ);
    fill_f32(fx->B, bs, K * N);
    fx->values->requires_grad = true;
    fx->B->requires_grad = true;

    int dense_shape[2] = {M, K};
    fx->sp = sparse_coo_tensor(fx->indices, fx->values, dense_shape, 2);
    if (!fx->sp) return 0;
    fx->C = sparse_matmul(fx->sp, fx->B);
    if (!fx->C) return 0;

    ReduceParams rp = {0};
    fx->loss = uop_sum(fx->C, &rp);
    if (!fx->loss) return 0;
    tensor_backward(fx->loss, NULL, false, create_graph);

    tensor_ensure_executed(fx->B->grad);
    tensor_ensure_executed(fx->sp->values->grad);
    return 1;
}

/* Reference forward: plain triple loop, independent of the IR machinery. */
static float ref_sum_spmm(const float* xs, const float* bs) {
    float total = 0.0f;
    for (int m = 0; m < NNZ; m++) {
        float acc = 0.0f;
        for (int n = 0; n < N; n++)
            acc += xs[m] * bs[g_cols[m] * N + n];
        total += (g_rows[m] >= 0 && g_rows[m] < M) ? acc : 0.0f;
    }
    return total;
}

static int vec_close(const float* got, const float* want, size_t n, float tol) {
    for (size_t i = 0; i < n; i++) {
        float den = fmaxf(1.0f, fmaxf(fabsf(got[i]), fabsf(want[i])));
        if (fabsf(got[i] - want[i]) / den > tol) {
            printf("\n    MISMATCH at %zu: got %.6f want %.6f", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

/* ── shared numeric checks (run under both engines) ────────────────────── */
static int check_grad_b(float tol) {
    float xs[NNZ], bs[K * N];
    for (int i = 0; i < NNZ; i++) xs[i] = 0.4f - 0.13f * i;
    for (int i = 0; i < K * N; i++) bs[i] = 0.25f + 0.11f * i;

    cml_reset_ir_context();
    SpmmFixture fx;
    if (!run_spmm_backward(&fx, xs, bs, false)) { fixture_free(&fx); return 0; }

    /* d sum(C)/dB[k,n] = sum_m A[k_col?] : column sums weighted by values */
    float want[K * N];
    memset(want, 0, sizeof(want));
    for (int m = 0; m < NNZ; m++)
        for (int n = 0; n < N; n++)
            want[g_cols[m] * N + n] += xs[m];

    int ok = fx.B->grad != NULL &&
             fx.B->grad->shape[0] == K && fx.B->grad->shape[1] == N &&
             vec_close((float*)tensor_data_ptr(fx.B->grad), want, K * N, tol);

    /* finite-difference cross-check on two representative entries */
    const float eps = 1e-3f;
    int probes[2] = {1, K * N - 2};
    for (int p = 0; p < 2 && ok; p++) {
        int e = probes[p];
        float saved = bs[e];
        bs[e] = saved + eps;
        float fp = ref_sum_spmm(xs, bs);
        bs[e] = saved - eps;
        float fm = ref_sum_spmm(xs, bs);
        bs[e] = saved;
        float num = (fp - fm) / (2.0f * eps);
        float ana  = ((float*)tensor_data_ptr(fx.B->grad))[e];
        float den  = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana)));
        if (fabsf(num - ana) / den > 0.05f) {
            printf("\n    FD mismatch at B[%d]: num=%.5f ana=%.5f", e, num, ana);
            ok = 0;
        }
    }
    fixture_free(&fx);
    cml_reset_ir_context();
    return ok;
}

static int check_grad_values(float tol) {
    float xs[NNZ], bs[K * N];
    for (int i = 0; i < NNZ; i++) xs[i] = 0.7f - 0.17f * i;
    for (int i = 0; i < K * N; i++) bs[i] = -0.35f + 0.19f * i;

    cml_reset_ir_context();
    SpmmFixture fx;
    if (!run_spmm_backward(&fx, xs, bs, false)) { fixture_free(&fx); return 0; }

    if (!fx.sp->values->grad || fx.sp->values->grad->shape[0] != NNZ) {
        printf("\n    values grad missing or wrong shape");
        fixture_free(&fx);
        return 0;
    }

    /* duplicate coordinates accumulate into every duplicate entry's grad */
    float want[NNZ];
    for (int m = 0; m < NNZ; m++) {
        want[m] = 0.0f;
        for (int n = 0; n < N; n++)
            want[m] += bs[g_cols[m] * N + n];
    }
    int ok = vec_close((float*)tensor_data_ptr(fx.sp->values->grad), want, NNZ, tol);

    const float eps = 1e-3f;
    for (int m = 0; m < NNZ && ok; m++) {
        float saved = xs[m];
        xs[m] = saved + eps;
        float fp = ref_sum_spmm(xs, bs);
        xs[m] = saved - eps;
        float fm = ref_sum_spmm(xs, bs);
        xs[m] = saved;
        float num = (fp - fm) / (2.0f * eps);
        float ana = ((float*)tensor_data_ptr(fx.sp->values->grad))[m];
        float den = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana)));
        if (fabsf(num - ana) / den > 0.05f) {
            printf("\n    FD mismatch at values[%d]: num=%.5f ana=%.5f", m, num, ana);
            ok = 0;
        }
    }
    fixture_free(&fx);
    cml_reset_ir_context();
    return ok;
}

/* ── tests ─────────────────────────────────────────────────────────────── */
static int test_grad_b_analytic_and_fd(void) { return check_grad_b(1e-3f); }
static int test_grad_values_analytic_and_fd(void) { return check_grad_values(1e-3f); }

static int test_grad_shapes(void) {
    float xs[NNZ] = {0.5f, -0.25f, 0.75f, 1.0f, -0.5f, 0.125f};
    float bs[K * N];
    for (int i = 0; i < K * N; i++) bs[i] = 0.1f + 0.05f * i;

    cml_reset_ir_context();
    SpmmFixture fx;
    if (!run_spmm_backward(&fx, xs, bs, false)) { fixture_free(&fx); return 0; }

    int ok = fx.B->grad != NULL && fx.B->grad->ndim == 2 &&
             fx.B->grad->shape[0] == K && fx.B->grad->shape[1] == N &&
             fx.sp->values->grad != NULL && fx.sp->values->grad->ndim == 1 &&
             fx.sp->values->grad->shape[0] == NNZ &&
             fx.indices->grad == NULL; /* discrete coordinates take no grad */
    if (!ok)
        printf("\n    B grad %p / values grad %p shapes wrong",
               (void*)fx.B->grad, fx.sp->values ? (void*)fx.sp->values->grad : NULL);
    fixture_free(&fx);
    cml_reset_ir_context();
    return ok;
}

static int test_zero_density(void) {
    cml_reset_ir_context();

    /* Eager allocation: tensor_zeros would create a lazy FILL node, and the
     * executor has no typed kernel for a zero-element INT32 fill. */
    int idx_shape[2] = {0, 2};
    Tensor* indices = tensor_empty(idx_shape, 2, &i32cfg);
    int val_shape[1] = {0};
    Tensor* values = tensor_empty(val_shape, 1, &f32cfg);
    int b_shape[2] = {K, N};
    Tensor* B = tensor_zeros(b_shape, 2, &f32cfg);
    if (!indices || !values || !B) return 0;
    values->requires_grad = true;
    B->requires_grad = true;

    int dense_shape[2] = {M, K};
    SparseCOOData* sp = sparse_coo_tensor(indices, values, dense_shape, 2);
    Tensor* C = sp ? sparse_matmul(sp, B) : NULL;
    ReduceParams rp = {0};
    Tensor* loss = C ? uop_sum(C, &rp) : NULL;
    int ok = loss != NULL;
    if (ok) {
        tensor_backward(loss, NULL, false, false);
        tensor_ensure_executed(B->grad);
        tensor_ensure_executed(sp->values->grad);
        ok = B->grad != NULL && sp->values->grad != NULL;
        if (ok) {
            float* gb = (float*)tensor_data_ptr(B->grad);
            for (int i = 0; i < K * N && ok; i++) ok = gb[i] == 0.0f;
        }
    }
    printf("%s", ok ? "" : "\n    zero-density backward failed");
    if (sp) sparse_free(sp);
    tensor_free(indices);
    tensor_free(values);
    tensor_free(B);
    cml_reset_ir_context();
    return ok;
}

/* Eager-engine spot check: GRAD_MODE is cached process-wide, so verify the
 * cpu_backward_node path in a forked child. */
static int test_eager_engine_child(void) {
    pid_t pid = fork();
    if (pid < 0) return 0;
    if (pid > 0) {
        int status = 1;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    /* child */
    setenv("GRAD_MODE", "eager", 1);
    int ok = check_grad_b(1e-3f) && check_grad_values(1e-3f);
    fflush(stdout);
    _exit(ok ? 0 : 1);
}

int main(void) {
    printf("sparse matmul autograd:\n");
    TEST(grad_b_analytic_and_fd);
    TEST(grad_values_analytic_and_fd);
    TEST(grad_shapes);
    TEST(zero_density);
    TEST(eager_engine_child);
    return TEST_SUMMARY();
}
