/* Per-op gradient coverage for the graph autodiff.
 *
 * grad_check.c verifies whole layers, which is why 34 ops could sit in
 * cml_ir_grad with no VJP at all and nothing noticed: a model using erf or
 * cumsum trained with no gradient, no error, and a green test suite. This
 * checks ops one at a time, and checks the value rather than the presence --
 * a wrong gradient is worse than a missing one.
 *
 * Every rule is compared against a central finite difference of the op's own
 * forward, so the test cannot drift with the implementation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "test_harness.h"

static TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};

typedef Tensor* (*OpFn)(Tensor*);
static int   g_rows = 1, g_cols = 6;
static float g_lo = 0.35f, g_hi = 0.95f;

static float forward_sum(OpFn op, const float* xs, int n) {
    cml_reset_ir_context();
    int sh[2] = { g_rows, g_cols };
    Tensor* x = tensor_zeros(sh, g_rows > 1 ? 2 : 1, &cfg);
    if (!x) return NAN;
    memcpy(tensor_data_ptr(x), xs, sizeof(float) * (size_t)n);
    Tensor* y = op(x);
    if (!y) { cml_reset_ir_context(); return NAN; }
    ReduceParams rp = {0};
    Tensor* s = uop_sum(y, &rp);
    if (!s) { cml_reset_ir_context(); return NAN; }
    tensor_ensure_executed(s);
    float v = ((float*)tensor_data_ptr(s))[0];
    cml_reset_ir_context();
    return v;
}

/* Returns 1 when the analytic gradient matches a central difference. */
static int grad_matches(OpFn op) {
    const int n = g_rows * g_cols;
    float xs[64];
    for (int i = 0; i < n; i++) xs[i] = g_lo + (g_hi - g_lo) * (i + 0.5f) / n;

    cml_reset_ir_context();
    int sh[2] = { g_rows, g_cols };
    Tensor* x = tensor_zeros(sh, g_rows > 1 ? 2 : 1, &cfg);
    if (!x) return 0;
    memcpy(tensor_data_ptr(x), xs, sizeof(float) * (size_t)n);
    x->requires_grad = true;

    Tensor* y = op(x);
    if (!y) { cml_reset_ir_context(); return 0; }
    ReduceParams rp = {0};
    Tensor* s = uop_sum(y, &rp);
    if (!s) { cml_reset_ir_context(); return 0; }
    tensor_backward(s, NULL, false, false);
    if (!x->grad) { cml_reset_ir_context(); return 0; }

    float ana[64];
    memcpy(ana, (float*)tensor_data_ptr(x->grad), sizeof(float) * (size_t)n);
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < n; i++) {
        float saved = xs[i];
        xs[i] = saved + eps; float fp = forward_sum(op, xs, n);
        xs[i] = saved - eps; float fm = forward_sum(op, xs, n);
        xs[i] = saved;
        if (isnan(fp) || isnan(fm)) return 0;
        float num = (fp - fm) / (2.0f * eps);
        float den = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana[i])));
        if (fabsf(num - ana[i]) / den > 0.02f) return 0;
    }
    return 1;
}

/* Ops needing extra arguments get a wrapper with the sizes this test uses. */
static Tensor* w_hard_tanh(Tensor* a) { return uop_hard_tanh(a); }
static Tensor* w_relu6(Tensor* a)     { return uop_relu6(a); }
static Tensor* w_qgelu(Tensor* a)     { return uop_quick_gelu(a); }
static Tensor* w_triu(Tensor* a)      { return uop_triu(a, 0); }
static Tensor* w_tril(Tensor* a)      { return uop_tril(a, 1); }
static Tensor* w_roll(Tensor* a)      { return uop_roll(a, 1, 1); }
static Tensor* w_cumsum(Tensor* a)    { return uop_cumsum(a, 1); }
static Tensor* w_cumprod(Tensor* a)   { return uop_cumprod(a, 1); }
static Tensor* w_lcse(Tensor* a)      { return uop_logcumsumexp(a, 1); }
static Tensor* w_prod(Tensor* a)      { ReduceParams p = {0}; return uop_prod(a, &p); }
static Tensor* w_lse(Tensor* a)       { ReduceParams p = {0}; return uop_logsumexp(a, &p); }
static Tensor* w_trace(Tensor* a)     { return uop_trace(a); }
static Tensor* w_flatten(Tensor* a)   { return uop_flatten(a, 0, 1); }
static Tensor* w_tile(Tensor* a)      { int r[2] = {2, 3}; return uop_tile(a, r, 2); }
static Tensor* w_ri1(Tensor* a)       { return uop_repeat_interleave(a, 3, 1); }
static Tensor* w_ri0(Tensor* a)       { return uop_repeat_interleave(a, 2, 0); }
static Tensor* w_diagonal(Tensor* a)  { return uop_diagonal(a, 0, 0, 1); }
static Tensor* w_sort(Tensor* a)      { return uop_sort(a, 1, false); }
static Tensor* w_sort_desc(Tensor* a) { return uop_sort(a, 1, true); }
static Tensor* w_topk(Tensor* a)      { Tensor* i = NULL; return uop_topk(a, 2, 1, true, &i); }

/* ── 1-D elementwise: domain chosen so each op stays differentiable ─────── */
#define ELEMENTWISE(name, fn, lo, hi)                                                              \
    static int test_##name(void) {                                                                 \
        g_rows = 1; g_cols = 6; g_lo = (lo); g_hi = (hi);                                           \
        return grad_matches(fn);                                                                    \
    }
ELEMENTWISE(asin,        uop_asin,        -0.8f, 0.8f)
ELEMENTWISE(acos,        uop_acos,        -0.8f, 0.8f)
ELEMENTWISE(atan,        uop_atan,        -2.0f, 2.0f)
ELEMENTWISE(asinh,       uop_asinh,       -2.0f, 2.0f)
ELEMENTWISE(acosh,       uop_acosh,        1.5f, 3.0f)
ELEMENTWISE(atanh,       uop_atanh,       -0.7f, 0.7f)
ELEMENTWISE(erf,         uop_erf,         -1.5f, 1.5f)
ELEMENTWISE(sinh,        uop_sinh,        -1.5f, 1.5f)
ELEMENTWISE(cosh,        uop_cosh,        -1.5f, 1.5f)
ELEMENTWISE(log2,        uop_log2,         0.5f, 3.0f)
ELEMENTWISE(log10,       uop_log10,        0.5f, 3.0f)
ELEMENTWISE(exp2,        uop_exp2,        -1.5f, 1.5f)
ELEMENTWISE(hard_sigmoid, uop_hard_sigmoid, -2.0f, 2.0f)
ELEMENTWISE(hard_tanh,   w_hard_tanh,     -0.8f, 0.8f)
ELEMENTWISE(relu6,       w_relu6,          0.5f, 5.0f)
ELEMENTWISE(quick_gelu,  w_qgelu,         -1.5f, 1.5f)
ELEMENTWISE(softplus,    uop_softplus,    -1.5f, 1.5f)
ELEMENTWISE(softsign,    uop_softsign,    -1.5f, 1.5f)
ELEMENTWISE(logsigmoid,  uop_logsigmoid,  -1.5f, 1.5f)

/* ── 3x3: structural, reduction and cumulative rules ───────────────────── */
#define MATRIX(name, fn)                                                                           \
    static int test_##name(void) {                                                                 \
        g_rows = 3; g_cols = 3; g_lo = 0.35f; g_hi = 0.95f;                                         \
        return grad_matches(fn);                                                                    \
    }
MATRIX(triu,          w_triu)
MATRIX(tril,          w_tril)
MATRIX(roll,          w_roll)
MATRIX(flatten,       w_flatten)
MATRIX(cumsum,        w_cumsum)
MATRIX(cumprod,       w_cumprod)
MATRIX(logcumsumexp,  w_lcse)
MATRIX(prod,          w_prod)
MATRIX(logsumexp,     w_lse)
MATRIX(trace,         w_trace)
MATRIX(tile,          w_tile)
MATRIX(repeat_interleave_dim1, w_ri1)
MATRIX(repeat_interleave_dim0, w_ri0)
MATRIX(diagonal,      w_diagonal)
MATRIX(sort,          w_sort)
MATRIX(sort_descending, w_sort_desc)
MATRIX(topk,          w_topk)

/* Derivative is zero almost everywhere. What matters is that the chain
 * terminates with zeros rather than breaking: returning no gradient leaves
 * every parameter upstream of a rounded value silently untrained. */
static int zero_grad_op(OpFn op) {
    cml_reset_ir_context();
    int sh[1] = {6};
    Tensor* x = tensor_rand(sh, 1, &cfg);
    if (!x) return 0;
    x->requires_grad = true;
    Tensor* y = op(x);
    if (!y) { cml_reset_ir_context(); return 0; }
    ReduceParams rp = {0};
    tensor_backward(uop_sum(y, &rp), NULL, false, false);
    int ok = x->grad != NULL;
    if (ok) {
        float* g = (float*)tensor_data_ptr(x->grad);
        for (size_t i = 0; i < x->numel && ok; i++) if (g[i] != 0.0f) ok = 0;
    }
    cml_reset_ir_context();
    return ok;
}
static int test_floor_zero_grad(void) { return zero_grad_op(uop_floor); }
static int test_ceil_zero_grad(void)  { return zero_grad_op(uop_ceil); }
static int test_round_zero_grad(void) { return zero_grad_op(uop_round); }
static int test_sign_zero_grad(void)  { return zero_grad_op(uop_sign); }

/* Multi-axis reduce: both the shape inference and the kernels read only
 * dims[0], so a two-axis request used to reduce one axis and return a shape
 * claiming exactly that -- wrong numbers under a plausible shape. */
static int test_reduce_multi_axis(void) {
    cml_reset_ir_context();
    int sh[3] = {2, 2, 2};
    Tensor* a = tensor_zeros(sh, 3, &cfg);
    if (!a) return 0;
    float v[8]; for (int i = 0; i < 8; i++) v[i] = (float)(i + 1);
    memcpy(tensor_data_ptr(a), v, sizeof v);
    int dims[2] = {0, 2};
    ReduceParams p = { dims, 2, false };
    Tensor* r = uop_sum(a, &p);
    if (!r) { cml_reset_ir_context(); return 0; }
    tensor_ensure_executed(r);
    float* o = (float*)tensor_data_ptr(r);
    /* a[0]=[[1,2],[3,4]] a[1]=[[5,6],[7,8]] -> j=0: 1+2+5+6, j=1: 3+4+7+8 */
    int ok = r->numel == 2 && o && o[0] == 14.0f && o[1] == 22.0f;
    cml_reset_ir_context();
    return ok;
}

/* top-k over more than one dimension used to copy the first k values of the
 * flat buffer, ignoring the axis and leaving later lanes uninitialised. */
static int test_topk_values_and_indices(void) {
    cml_reset_ir_context();
    int sh[2] = {2, 3};
    Tensor* a = tensor_zeros(sh, 2, &cfg);
    if (!a) return 0;
    float v[6] = {3, 1, 2, 9, 7, 8};
    memcpy(tensor_data_ptr(a), v, sizeof v);
    Tensor* idx = NULL;
    Tensor* vals = uop_topk(a, 2, 1, true, &idx);
    if (!vals || !idx) { cml_reset_ir_context(); return 0; }
    tensor_ensure_executed(vals);
    tensor_ensure_executed(idx);
    float* vd = (float*)tensor_data_ptr(vals);
    float* id = (float*)tensor_data_ptr(idx);
    int ok = vd && id
          && vd[0] == 3 && vd[1] == 2 && vd[2] == 9 && vd[3] == 8
          && id[0] == 0 && id[1] == 2 && id[2] == 0 && id[3] == 2;
    cml_reset_ir_context();
    return ok;
}

/* Backward through a rank-increasing reshape followed by an axis reduce.
 *
 * This used to be a heap-use-after-free. uop_reshape() could only return a view
 * for a contiguous input; otherwise it fell back to tensor_contiguous(), which
 * clones, which calls tensor_ensure_executed() -- so constructing a reshape
 * quietly executed the graph. In the backward pass that graph is still being
 * assembled and the incoming gradient is routinely a non-contiguous expand
 * output, so the partial execution read buffers whose producers had not run or
 * had already been recycled. Every rank and every axis crashed. */
static int reshape_then_reduce_backward(int rank, int axis) {
    cml_reset_ir_context();
    int sh[2] = {3, 9};
    Tensor* a = tensor_rand(sh, 2, &cfg);
    if (!a) return 0;
    a->requires_grad = true;

    int v3[3] = {3, 3, 3}, v4[4] = {3, 3, 3, 1};
    Tensor* r = rank == 3 ? uop_reshape_to(a, v3, 3) : uop_reshape_to(a, v4, 4);
    if (!r) { cml_reset_ir_context(); return 0; }

    int d[1] = { axis };
    ReduceParams p = { d, 1, false };
    Tensor* s = uop_sum(r, &p);
    if (!s) { cml_reset_ir_context(); return 0; }

    ReduceParams rp = {0};
    Tensor* tot = uop_sum(s, &rp);
    if (!tot) { cml_reset_ir_context(); return 0; }
    tensor_backward(tot, NULL, false, false);

    /* Summing everything means every input contributes exactly once. */
    int ok = a->grad != NULL;
    if (ok) {
        float* g = (float*)tensor_data_ptr(a->grad);
        for (size_t i = 0; i < a->numel && ok; i++) if (fabsf(g[i] - 1.0f) > 1e-4f) ok = 0;
    }
    cml_reset_ir_context();
    return ok;
}
static int test_reshape3_reduce_axis0(void) { return reshape_then_reduce_backward(3, 0); }
static int test_reshape3_reduce_axis1(void) { return reshape_then_reduce_backward(3, 1); }
static int test_reshape3_reduce_axis2(void) { return reshape_then_reduce_backward(3, 2); }
static int test_reshape4_reduce_axis0(void) { return reshape_then_reduce_backward(4, 0); }
static int test_reshape4_reduce_axis2(void) { return reshape_then_reduce_backward(4, 2); }

int main(void) {
    cml_init();
    printf("=== per-op autodiff ===\n");

    TEST(asin); TEST(acos); TEST(atan); TEST(asinh); TEST(acosh); TEST(atanh);
    TEST(erf); TEST(sinh); TEST(cosh); TEST(log2); TEST(log10); TEST(exp2);
    TEST(hard_sigmoid); TEST(hard_tanh); TEST(relu6); TEST(quick_gelu);
    TEST(softplus); TEST(softsign); TEST(logsigmoid);

    TEST(triu); TEST(tril); TEST(roll); TEST(flatten);
    TEST(cumsum); TEST(cumprod); TEST(logcumsumexp);
    TEST(prod); TEST(logsumexp); TEST(trace);
    TEST(tile); TEST(repeat_interleave_dim1); TEST(repeat_interleave_dim0);
    TEST(diagonal); TEST(sort); TEST(sort_descending); TEST(topk);

    TEST(floor_zero_grad); TEST(ceil_zero_grad);
    TEST(round_zero_grad); TEST(sign_zero_grad);

    TEST(reshape3_reduce_axis0); TEST(reshape3_reduce_axis1);
    TEST(reshape3_reduce_axis2); TEST(reshape4_reduce_axis0);
    TEST(reshape4_reduce_axis2);

    TEST(reduce_multi_axis);
    TEST(topk_values_and_indices);

    cml_cleanup();
    return TEST_SUMMARY();
}
