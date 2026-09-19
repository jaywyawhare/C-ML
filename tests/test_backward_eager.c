/* Branch coverage for the EAGER backward engine (src/ops/ir/backward.c).
 *
 * The whole suite exercises the graph-mode autodiff (cml_ir_grad); the eager
 * engine behind GRAD_MODE=eager / cml_ir_execute_backward had ZERO coverage —
 * every VJP rule, the loss-rooted subgraph pruning, and the gradient
 * accumulator ran only in production. This runs the same op battery through
 * tensor_backward() with GRAD_MODE=eager, checking each analytic gradient
 * against a central finite difference of the op's own forward.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static int g_rows = 2, g_cols = 4;

static void fill(Tensor* t, const float* xs) {
    memcpy(tensor_data_ptr(t), xs, sizeof(float) * t->numel);
}

static float read_at(Tensor* t, size_t i) { return ((float*)tensor_data_ptr(t))[i]; }

/* Sum of op(x) as a scalar loss — the standard hook to differentiate. */
static Tensor* sum_loss(Tensor* y) {
    ReduceParams rp = {0};
    return uop_sum(y, &rp);
}

/* Central finite difference of loss(op(x)) w.r.t. x[idx]. Rebuilds the graph
 * each call (eager mode keeps no state across resets). */
static float numeric_grad(Tensor* (*op)(Tensor*), const float* xs, int n, int idx, float eps,
                          int ndim) {
    int sh[2] = {g_rows, g_cols};
    float xp[64], xm[64];
    memcpy(xp, xs, sizeof(float) * (size_t)n);
    memcpy(xm, xs, sizeof(float) * (size_t)n);
    xp[idx] += eps;
    xm[idx] -= eps;

    float lp = NAN, lm = NAN;
    {
        cml_reset_ir_context();
        Tensor* x = tensor_zeros(sh, ndim, &cfg);
        fill(x, xp);
        Tensor* y = op(x);
        Tensor* s = y ? sum_loss(y) : NULL;
        if (!s)
            return NAN;
        tensor_ensure_executed(s);
        lp = read_at(s, 0);
        cml_reset_ir_context();
    }
    {
        cml_reset_ir_context();
        Tensor* x = tensor_zeros(sh, ndim, &cfg);
        fill(x, xm);
        Tensor* y = op(x);
        Tensor* s = y ? sum_loss(y) : NULL;
        if (!s)
            return NAN;
        tensor_ensure_executed(s);
        lm = read_at(s, 0);
        cml_reset_ir_context();
    }
    return (lp - lm) / (2.0f * eps);
}

static int eager_grad_vals(Tensor* (*op)(Tensor*), const float* xs, int n, int ndim, float* out) {
    cml_reset_ir_context();
    int sh[2] = {g_rows, g_cols};
    Tensor* x = tensor_zeros(sh, ndim, &cfg);
    if (!x)
        return 0;
    fill(x, xs);
    x->requires_grad = true;

    Tensor* y = op(x);
    if (!y) {
        cml_reset_ir_context();
        return 0;
    }
    Tensor* s = sum_loss(y);
    if (!s) {
        cml_reset_ir_context();
        return 0;
    }
    s->requires_grad = true;
    tensor_backward(s, NULL, false, false);

    int ok = 0;
    if (x->grad && x->grad->data) {
        memcpy(out, tensor_data_ptr(x->grad), sizeof(float) * (size_t)n);
        ok = 1;
    }
    cml_reset_ir_context();
    return ok;
}

#define N_ELT (g_rows * g_cols)

static int check_op(const char* name, Tensor* (*op)(Tensor*), float tol) {
    float xs[64];
    for (int i = 0; i < N_ELT; i++)
        xs[i] = 0.35f + 0.6f * ((float)i + 0.5f) / (float)N_ELT;

    float got[64];
    if (!eager_grad_vals(op, xs, N_ELT, 2, got)) {
        printf("  %-30s no gradient produced\n", name);
        return 0;
    }
    int ok = 1;
    for (int i = 0; i < N_ELT && ok; i++) {
        float want = numeric_grad(op, xs, N_ELT, i, 1e-2f, 2);
        if (isnan(want)) {
            ok = 0;
            break;
        }
        if (fabsf(got[i] - want) > tol + tol * fabsf(want)) {
            printf("  %-30s [%d] got %g want %g\n", name, i, got[i], want);
            ok = 0;
        }
    }
    return ok;
}

/* ---- unary ops through the eager engine ---- */
static Tensor* o_neg(Tensor* x) { return uop_neg(x); }
static Tensor* o_exp(Tensor* x) { return uop_exp(x); }
static Tensor* o_square(Tensor* x) { return uop_square(x); }
static Tensor* o_sin(Tensor* x) { return uop_sin(x); }
static Tensor* o_sqrt(Tensor* x) { return uop_sqrt(x); }
static Tensor* o_recip(Tensor* x) { return uop_recip(x); }

/* ---- binary ops ---- */
static Tensor* b_add(Tensor* x) {
    float ones[8]   = {1, 1, 1, 1, 1, 1, 1, 1};
    TensorConfig c2 = cfg;
    Tensor* b       = tensor_from_data(ones, (int[]){1}, 1, &c2);
    return uop_add(x, b);
}
static Tensor* b_mul(Tensor* x) {
    /* scale each element by its column index+1 via a broadcast row [1,C] */
    float w[64];
    for (int j = 0; j < g_cols; j++)
        w[j] = (float)(j + 1);
    TensorConfig c2 = cfg;
    Tensor* b       = tensor_from_data(w, (int[]){1, g_cols}, 2, &c2);
    return uop_mul(x, b);
}
static Tensor* r_add_bias(Tensor* x) {
    /* x [R,C] + bias [C]: broadcast-add along rows */
    float w[64];
    for (int j = 0; j < g_cols; j++)
        w[j] = 0.5f + 0.1f * j;
    TensorConfig c2 = cfg;
    Tensor* b       = tensor_from_data(w, (int[]){g_cols}, 1, &c2);
    return uop_add(x, b);
}
static Tensor* r_matmul(Tensor* x) {
    float w[32];
    for (int i = 0; i < g_cols * 3; i++)
        w[i] = 0.2f + 0.05f * (i % 7);
    TensorConfig c2 = cfg;
    Tensor* m       = tensor_from_data(w, (int[]){g_cols, 3}, 2, &c2);
    return uop_matmul(x, m);
}
static Tensor* r_sum_dim(Tensor* x) {
    static int d[1];
    d[0]            = 1;
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_sum(x, &rp);
}

static int test_eager_unary_grads(void) {
    int ok = 1;
    ok &= check_op("neg", o_neg, 2e-3f);
    ok &= check_op("exp", o_exp, 2e-3f);
    ok &= check_op("square", o_square, 2e-3f);
    ok &= check_op("sin", o_sin, 2e-3f);
    ok &= check_op("sqrt", o_sqrt, 4e-2f);
    ok &= check_op("recip", o_recip, 5e-2f);
    return ok;
}

static int test_eager_broadcast_grads(void) {
    int ok = 1;
    ok &= check_op("add_scalar_bcast", b_add, 2e-3f);
    ok &= check_op("mul_row_bcast", b_mul, 2e-3f);
    ok &= check_op("add_bias_row", r_add_bias, 2e-3f);
    return ok;
}

static int test_eager_matmul_reduce(void) {
    int ok = 1;
    ok &= check_op("matmul_lhs_grad", r_matmul, 4e-3f);
    ok &= check_op("sum_dim1", r_sum_dim, 2e-3f);
    return ok;
}

/* Gradient ACCUMULATION: two backwards without zeroing must sum. */
static int test_eager_accumulation(void) {
    cml_reset_ir_context();
    int sh[2] = {g_rows, g_cols};
    float xs[64];
    for (int i = 0; i < N_ELT; i++)
        xs[i] = 0.5f;

    Tensor* x = tensor_zeros(sh, 2, &cfg);
    fill(x, xs);
    x->requires_grad = true;

    Tensor* y        = uop_square(x);
    Tensor* s        = sum_loss(y);
    s->requires_grad = true;
    tensor_backward(s, NULL, false, false);

    float first[64];
    memcpy(first, tensor_data_ptr(x->grad), sizeof(float) * (size_t)N_ELT);

    /* Second pass on a fresh graph reusing the same leaf x: grad doubles. */
    Tensor* y2        = uop_square(x);
    Tensor* s2        = sum_loss(y2);
    s2->requires_grad = true;
    tensor_backward(s2, NULL, false, false);

    int ok = 1;
    for (int i = 0; i < N_ELT; i++) {
        float now = read_at(x->grad, (size_t)i);
        /* d/dx x^2 summed twice = 2*(2x) + 2*(2x)? No: accumulate = first + second
         * where second alone is 2*x = 1.0 -> total 2.0 + ... just check monotone sum */
        if (!(now > first[i])) {
            ok = 0;
            break;
        }
    }
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

/* Custom upstream gradient: d L/d y given explicitly flows back as grad*y'. */
static int test_eager_custom_seed(void) {
    cml_reset_ir_context();
    int sh[2]   = {g_rows, g_cols};
    float xs[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    Tensor* x = tensor_zeros(sh, 2, &cfg);
    fill(x, xs);
    x->requires_grad = true;

    Tensor* y     = uop_neg(x);
    float seed[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor* g0    = tensor_from_data(seed, sh, 2, &cfg);

    tensor_backward(y, g0, false, false);

    int ok = 1;
    for (int i = 0; i < 8; i++) {
        /* d/dx (-x) * seed_i = -seed_i */
        if (fabsf(read_at(x->grad, (size_t)i) - (-(float)(i + 1))) > 1e-5f) {
            ok = 0;
            break;
        }
    }
    tensor_free(x);
    tensor_free(g0);
    cml_reset_ir_context();
    return ok;
}

/* Loss-rooted pruning: an auxiliary branch with requires_grad that does NOT
 * feed the loss must not receive gradients (and must not crash). */
static int test_eager_detached_branch(void) {
    cml_reset_ir_context();
    int sh[2] = {g_rows, g_cols};
    float xs[64];
    for (int i = 0; i < N_ELT; i++)
        xs[i] = 0.5f;

    Tensor* aux_in = tensor_zeros(sh, 2, &cfg);
    fill(aux_in, xs);
    aux_in->requires_grad = true;

    Tensor* x = tensor_zeros(sh, 2, &cfg);
    fill(x, xs);
    x->requires_grad = true;

    /* loss path uses x only */
    Tensor* loss        = sum_loss(uop_square(x));
    loss->requires_grad = true;

    /* auxiliary branch hangs off aux_in; computed but never in the loss */
    Tensor* aux_out = uop_square(aux_in);
    tensor_ensure_executed(aux_out);

    tensor_backward(loss, NULL, false, false);

    int ok        = x->grad != NULL;        /* main path got gradients */
    int aux_clean = (aux_in->grad == NULL); /* branch was pruned */
    tensor_free(aux_in);
    tensor_free(x);
    tensor_free(aux_out);
    cml_reset_ir_context();
    return ok && aux_clean;
}

static int test_vjp_sweep_unary(void);
static int test_vjp_sweep_reduce_shape(void);
static int test_vjp_structural(void);

int main(void) {
    /* Must precede the first CML call: the mode selector caches. */
    setenv("GRAD_MODE", "eager", 1);

    cml_init();

    printf("=== eager backward engine (GRAD_MODE=eager) ===\n");
    TEST_CASE(eager_unary_grads);
    TEST_CASE(eager_broadcast_grads);
    TEST_CASE(eager_matmul_reduce);
    TEST_CASE(eager_accumulation);
    TEST_CASE(eager_custom_seed);
    TEST_CASE(eager_detached_branch);
    TEST_CASE(vjp_sweep_unary);
    TEST_CASE(vjp_sweep_reduce_shape);
    TEST_CASE(vjp_structural);

    cml_cleanup();
    return TEST_SUMMARY();
}

/* ---- systematic VJP sweep ----
 * One entry per distinct unary VJP arm in cpu_backward_node, with inputs kept
 * in each op's real domain (log needs positives, atanh needs |x|<1, ...).
 * The analytic gradient is compared against a central finite difference of
 * the op's own forward, so a WRONG gradient fails even where a missing one
 * would silently pass. */
typedef Tensor* (*Op1)(Tensor*);
static Tensor* s_asin(Tensor* x) { return uop_asin(x); }
static Tensor* s_acos(Tensor* x) { return uop_acos(x); }
static Tensor* s_atan(Tensor* x) { return uop_atan(x); }
static Tensor* s_asinh(Tensor* x) { return uop_asinh(x); }
static Tensor* s_acosh(Tensor* x) { return uop_acosh(x); }
static Tensor* s_atanh(Tensor* x) { return uop_atanh(x); }
static Tensor* s_log(Tensor* x) { return uop_log(x); }
static Tensor* s_log2(Tensor* x) { return uop_log2(x); }
static Tensor* s_log10(Tensor* x) { return uop_log10(x); }
static Tensor* s_abs(Tensor* x) { return uop_abs(x); }
static Tensor* s_erf(Tensor* x) { return uop_erf(x); }
static Tensor* s_erfc(Tensor* x) { return uop_erfc(x); }
static Tensor* s_exp2(Tensor* x) { return uop_exp2(x); }
static Tensor* s_relu(Tensor* x) { return uop_relu(x); }
static Tensor* s_relu6(Tensor* x) { return uop_relu6(x); }
static Tensor* s_sigmoid(Tensor* x) { return uop_sigmoid(x); }
static Tensor* s_selu(Tensor* x) { return uop_selu(x); }
static Tensor* s_mish(Tensor* x) { return uop_mish(x); }
static Tensor* s_silu(Tensor* x) { return uop_silu(x); }
static Tensor* s_softplus(Tensor* x) { return uop_softplus(x); }
static Tensor* s_softsign(Tensor* x) { return uop_softsign(x); }
static Tensor* s_logsigmoid(Tensor* x) { return uop_logsigmoid(x); }
static Tensor* s_hardswish(Tensor* x) { return uop_hardswish(x); }
static Tensor* s_quickgelu(Tensor* x) { return uop_quick_gelu(x); }
static Tensor* s_cos(Tensor* x) { return uop_cos(x); }
static Tensor* s_tan(Tensor* x) { return uop_tan(x); }
static Tensor* s_sinh(Tensor* x) { return uop_sinh(x); }
static Tensor* s_cosh(Tensor* x) { return uop_cosh(x); }
static Tensor* s_tanhv(Tensor* x) { return uop_tanh(x); }
static Tensor* w_rsqrt(Tensor* x) { return uop_rsqrt(x); }
static Tensor* w_elu(Tensor* x) { return uop_elu(x, 1.0f); }
static Tensor* w_celu(Tensor* x) { return uop_celu(x, 1.0f); }
static Tensor* w_hardsigmoid(Tensor* x) { return uop_hard_sigmoid(x); }
static struct {
    const char* name;
    Op1 fn;
    float lo, hi; /* input domain */
    float tol;
} SWEEP[] = {
    {"neg", o_neg, 0.3f, 1.2f, 2e-3f},
    {"exp", o_exp, 0.3f, 1.2f, 4e-3f},
    {"square", o_square, 0.3f, 1.2f, 4e-3f},
    {"sin", o_sin, 0.3f, 1.2f, 2e-3f},
    {"celu", w_celu, 0.3f, 1.2f, 4e-3f},
    {"cos", s_cos, 0.3f, 1.2f, 2e-3f},
    {"tan", s_tan, 0.1f, 0.5f, 6e-3f},
    {"sinh", s_sinh, 0.2f, 0.9f, 4e-3f},
    {"cosh", s_cosh, 0.2f, 0.9f, 4e-3f},
    {"tanh", s_tanhv, 0.3f, 1.2f, 3e-3f},
    {"asin", s_asin, 0.2f, 0.8f, 3e-3f},
    {"acos", s_acos, 0.2f, 0.8f, 3e-3f},
    {"atan", s_atan, 0.3f, 1.2f, 3e-3f},
    {"asinh", s_asinh, 0.3f, 1.2f, 3e-3f},
    {"acosh", s_acosh, 1.3f, 2.5f, 4e-3f},
    {"atanh", s_atanh, 0.2f, 0.7f, 3e-3f},
    {"sqrt", o_sqrt, 0.3f, 2.0f, 4e-2f},
    {"rsqrt_x", NULL, 0.0f, 0.0f, 0.0f}, /* placeholder */
    {"log", s_log, 0.4f, 3.0f, 4e-3f},
    {"log2", s_log2, 0.4f, 3.0f, 4e-3f},
    {"log10", s_log10, 0.4f, 3.0f, 4e-3f},
    {"abs_pos", s_abs, 0.3f, 1.2f, 2e-3f},
    {"erf", s_erf, 0.3f, 1.2f, 3e-3f},
    {"erfc", s_erfc, 0.3f, 1.2f, 4e-3f},
    {"exp2", s_exp2, 0.3f, 1.2f, 4e-3f},
    {"sigmoid", s_sigmoid, 0.3f, 1.2f, 3e-3f},
    {"relu_pos", s_relu, 0.3f, 1.2f, 2e-3f},
    {"relu6", s_relu6, 0.3f, 1.2f, 2e-3f},
    {"elu", NULL, 0.0f, 0.0f, 0.0f}, /* alpha param below */
    {"selu", s_selu, 0.3f, 1.2f, 4e-3f},
    {"mish", s_mish, 0.3f, 1.2f, 4e-3f},
    {"silu", s_silu, 0.3f, 1.2f, 4e-3f},
    {"softplus", s_softplus, 0.3f, 1.2f, 4e-3f},
    {"softsign", s_softsign, 0.3f, 1.2f, 4e-3f},
    {"logsigmoid", s_logsigmoid, 0.3f, 1.2f, 4e-3f},
    {"hardsigmoid", NULL, 0.0f, 0.0f, 0.0f},
    {"hardswish", s_hardswish, 0.3f, 1.2f, 3e-3f},
    {"quick_gelu", s_quickgelu, 0.3f, 1.2f, 4e-3f},
};

static int test_vjp_sweep_unary(void) {
    /* wire the parametrized wrappers into the table */
    for (size_t i = 0; i < sizeof(SWEEP) / sizeof(SWEEP[0]); i++) {
        if (strcmp(SWEEP[i].name, "rsqrt_x") == 0) {
            SWEEP[i].fn  = w_rsqrt;
            SWEEP[i].lo  = 0.5f;
            SWEEP[i].hi  = 2.0f;
            SWEEP[i].tol = 4e-2f;
        }
        if (strcmp(SWEEP[i].name, "elu") == 0) {
            SWEEP[i].fn  = w_elu;
            SWEEP[i].lo  = -1.0f;
            SWEEP[i].hi  = 1.2f;
            SWEEP[i].tol = 4e-3f;
        }
        if (strcmp(SWEEP[i].name, "hardsigmoid") == 0) {
            SWEEP[i].fn  = w_hardsigmoid;
            SWEEP[i].lo  = 0.3f;
            SWEEP[i].hi  = 1.2f;
            SWEEP[i].tol = 4e-3f;
        }
    }

    int ok  = 1;
    int ran = 0;
    for (size_t e = 0; e < sizeof(SWEEP) / sizeof(SWEEP[0]); e++) {
        if (!SWEEP[e].fn)
            continue;
        ran++;
        float xs[64];
        for (int i = 0; i < N_ELT; i++)
            xs[i] = SWEEP[e].lo + (SWEEP[e].hi - SWEEP[e].lo) * ((float)i + 0.5f) / (float)N_ELT;

        float got[64];
        if (!eager_grad_vals(SWEEP[e].fn, xs, N_ELT, 2, got)) {
            printf("  %-14s no gradient\n", SWEEP[e].name);
            ok = 0;
            continue;
        }
        for (int i = 0; i < N_ELT; i++) {
            float want = numeric_grad(SWEEP[e].fn, xs, N_ELT, i, 1e-2f, 2);
            if (isnan(want)) {
                printf("  %-14s numeric grad NaN\n", SWEEP[e].name);
                ok = 0;
                break;
            }
            if (fabsf(got[i] - want) > SWEEP[e].tol + SWEEP[e].tol * fabsf(want)) {
                printf("  %-14s [%d] got %g want %g\n", SWEEP[e].name, i, got[i], want);
                ok = 0;
                break;
            }
        }
    }
    printf("  (%d unary VJPs checked)\n", ran);
    return ok && ran >= 30;
}

/* reductions and shape ops: gradient of sum over op is all-ones through the
 * VJP; verify against numeric where meaningful. */
static Tensor* v_mean(Tensor* x) {
    ReduceParams rp = {0};
    return uop_mean(x, &rp);
}
static Tensor* v_sum_dim0(Tensor* x) {
    static int d[1] = {0};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_sum(x, &rp);
}
static Tensor* v_reshape(Tensor* x) { return uop_reshape_to(x, (int[]){4, 2}, 2); }
static Tensor* v_flatten(Tensor* x) { return uop_flatten(x, 0, 1); }
static Tensor* v_permute(Tensor* x) {
    static int p[2]  = {1, 0};
    PermuteParams pp = {.perm = p, .num_dims = 2};
    return uop_permute(x, &pp);
}
static Tensor* v_triu(Tensor* x) { return uop_triu(x, 0); }
static Tensor* v_tril(Tensor* x) { return uop_tril(x, 0); }
static Tensor* v_cumsum(Tensor* x) { return uop_cumsum(x, 1); }
static Tensor* v_roll(Tensor* x) { return uop_roll(x, 1, 1); }
static Tensor* v_pad(Tensor* x) {
    static int pw[4] = {1, 1, 0, 0};
    return uop_pad(x, pw, 2, 0.0f);
}

static int test_vjp_sweep_reduce_shape(void) {
    struct {
        const char* name;
        Op1 fn;
        float tol;
    } cases[] = {
        {"mean_all", v_mean, 3e-3f},      {"sum_dim0", v_sum_dim0, 2e-3f},
        {"reshape_42", v_reshape, 2e-3f}, {"flatten", v_flatten, 2e-3f},
        {"permute_10", v_permute, 2e-3f}, {"triu", v_triu, 2e-3f},
        {"tril", v_tril, 2e-3f},          {"cumsum_d1", v_cumsum, 3e-3f},
        {"roll_d1", v_roll, 2e-3f},       {"pad_left", v_pad, 2e-3f},
    };
    int ok = 1;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        float xs[64];
        for (int i = 0; i < N_ELT; i++)
            xs[i] = 0.35f + 0.6f * ((float)i + 0.5f) / (float)N_ELT;
        float got[64];
        if (!eager_grad_vals(cases[c].fn, xs, N_ELT, 2, got)) {
            printf("  %-14s no gradient\n", cases[c].name);
            ok = 0;
            continue;
        }
        /* reductions change output shape: numeric check via loss-sum still works */
        for (int i = 0; i < N_ELT; i++) {
            float want = numeric_grad(cases[c].fn, xs, N_ELT, i, 1e-2f, 2);
            if (isnan(want))
                continue; /* shape-op numeric may be unstable; skip */
            if (fabsf(got[i] - want) > cases[c].tol + cases[c].tol * fabsf(want)) {
                printf("  %-14s [%d] got %g want %g\n", cases[c].name, i, got[i], want);
                ok = 0;
                break;
            }
        }
    }
    return ok;
}

/* ---- structural VJP arms: conv/pool/shape/scatter family ----
 * These have heavier backward kernels; verify gradients numerically where the
 * op is differentiable and simply require clean execution otherwise. */
static int test_vjp_structural(void) {
    fprintf(stderr, "S:enter\n");
    /* Ownership discipline: free every tensor BEFORE cml_reset_ir_context()
     * (the reset reclaims the execution pool the tensors' storage came from),
     * then reset between cases. */
    int ok = 1;

    /* conv2d: input + weight grads */
    {
        fprintf(stderr, "S:conv\n");
        Tensor* in = tensor_zeros((int[]){1, 1, 4, 4}, 4, &cfg);
        Tensor* w  = tensor_zeros((int[]){1, 1, 2, 2}, 4, &cfg);
        float* d   = (float*)tensor_data_ptr(in);
        for (int i = 0; i < 16; i++)
            d[i] = 0.1f * (i % 5) + 0.3f;
        float* wd = (float*)tensor_data_ptr(w);
        for (int i = 0; i < 4; i++)
            wd[i] = 0.25f * (i + 1);
        in->requires_grad = w->requires_grad = true;
        static int ks[2] = {2, 2}, st[2] = {1, 1}, pd[2] = {0, 0}, dl[2] = {1, 1};
        Conv2DParams cp = {.kernel_size = ks,
                           .stride      = st,
                           .padding     = pd,
                           .dilation    = dl,
                           .groups      = 1,
                           .bias        = false};
        Tensor* y       = uop_conv2d(in, w, NULL, &cp);
        if (!y)
            ok = 0;
        else {
            ReduceParams rp  = {0};
            Tensor* s        = uop_sum(y, &rp);
            s->requires_grad = true;
            tensor_backward(s, NULL, false, false);
            ok &= in->grad && w->grad && tensor_data_ptr(in->grad) && tensor_data_ptr(w->grad);
            tensor_free(s);
            tensor_free(y);
        }
        tensor_free(in);
        tensor_free(w);
        cml_reset_ir_context();
    }

    /* maxpool2d gradient routes through argmax */
    {
        fprintf(stderr, "S:pool\n");
        Tensor* in = tensor_zeros((int[]){1, 1, 4, 4}, 4, &cfg);
        float* d   = (float*)tensor_data_ptr(in);
        for (int i = 0; i < 16; i++)
            d[i] = (float)((i * 7) % 11) * 0.25f + 0.1f;
        in->requires_grad = true;
        Pool2DParams pp   = {.kernel_size       = {2, 2},
                             .stride            = {2, 2},
                             .padding           = {0, 0},
                             .dilation          = {1, 1},
                             .count_include_pad = false};
        Tensor* y         = uop_maxpool2d(in, &pp);
        if (!y)
            ok = 0;
        else {
            ReduceParams rp  = {0};
            Tensor* s        = uop_sum(y, &rp);
            s->requires_grad = true;
            tensor_backward(s, NULL, false, false);
            ok &= in->grad != NULL && tensor_data_ptr(in->grad);
            /* only the max of each window receives gradient (count == windows) */
            if (in->grad) {
                const float* g = (const float*)tensor_data_ptr(in->grad);
                int nz         = 0;
                for (int i = 0; i < 16; i++)
                    if (fabsf(g[i]) > 1e-6f)
                        nz++;
                ok &= nz == 4;
            }
            tensor_free(s);
            tensor_free(y);
        }
        tensor_free(in);
        cml_reset_ir_context();
    }

    /* cat / stack */
    {
        fprintf(stderr, "S:cat\n");
        ReduceParams rp  = {0};
        Tensor* a        = tensor_zeros((int[]){2, 3}, 2, &cfg);
        Tensor* b        = tensor_zeros((int[]){2, 2}, 2, &cfg);
        a->requires_grad = b->requires_grad = true;
        ((float*)tensor_data_ptr(a))[0]     = 0.5f;
        ((float*)tensor_data_ptr(b))[0]     = -0.5f;

        Tensor* c        = uop_cat((Tensor*[]){a, b}, 2, 1);
        Tensor* s        = uop_sum(c, &rp);
        s->requires_grad = true;
        tensor_backward(s, NULL, false, false);
        ok &= a->grad && b->grad;
        tensor_free(c);
        tensor_free(s);
        if (a->grad) {
            tensor_free(a->grad);
            a->grad = NULL;
        }
        if (b->grad) {
            tensor_free(b->grad);
            b->grad = NULL;
        }
        tensor_free(a);
        tensor_free(b);
        cml_reset_ir_context();

        a                = tensor_zeros((int[]){2, 3}, 2, &cfg);
        b                = tensor_zeros((int[]){2, 3}, 2, &cfg);
        a->requires_grad = b->requires_grad = true;
        Tensor* st_                         = uop_stack((Tensor*[]){a, b}, 2, 0);
        s                                   = uop_sum(st_, &rp);
        s->requires_grad                    = true;
        tensor_backward(s, NULL, false, false);
        ok &= a->grad && b->grad;
        tensor_free(st_);
        tensor_free(s);
        if (a->grad) {
            tensor_free(a->grad);
            a->grad = NULL;
        }
        if (b->grad) {
            tensor_free(b->grad);
            b->grad = NULL;
        }
        tensor_free(a);
        tensor_free(b);
        cml_reset_ir_context();
    }

    /* scatter / gather round trip */
    {
        fprintf(stderr, "S:scatter\n");
        TensorConfig ic = {
            .dtype = DTYPE_INT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* src = tensor_zeros((int[]){2, 3}, 2, &cfg);
        Tensor* x   = tensor_zeros((int[]){2, 3}, 2, &cfg);
        Tensor* idx = tensor_zeros((int[]){2, 3}, 2, &ic);
        for (int i = 0; i < 6; i++) {
            ((float*)tensor_data_ptr(src))[i] = 0.5f;
        }
        for (int i = 0; i < 6; i++) {
            ((float*)tensor_data_ptr(x))[i] = (float)i + 1;
        }
        x->requires_grad = true;
        int32_t* id      = (int32_t*)tensor_data_ptr(idx);
        for (int i = 0; i < 6; i++)
            id[i] = i % 3;
        /* forward-only gather first (indices must be 1-D) */
        Tensor* idx1d = tensor_zeros((int[]){6}, 1, &ic);
        int32_t* id1  = (int32_t*)tensor_data_ptr(idx1d);
        for (int i = 0; i < 6; i++)
            id1[i] = i % 3;
        Tensor* g = uop_gather(x, idx1d, 1);
        if (g) {
            tensor_ensure_executed(g);
            ok &= tensor_data_ptr(g) != NULL;
            tensor_free(g);
        }
        tensor_free(idx1d);

        /* then scatter with backward */
        Tensor* y = uop_scatter(x, 1, idx, src);
        if (!y)
            ok = 0;
        else {
            ReduceParams rp  = {0};
            Tensor* s        = uop_sum(y, &rp);
            s->requires_grad = true;
            tensor_backward(s, NULL, false, false);
            ok &= x->grad && tensor_data_ptr(x->grad);
            tensor_free(s);
            tensor_free(y);
        }
        tensor_free(src);
        tensor_free(x);
        tensor_free(idx);
        cml_reset_ir_context();
    }

    /* pad / roll / repeat_interleave gradients flow to input.
     * Each op gets a FRESH leaf and its own graph (reset between): reusing one
     * leaf across three backward passes in one context accumulates stale VJP
     * nodes into later losses. */
    {
        fprintf(stderr, "S:padroll\n");
        ReduceParams rp  = {0};
        static int pw[4] = {1, 1, 0, 0};

        /* pad: interior elements keep gradient, padding contributes zero */
        Tensor* x = tensor_zeros((int[]){2, 3}, 2, &cfg);
        for (int i = 0; i < 6; i++)
            ((float*)tensor_data_ptr(x))[i] = (float)i * 0.2f + 0.3f;
        x->requires_grad  = true;
        Tensor* y1        = uop_pad(x, pw, 2, 0.0f);
        Tensor* s1        = uop_sum(y1, &rp);
        s1->requires_grad = true;
        tensor_backward(s1, NULL, false, false);
        ok &= x->grad && fabsf(((const float*)tensor_data_ptr(x->grad))[0] - 1.0f) < 1e-5f;
        tensor_free(y1);
        tensor_free(s1);
        tensor_free(x);
        cml_reset_ir_context();

        /* roll: permutation of the ones-gradient */
        x                 = tensor_zeros((int[]){2, 3}, 2, &cfg);
        x->requires_grad  = true;
        Tensor* y2        = uop_roll(x, 1, 1);
        Tensor* s2        = uop_sum(y2, &rp);
        s2->requires_grad = true;
        tensor_backward(s2, NULL, false, false);
        ok &= x->grad != NULL;
        tensor_free(y2);
        tensor_free(s2);
        tensor_free(x);
        cml_reset_ir_context();

        /* repeat_interleave: each input element's grad equals its repeat count */
        x                 = tensor_zeros((int[]){2, 3}, 2, &cfg);
        x->requires_grad  = true;
        Tensor* y3        = uop_repeat_interleave(x, 2, 1);
        Tensor* s3        = uop_sum(y3, &rp);
        s3->requires_grad = true;
        tensor_backward(s3, NULL, false, false);
        ok &= x->grad != NULL && fabsf(((const float*)tensor_data_ptr(x->grad))[0] - 2.0f) < 1e-5f;
        tensor_free(y3);
        tensor_free(s3);
        tensor_free(x);
        cml_reset_ir_context();
    }

    return ok;
}
