/* Exhaustive op matrix for the executor (branch-coverage targeted).
 *
 * The value-level suites prove correctness on the paths they take; this file
 * exists to TAKE all the paths. Every UOP family is driven through the
 * executor across:
 *
 *   - shape classes: 1-D, 2-D, 3-D, empty, size-1 dims
 *   - dtypes:        f32 / f64 / i32 / i8 (where legal)
 *   - broadcast classes for binary ops: same / row / col / scalar
 *   - executor arms: default fusion+JIT, DISABLE_FUSION=1 interpreter,
 *                    TRANSCENDENTAL=2 polynomial, WINO=0/1 conv,
 *                    CHECK_OOB=1, NOOPT + SPLIT_REDUCEOP=0
 *
 * Assertions are deliberately structural (executes and produces data, or a
 * clean NULL) — numeric truth is owned by the conformance/grad suites.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "ops/uops.h"
#include "test_harness.h"

static TensorConfig f32c = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
static TensorConfig i32c(void) {
    TensorConfig c = {
        .dtype = DTYPE_INT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return c;
}

static int g_fail = 0;

/* Execute and release; record structural failure only. */
static void run(Tensor* t) {
    if (!t)
        return; /* clean rejection is acceptable */
    tensor_ensure_executed(t);
    if (t->numel > 0 && !tensor_data_ptr(t))
        g_fail++;
    tensor_free(t);
}

static Tensor* mk(int r, int c) {
    int sh[2] = {r, c};
    Tensor* t = tensor_zeros(sh, 2, &f32c);
    if (!t)
        return NULL;
    float* d = (float*)tensor_data_ptr(t);
    for (int i = 0; i < r * c; i++)
        d[i] = 0.25f + 0.5f * (float)(i % 7);
    return t;
}
static Tensor* mk1(int n) {
    Tensor* t = tensor_zeros((int[]){n}, 1, &f32c);
    if (!t)
        return NULL;
    float* d = (float*)tensor_data_ptr(t);
    for (int i = 0; i < n; i++)
        d[i] = 0.4f + 0.3f * (float)(i % 5);
    return t;
}
static Tensor* mk3(void) {
    Tensor* t = tensor_zeros((int[]){2, 2, 2}, 3, &f32c);
    if (!t)
        return NULL;
    float* d = (float*)tensor_data_ptr(t);
    for (int i = 0; i < 8; i++)
        d[i] = 0.3f + 0.4f * (float)(i % 5);
    return t;
}
static Tensor* mk_empty(void) { return tensor_zeros((int[]){0}, 1, &f32c); }

/* ---- unary table ---- */
typedef Tensor* (*U1)(Tensor*);
static const struct {
    const char* n;
    U1 f;
} UN[] = {
    {"neg", uop_neg},
    {"exp", uop_exp},
    {"log", uop_log},
    {"sqrt", uop_sqrt},
    {"recip", uop_recip},
    {"abs", uop_abs},
    {"sin", uop_sin},
    {"cos", uop_cos},
    {"tan", uop_tan},
    {"tanh", uop_tanh},
    {"sigmoid", uop_sigmoid},
    {"sign", uop_sign},
    {"floor", uop_floor},
    {"ceil", uop_ceil},
    {"round", uop_round},
    {"log2", uop_log2},
    {"exp2", uop_exp2},
    {"asin", uop_asin},
    {"acos", uop_acos},
    {"atan", uop_atan},
    {"square", uop_square},
    {"rsqrt", uop_rsqrt},
    {"erf", uop_erf},
    {"sinh", uop_sinh},
    {"cosh", uop_cosh},
    {"asinh", uop_asinh},
    {"acosh", uop_acosh},
    {"atanh", uop_atanh},
    {"trunc", uop_trunc},
    {"isinf", uop_isinf},
    {"isnan", uop_isnan},
    {"isfinite", uop_isfinite},
    {"logical_not", uop_logical_not},
    {"relu", uop_relu},
    {"relu6", uop_relu6},
    {"gelu", uop_gelu},
    {"mish", uop_mish},
    {"silu", uop_silu},
    {"hardswish", uop_hardswish},
    {"selu", uop_selu},
    {"softplus", uop_softplus},
    {"softsign", uop_softsign},
    {"logsigmoid", uop_logsigmoid},
    {"quick_gelu", uop_quick_gelu},
    {"hardsigmoid", uop_hard_sigmoid},
    {"hardtanh", uop_hard_tanh},
};
#define NUN ((int)(sizeof(UN) / sizeof(UN[0])))

/* ---- reductions over axis/keepdim matrix ---- */
typedef Tensor* (*RedFn)(Tensor*, ReduceParams*);
static int phase_reductions_inner(void) {
    static int d0[1] = {0}, d1[1] = {1}, dneg[1] = {-1};
    ReduceParams all = {0};
    ReduceParams p0  = {.dims = d0, .num_dims = 1, .keepdim = false};
    ReduceParams p0k = {.dims = d0, .num_dims = 1, .keepdim = true};
    ReduceParams p1  = {.dims = d1, .num_dims = 1, .keepdim = false};
    ReduceParams pn  = {.dims = dneg, .num_dims = 1, .keepdim = false};

    const struct {
        const char* n;
        RedFn f;
    } R[] = {
        {"sum", uop_sum},           {"mean", uop_mean},
        {"prod", uop_prod},         {"var", uop_var},
        {"std", uop_std},           {"maxred", uop_max_reduce},
        {"minred", uop_min_reduce}, {"logsumexp", uop_logsumexp},
    };
    for (size_t i = 0; i < sizeof(R) / sizeof(R[0]); i++) {
        Tensor* a;
        a = mk(2, 3);
        run(R[i].f(a, &all));
        tensor_free(a);
        a = mk(2, 3);
        run(R[i].f(a, &p0));
        tensor_free(a);
        a = mk(2, 3);
        run(R[i].f(a, &p0k));
        tensor_free(a);
        a = mk(2, 3);
        run(R[i].f(a, &p1));
        tensor_free(a);
        a = mk(2, 3);
        run(R[i].f(a, &pn));
        tensor_free(a);
        a = mk_empty();
        run(R[i].f(a, &all));
        tensor_free(a); /* identity arms */
    }

    /* argmax/argmin + cumulative scans along both axes */
    Tensor* a = mk(2, 3);
    run(uop_argmax(a, &p1));
    run(uop_argmin(a, &p1));
    run(uop_cumsum(a, 0));
    run(uop_cumsum(a, 1));
    run(uop_cumprod(a, 1));
    run(uop_cummax(a, 0));
    run(uop_cummin(a, 1));
    run(uop_logcumsumexp(a, 1));
    run(uop_softmax(a, 1));
    run(uop_softmax(a, 0));
    tensor_free(a);
    return !g_fail;
}

/* ---- shape/layout ops ---- */
static int phase_shape_ops_inner(void) {
    Tensor* a = mk(2, 3);
    run(uop_reshape_to(a, (int[]){3, 2}, 2));
    run(uop_flatten(a, 0, 1));
    {
        int sizes[2] = {2, 3};
        run(uop_unflatten(a, 0, sizes, 2));
    }
    {
        static int p[2]  = {1, 0};
        PermuteParams pp = {.perm = p, .num_dims = 2};
        run(uop_permute(a, &pp));
    }
    {
        static int st[2] = {0, 0}, en[2] = {2, 2}, sp[2] = {1, 1};
        SliceParams sp2 = {.start = st, .end = en, .step = sp, .num_dims = 2};
        run(uop_slice(a, &sp2));
    }
    {
        static int pw[4] = {1, 1, 1, 1};
        run(uop_pad(a, pw, 2, 0.0f));
        run(uop_pad_reflect(a, (int[]){0, 0, 1, 1}, 2));
        run(uop_pad_replicate(a, (int[]){1, 1, 0, 0}, 2));
    }
    run(uop_roll(a, 2, 0));
    run(uop_repeat_interleave(a, 2, 1));
    {
        static int reps[2] = {2, 2};
        run(uop_tile(a, reps, 2));
    }
    run(uop_triu(a, 0));
    run(uop_tril(a, -1));
    run(uop_expand_to(a, (int[]){2, 6}, 2));

    Tensor* v4 = mk1(4);
    run(uop_one_hot(v4, 5));
    tensor_free(v4);

    { /* where */
        WhereParams wp = {.cond = mk(2, 3), .a = mk(2, 3), .b = mk(2, 3)};
        float* cd      = (float*)tensor_data_ptr(wp.cond);
        for (int i = 0; i < 6; i++)
            cd[i] = (float)(i % 2);
        run(uop_where(&wp));
        tensor_free(wp.cond);
        tensor_free(wp.a);
        tensor_free(wp.b);
    }

    {
        Tensor* m = mk(2, 3);
        float* md = (float*)tensor_data_ptr(m);
        for (int i = 0; i < 6; i++)
            md[i] = (float)(i % 2);
        run(uop_masked_select(a, m));
        tensor_free(m);
    }

    {
        Tensor* b = mk(2, 3);
        run(uop_cat((Tensor*[]){a, b}, 2, 0));
        run(uop_cat((Tensor*[]){a, b}, 2, 1));
        run(uop_stack((Tensor*[]){a, b}, 2, 0));
        tensor_free(b);
    }

    Tensor* sq = mk(3, 3);
    run(uop_trace(sq));
    run(uop_diag(sq, 0));
    run(uop_diagonal(sq, 0, 0, 1));
    tensor_free(sq);

    { /* gather/scatter with int32 indices */
        TensorConfig ic = i32c();
        Tensor* idx     = tensor_zeros((int[]){2, 3}, 2, &ic);
        int32_t* id     = (int32_t*)tensor_data_ptr(idx);
        for (int i = 0; i < 6; i++)
            id[i] = i % 3;
        Tensor* src = mk(2, 3);
        run(uop_gather(a, idx, 1));
        run(uop_scatter(a, 1, idx, src));
        tensor_free(idx);
        tensor_free(src);
    }

    { /* conv/pool/unfold on NCHW */
        Tensor* w        = mk(1, 1);
        static int ks[2] = {2, 2}, st[2] = {1, 1}, pd[2] = {0, 0}, dl[2] = {1, 1};
        Conv2DParams cp = {.kernel_size = ks,
                           .stride      = st,
                           .padding     = pd,
                           .dilation    = dl,
                           .groups      = 1,
                           .bias        = false};
        Tensor* in      = tensor_zeros((int[]){1, 1, 4, 4}, 4, &f32c);
        float* dd       = (float*)tensor_data_ptr(in);
        for (int i = 0; i < 16; i++)
            dd[i] = 0.3f + (float)i * 0.05f;
        run(uop_conv2d(in, w, NULL, &cp));
        Pool2DParams pp2 = {.kernel_size       = {2, 2},
                            .stride            = {1, 1},
                            .padding           = {0, 0},
                            .dilation          = {1, 1},
                            .count_include_pad = false};
        run(uop_maxpool2d(in, &pp2));
        run(uop_avgpool2d(in, &pp2));
        run(uop_unfold(in, 2, 1));
        tensor_free(in);
        tensor_free(w);
    }

    run(uop_sort(a, 1, false));
    run(uop_argsort(a, 1, false));
    {
        Tensor* idx = NULL;
        run(uop_topk(a, 2, 1, true, &idx));
        if (idx)
            tensor_free(idx);
    }
    run(uop_nonzero(a));

    tensor_free(a);
    return !g_fail;
}

/* ---- binary ops with broadcast classes ---- */
typedef Tensor* (*B2)(Tensor*, Tensor*);
static const struct {
    const char* n;
    B2 f;
} BN[] = {
    {"add", uop_add},           {"sub", uop_sub},
    {"mul", uop_mul},           {"div", uop_div},
    {"max2", uop_max},          {"minimum", uop_minimum},
    {"copysign", uop_copysign}, {"logaddexp", uop_logaddexp},
};
static int phase_binary_inner(void) {
    for (int i = 0; i < (int)(sizeof(BN) / sizeof(BN[0])); i++) {
        Tensor *a = mk(2, 3), *b = mk(2, 3);
        run(BN[i].f(a, b));
        tensor_free(a);
        tensor_free(b);

        a = mk(2, 3);
        b = mk1(3); /* row broadcast [2,3]x[3] */
        run(BN[i].f(a, b));
        tensor_free(a);
        tensor_free(b);

        a = mk(2, 3);
        b = mk(2, 1); /* col broadcast */
        run(BN[i].f(a, b));
        tensor_free(a);
        tensor_free(b);

        a = mk(1, 1);
        b = mk(2, 3); /* scalar splat */
        run(BN[i].f(a, b));
        tensor_free(a);
        tensor_free(b);

        a = mk(2, 3);
        b = mk(5, 2); /* non-broadcastable: clean fail */
        run(BN[i].f(a, b));
        tensor_free(a);
        tensor_free(b);
    }

    Tensor *a = mk(2, 3), *b = mk(2, 3);
    run(uop_cmplt(a, b));
    run(uop_cmpgt(a, b));
    run(uop_cmpeq(a, b));
    run(uop_cmpne(a, b));
    run(uop_cmple(a, b));
    run(uop_cmpge(a, b));
    run(uop_logical_and(a, b));
    run(uop_logical_or(a, b));
    run(uop_pow(a, b));
    tensor_free(a);
    tensor_free(b);

    a = mk(2, 3);
    b = mk(2, 3);
    {
        int32_t* ad = (int32_t*)tensor_data_ptr(a);
        int32_t* bd = (int32_t*)tensor_data_ptr(b);
        for (int i = 0; i < 6; i++) {
            ad[i] = i + 8;
            bd[i] = (i % 3) + 1;
        }
    }
    run(uop_mod(a, b));
    run(uop_idiv(a, b));
    run(uop_lshift(a, b));
    run(uop_rshift(a, b));
    tensor_free(a);
    tensor_free(b);
    return !g_fail;
}

/* ---- dtype matrix through native/typed paths ---- */
static int phase_dtypes_inner(void) {
    /* f64 */
    TensorConfig cf = {
        .dtype = DTYPE_FLOAT64, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* da  = tensor_zeros((int[]){2, 3}, 2, &cf);
    Tensor* db  = tensor_zeros((int[]){2, 3}, 2, &cf);
    double* dad = (double*)tensor_data_ptr(da);
    double* dbd = (double*)tensor_data_ptr(db);
    for (int i = 0; i < 6; i++) {
        dad[i] = 0.5 * (double)i + 0.5;
        dbd[i] = 1.5 - 0.1 * (double)i;
    }
    run(uop_add(da, db));
    run(uop_mul(da, db));
    run(uop_div(da, db));
    run(uop_neg(da));
    run(uop_exp(da));
    run(uop_sqrt(da));
    ReduceParams rp = {0};
    run(uop_sum(da, &rp));
    run(uop_mean(da, &rp));
    run(uop_matmul(da, db));
    {
        static int d[1]  = {1};
        ReduceParams rp1 = {.dims = d, .num_dims = 1, .keepdim = false};
        run(uop_argmax(da, &rp1));
        run(uop_cmpgt(da, db));
    }
    tensor_free(da);
    tensor_free(db);

    /* i32 */
    TensorConfig ci = i32c();
    Tensor* ia      = tensor_zeros((int[]){2, 3}, 2, &ci);
    Tensor* ib      = tensor_zeros((int[]){2, 3}, 2, &ci);
    int32_t* ad     = (int32_t*)tensor_data_ptr(ia);
    int32_t* bd     = (int32_t*)tensor_data_ptr(ib);
    for (int i = 0; i < 6; i++) {
        ad[i] = i + 1;
        bd[i] = (i % 3) + 1;
    }
    run(uop_add(ia, ib));
    run(uop_mul(ia, ib));
    run(uop_sub(ia, ib));
    run(uop_sum(ia, &rp));
    run(uop_prod(ia, &rp));
    run(uop_max_reduce(ia, &rp));
    run(uop_matmul(ia, ib));
    {
        static int d[1]  = {1};
        ReduceParams rp1 = {.dims = d, .num_dims = 1, .keepdim = false};
        run(uop_cumsum(ia, 1));
        run(uop_argmax(ia, &rp1));
        run(uop_sort(ia, 1, false));
        run(uop_cmpgt(ia, ib));
        run(uop_min_reduce(ia, &rp));
        run(uop_mean(ia, &rp));
        run(uop_neg(ia));
        run(uop_abs(ia));
    }
    tensor_free(ia);
    tensor_free(ib);

    /* i8 */
    TensorConfig c8 = {
        .dtype = DTYPE_INT8, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* a8  = tensor_zeros((int[]){2, 3}, 2, &c8);
    Tensor* b8  = tensor_zeros((int[]){2, 3}, 2, &c8);
    int8_t* ad8 = (int8_t*)tensor_data_ptr(a8);
    int8_t* bd8 = (int8_t*)tensor_data_ptr(b8);
    for (int i = 0; i < 6; i++) {
        ad8[i] = (int8_t)(i + 1);
        bd8[i] = (int8_t)((i % 3) + 1);
    }
    run(uop_add(a8, b8));
    run(uop_mul(a8, b8));
    run(uop_sum(a8, &rp));
    run(uop_matmul(a8, b8));
    {
        static int d[1]  = {1};
        ReduceParams rp1 = {.dims = d, .num_dims = 1, .keepdim = false};
        run(uop_argmax(a8, &rp1));
        run(uop_cmpgt(a8, b8));
        run(uop_abs(a8));
    }
    tensor_free(a8);
    tensor_free(b8);
    return !g_fail;
}

/* ---- creation ops ---- */
static int test_creation_ops_impl(void) {
    run(uop_fill_ex((int[]){2, 2}, 2, 3.0f, DTYPE_FLOAT32, DEVICE_CPU));
    {
        float v = 1.5f;
        run(uop_const(&v, sizeof(v), (int[]){2, 2}, 2, DTYPE_FLOAT32, DEVICE_CPU));
    }
    run(uop_arange_op(0.0f, 6.0f, 1.0f, DTYPE_FLOAT32, DEVICE_CPU));
    run(uop_eye_op(3, DTYPE_FLOAT32, DEVICE_CPU));
    run(uop_rand_uniform((int[]){2, 2}, 2, DTYPE_FLOAT32, DEVICE_CPU));
    run(uop_rand_normal((int[]){2, 2}, 2, DTYPE_FLOAT32, DEVICE_CPU));
    run(uop_alloc((int[]){2, 2}, 2, DTYPE_FLOAT32, DEVICE_CPU));
    run(uop_arange_op(0.0f, 5.0f, -1.0f, DTYPE_FLOAT32, DEVICE_CPU)); /* bad step */
    run(uop_eye_op(-1, DTYPE_FLOAT32, DEVICE_CPU));                   /* bad size */
    return !g_fail;
}

/* ---- full matrix under current flags ---- */
static int run_matrix(const char* name) {
    g_fail = 0;
    for (int i = 0; i < NUN; i++) {
        run(UN[i].f(mk(2, 3)));
        run(UN[i].f(mk1(6)));
        run(UN[i].f(mk3()));
        run(UN[i].f(mk_empty()));
        run(UN[i].f(mk(1, 1)));
    }
    int ok = phase_binary_inner();
    ok &= phase_reductions_inner();
    ok &= phase_shape_ops_inner();
    ok &= phase_dtypes_inner();
    ok &= test_creation_ops_impl();
    tests_run++;
    if (ok && !g_fail) {
        tests_passed++;
        printf("  (%s) [PASS]\n", name);
    } else
        printf("  (%s) [FAIL]\n", name);
    return ok && !g_fail;
}

static int test_default_arm(void) { return run_matrix("arm: default (fusion+jit)"); }
static int test_interpreter_arm(void) {
    setenv("DISABLE_FUSION", "1", 1);
    setenv("JIT", "0", 1);
    int ok = run_matrix("arm: interpreter (no fusion/jit)");
    unsetenv("DISABLE_FUSION");
    setenv("JIT", "1", 1);
    return ok;
}
static int test_poly_arm(void) {
    setenv("TRANSCENDENTAL", "2", 1);
    int ok = run_matrix("arm: polynomial transcendentals");
    setenv("TRANSCENDENTAL", "1", 1);
    return ok;
}
static int test_wino_on(void) {
    setenv("WINO", "1", 1);
    int ok = run_matrix("arm: winograd conv forced");
    unsetenv("WINO");
    return ok;
}
static int test_wino_off(void) {
    setenv("WINO", "0", 1);
    int ok = run_matrix("arm: winograd conv off");
    unsetenv("WINO");
    return ok;
}
static int test_check_oob_arm(void) {
    setenv("CHECK_OOB", "1", 1);
    int ok = run_matrix("arm: CHECK_OOB bounds checking");
    unsetenv("CHECK_OOB");
    return ok;
}
static int test_noopt_arm(void) {
    setenv("NOOPT", "1", 1);
    setenv("SPLIT_REDUCEOP", "0", 1);
    int ok = run_matrix("arm: NOOPT + unsplit reduce");
    unsetenv("NOOPT");
    setenv("SPLIT_REDUCEOP", "1", 1);
    return ok;
}

int main(void) {
    cml_init();

    printf("=== executor op matrix ===\n");
    TEST_CASE(default_arm);
    TEST_CASE(interpreter_arm);
    TEST_CASE(poly_arm);
    TEST_CASE(wino_on);
    TEST_CASE(wino_off);
    TEST_CASE(check_oob_arm);
    TEST_CASE(noopt_arm);

    cml_cleanup();
    return TEST_SUMMARY();
}
