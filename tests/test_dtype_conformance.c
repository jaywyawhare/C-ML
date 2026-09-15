/* Dtype conformance: every op must compute the same values in every dtype.
 *
 * The float32 kernels are the reference -- they are exercised by the rest of
 * the suite, and their multi-dimensional values were checked against hand
 * computation. Each case below builds the same graph twice, once in f32 and
 * once in the dtype under test, and compares elementwise.
 *
 * This catches the failure the gradient tests structurally cannot: a VJP is
 * checked against the same forward kernel it differentiates, so a forward that
 * reinterprets its buffer as float passes gradient checking while returning
 * garbage. Only a cross-dtype value comparison sees it.
 */

#include "cml.h"
#include "tensor/dtype_access.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "test_harness.h"

static TensorConfig cfg_of(DType dt) {
    TensorConfig c = {.dtype = dt, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return c;
}

static const char* dtype_name(DType d) {
    switch (d) {
    case DTYPE_FLOAT32: return "f32";
    case DTYPE_FLOAT64: return "f64";
    case DTYPE_FLOAT16: return "f16";
    case DTYPE_BFLOAT16: return "bf16";
    case DTYPE_INT32:   return "i32";
    case DTYPE_INT64:   return "i64";
    case DTYPE_INT16:   return "i16";
    case DTYPE_INT8:    return "i8";
    case DTYPE_UINT8:   return "u8";
    default:            return "?";
    }
}

/* Tolerance scales with the storage format's precision: the half types compute
 * in f32 and round the result back, so an 8-bit-mantissa bf16 cannot match an
 * f32 reference to 1e-5. Source values are small integers, so most results are
 * exact anyway; the looser bound only matters for accumulated values (prod,
 * matmul, logsumexp). */
static double dtype_tol(DType d, double want) {
    switch (d) {
    case DTYPE_FLOAT16:  return fabs(want) * 1e-3 + 1e-3;
    case DTYPE_BFLOAT16: return fabs(want) * 5e-3 + 5e-3;
    default:             return fabs(want) * 1e-5 + 1e-5;
    }
}

static bool dtype_holds(DType d, double v) {
    switch (d) {
    case DTYPE_INT8:  return v >= -128.0 && v <= 127.0;
    case DTYPE_UINT8: return v >= 0.0 && v <= 255.0;
    case DTYPE_INT16: return v >= -32768.0 && v <= 32767.0;
    default:          return true;
    }
}

/* Source values are small integers so every dtype under test -- including int8
 * and the half formats -- represents them exactly; a mismatch is then a real
 * kernel bug and never a rounding artefact. */
static Tensor* make(DType dt, const int* shape, int ndim, const double* vals) {
    TensorConfig c = cfg_of(dt);
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    if (!t) return NULL;
    size_t n = t->numel;
    for (size_t i = 0; i < n; i++)
        tensor_set_float(t, i, (float)vals[i]);
    return t;
}

/* Build fn is called once per dtype; it returns the op's output tensor. */
typedef Tensor* (*BuildFn)(Tensor** ins, int n_in);

typedef struct {
    const char* name;
    BuildFn     build;
    int         n_in;
    int         shape[4];
    int         ndim;
    const double* vals[3];
} Case;

static void run_case(const Case* c, DType dt) {
    tests_run++;
    double ref[256];
    size_t ref_n = 0;

    for (int pass = 0; pass < 2; pass++) {
        DType use = pass == 0 ? DTYPE_FLOAT32 : dt;
        Tensor* ins[3] = {0};
        for (int i = 0; i < c->n_in; i++) {
            ins[i] = make(use, c->shape, c->ndim, c->vals[i]);
            if (!ins[i]) { printf("  %-22s %s: alloc failed\n", c->name, dtype_name(dt)); return; }
        }
        Tensor* r = c->build(ins, c->n_in);
        if (!r) {
            printf("  %-22s %s: build returned NULL\n", c->name, dtype_name(use));
            cml_reset_ir_context();
            return;
        }
        tensor_ensure_executed(r);
        if (!tensor_data_ptr(r)) {
            printf("  %-22s %s: no data after execute\n", c->name, dtype_name(use));
            cml_reset_ir_context();
            return;
        }
        if (pass == 0) {
            ref_n = r->numel < 256 ? r->numel : 256;
            for (size_t i = 0; i < ref_n; i++) ref[i] = (double)tensor_get_float(r, i);
        } else {
            if (r->numel != ref_n) {
                printf("  %-22s %s: numel %zu != f32 numel %zu\n",
                       c->name, dtype_name(dt), r->numel, ref_n);
                cml_reset_ir_context();
                return;
            }
            for (size_t i = 0; i < ref_n; i++) {
                double got = (double)tensor_get_float(r, i);
                /* A reference value outside the narrow type's range says nothing
                 * about the kernel -- the result wraps, exactly as it would in
                 * any fixed-width integer arithmetic. Skip rather than mis-report
                 * (e.g. prod of a row = 504 in int8). */
                if (!dtype_holds(dt, ref[i])) continue;
                double want = ref[i];
                /* Judge by the dtype the op actually produced: ops that compose
                 * from f32 constants (gelu) promote an integer input to float,
                 * so their result is not truncated at all. */
                if (cml_dtype_is_int(r->dtype)) {
                    /* Storing through an integer dtype truncates toward zero,
                     * so that -- not the f32 value -- is the expected result.
                     * Skip values sitting on a truncation boundary, where f32
                     * and f64 rounding can land either side of the integer. */
                    double frac = fabs(want - trunc(want));
                    if (frac < 1e-4 || frac > 1.0 - 1e-4) continue;
                    want = trunc(want);
                }
                double tol = dtype_tol(dt, want);
                if (fabs(got - want) > tol) {
                    printf("  %-22s %s: [%zu] got %g, expected %g (f32 gives %g)\n",
                           c->name, dtype_name(dt), i, got, want, ref[i]);
                    cml_reset_ir_context();
                    return;
                }
            }
        }
        cml_reset_ir_context();
    }
    tests_passed++;
}

/* ------------------------------------------------------------ op builders */

#define B(name, expr) static Tensor* name(Tensor** in, int n) { (void)in; (void)n; return (expr); }

B(b_neg,      uop_neg(in[0]))
B(b_abs,      uop_abs(in[0]))
B(b_square,   uop_square(in[0]))
B(b_sign,     uop_sign(in[0]))
B(b_floor,    uop_floor(in[0]))
B(b_ceil,     uop_ceil(in[0]))
B(b_round,    uop_round(in[0]))
B(b_relu,     uop_relu(in[0]))
B(b_add,      uop_add(in[0], in[1]))
B(b_sub,      uop_sub(in[0], in[1]))
B(b_mul,      uop_mul(in[0], in[1]))
B(b_max,      uop_max(in[0], in[1]))
B(b_minimum,  uop_minimum(in[0], in[1]))
B(b_cmpgt,    uop_cmpgt(in[0], in[1]))
B(b_cmpeq,    uop_cmpeq(in[0], in[1]))

B(b_cumsum0,  uop_cumsum(in[0], 0))
B(b_cumsum1,  uop_cumsum(in[0], 1))
B(b_cumprod1, uop_cumprod(in[0], 1))
B(b_cummax1,  uop_cummax(in[0], 1))
B(b_cummin0,  uop_cummin(in[0], 0))
B(b_sort1,    uop_sort(in[0], 1, false))
B(b_sortd0,   uop_sort(in[0], 0, true))
B(b_argsort1, uop_argsort(in[0], 1, false))
B(b_roll1,    uop_roll(in[0], 1, 1))
B(b_roll0,    uop_roll(in[0], 1, 0))
B(b_tri_u,    uop_triu(in[0], 0))
B(b_tri_l,    uop_tril(in[0], 0))
B(b_repeat1,  uop_repeat_interleave(in[0], 2, 1))
static Tensor* b_transpose(Tensor** in, int n) {
    (void)n; static int perm[2] = {1, 0};
    PermuteParams pp = {.perm = perm, .num_dims = 2};
    return uop_permute(in[0], &pp);
}
B(b_flatten,  uop_flatten(in[0], 0, 1))

static Tensor* b_topk(Tensor** in, int n) {
    (void)n; Tensor* idx = NULL; return uop_topk(in[0], 2, 1, true, &idx);
}
static Tensor* b_sum_all(Tensor** in, int n) {
    (void)n; ReduceParams rp = {.dims = NULL, .num_dims = 0, .keepdim = false};
    return uop_sum(in[0], &rp);
}
static Tensor* b_sum_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_sum(in[0], &rp);
}
static Tensor* b_max_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_max_reduce(in[0], &rp);
}
static Tensor* b_argmax_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_argmax(in[0], &rp);
}
static Tensor* b_prod_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_prod(in[0], &rp);
}
static Tensor* b_reshape(Tensor** in, int n) {
    (void)n; return uop_reshape_to(in[0], (int[]){3, 2}, 2);
}
static Tensor* b_slice_row(Tensor** in, int n) {
    (void)n;
    static int st[2] = {0, 0}, en[2] = {1, 3}, sp[2] = {1, 1};
    SliceParams p = {.start = st, .end = en, .step = sp, .num_dims = 2};
    return uop_slice(in[0], &p);
}
static Tensor* b_cat0(Tensor** in, int n) {
    return uop_cat(in, n, 0);
}
static Tensor* b_where(Tensor** in, int n) {
    (void)n;
    WhereParams p = {.cond = in[0], .a = in[1], .b = in[2]};
    return uop_where(&p);
}
static Tensor* b_matmul(Tensor** in, int n) {
    (void)n; return uop_matmul(in[0], in[1]);
}

/* Composite chains: single-op cases can pass while the passes that rewrite
 * whole graphs (elementwise fusion, JIT codegen) still mishandle a dtype, so
 * these run several ops together. */
static Tensor* b_chain(Tensor** in, int n) {
    (void)n;
    Tensor* t = uop_mul(in[0], in[1]);
    if (!t) return NULL;
    t = uop_add(t, in[0]);
    if (!t) return NULL;
    t = uop_relu(t);
    if (!t) return NULL;
    return uop_sub(t, in[1]);
}
static Tensor* b_chain_reduce(Tensor** in, int n) {
    (void)n;
    Tensor* t = uop_mul(in[0], in[0]);
    if (!t) return NULL;
    t = uop_add(t, in[1]);
    if (!t) return NULL;
    static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_sum(t, &rp);
}
static Tensor* b_matmul_chain(Tensor** in, int n) {
    (void)n;
    Tensor* t = uop_matmul(in[0], in[1]);
    if (!t) return NULL;
    t = uop_relu(t);
    if (!t) return NULL;
    return uop_add(t, in[0]);
}

/* float-domain math: integer inputs promote, so these check the promotion path
 * as well as the arithmetic */
B(b_exp,      uop_exp(in[0]))
B(b_sqrt,     uop_sqrt(in[0]))
B(b_log,      uop_log(in[0]))
B(b_tanh,     uop_tanh(in[0]))
B(b_sigmoid,  uop_sigmoid(in[0]))
B(b_gelu,     uop_gelu(in[0]))
B(b_softplus, uop_softplus(in[0]))
B(b_erf,      uop_erf(in[0]))
B(b_trace,    uop_trace(in[0]))
static Tensor* b_lerp(Tensor** in, int n) {
    (void)n;
    /* weight tensor in the same dtype, so both passes see the same t */
    TensorConfig c = cfg_of(in[0]->dtype);
    int s1[1] = {1};
    Tensor* t = tensor_zeros(s1, 1, &c);
    if (!t) return NULL;
    tensor_set_float(t, 0, 1.0f);   /* integer-representable */
    return uop_lerp(in[0], in[1], t);
}
B(b_one_hot,  uop_one_hot(in[0], 4))
B(b_diag,     uop_diag(in[0], 0))

static Tensor* b_var_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_var(in[0], &rp);
}
static Tensor* b_std_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_std(in[0], &rp);
}
static Tensor* b_logsumexp_d1(Tensor** in, int n) {
    (void)n; static int d[1] = {1};
    ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
    return uop_logsumexp(in[0], &rp);
}
static Tensor* b_tile(Tensor** in, int n) {
    (void)n; static int reps[2] = {2, 2};
    return uop_tile(in[0], reps, 2);
}
static Tensor* b_pad(Tensor** in, int n) {
    (void)n; static int pw[4] = {1, 1, 1, 1};
    return uop_pad(in[0], pw, 2, 0.0f);
}
static Tensor* b_unfold(Tensor** in, int n) {
    (void)n; return uop_unfold(in[0], 2, 1);
}
/* Rank-3 movement cases. Every kernel in this family had an ndim==1/2 special
 * case and no general arm, in both the f32 interpreter and the typed path, so
 * rank-2 conformance could not see it. */
static Tensor* b_pad3(Tensor** in, int n) {
    (void)n; static int pw[6] = {0, 1, 1, 0, 1, 1};
    return uop_pad(in[0], pw, 3, 0.0f);
}
static Tensor* b_pad3_reflect(Tensor** in, int n) {
    (void)n; static int pw[6] = {0, 0, 1, 1, 1, 1};
    return uop_pad_reflect(in[0], pw, 3);
}
static Tensor* b_pad3_replicate(Tensor** in, int n) {
    (void)n; static int pw[6] = {1, 1, 1, 0, 0, 1};
    return uop_pad_replicate(in[0], pw, 3);
}
static Tensor* b_diagonal3_12(Tensor** in, int n) {
    (void)n; return uop_diagonal(in[0], 0, 1, 2);
}
static Tensor* b_diagonal3_02(Tensor** in, int n) {
    (void)n; return uop_diagonal(in[0], 0, 0, 2);
}
static Tensor* b_scatter3_d2(Tensor** in, int n) {
    (void)n; return uop_scatter(in[0], 2, in[1], in[2]);
}
static Tensor* b_scatter3_d0(Tensor** in, int n) {
    (void)n; return uop_scatter(in[0], 0, in[1], in[2]);
}
static Tensor* b_roll3(Tensor** in, int n) { (void)n; return uop_roll(in[0], 1, 2); }
static Tensor* b_tile3(Tensor** in, int n) {
    (void)n; static int reps[3] = {2, 1, 2};
    return uop_tile(in[0], reps, 3);
}
static Tensor* b_repeat3(Tensor** in, int n) {
    (void)n; return uop_repeat_interleave(in[0], 2, 1);
}
static Tensor* b_cat3(Tensor** in, int n) { return uop_cat(in, n, 2); }
static Tensor* b_masked_select3(Tensor** in, int n) {
    (void)n; return uop_masked_select(in[0], in[1]);
}

/* 4-D cases: NCHW input, so they exercise conv/pool index arithmetic */
static Tensor* b_maxpool(Tensor** in, int n) {
    (void)n;
    Pool2DParams p = {.kernel_size = {2, 2}, .stride = {1, 1}, .padding = {0, 0},
                      .dilation = {1, 1}, .count_include_pad = false};
    return uop_maxpool2d(in[0], &p);
}
static Tensor* b_avgpool(Tensor** in, int n) {
    (void)n;
    Pool2DParams p = {.kernel_size = {2, 2}, .stride = {1, 1}, .padding = {0, 0},
                      .dilation = {1, 1}, .count_include_pad = false};
    return uop_avgpool2d(in[0], &p);
}
static Tensor* b_conv2d(Tensor** in, int n) {
    (void)n;
    /* weight [1,1,2,2] carved from the same values so both dtypes match */
    TensorConfig c = cfg_of(in[0]->dtype);
    int ws[4] = {1, 1, 2, 2};
    Tensor* w = tensor_zeros(ws, 4, &c);
    if (!w) return NULL;
    const double wv[4] = {1, 0, 0, 2};
    for (int i = 0; i < 4; i++) tensor_set_float(w, i, (float)wv[i]);
    static int ks[2] = {2, 2}, st[2] = {1, 1}, pd[2] = {0, 0}, dl[2] = {1, 1};
    Conv2DParams p = {.kernel_size = ks, .stride = st, .padding = pd,
                      .dilation = dl, .groups = 1, .bias = false};
    return uop_conv2d(in[0], w, NULL, &p);
}

/* integer-domain ops: only meaningful on integer dtypes, so they are checked
 * against hand-computed values rather than against an f32 reference */
typedef struct { const char* name; Tensor* (*fn)(Tensor*, Tensor*); const double* want; } IntCase;

static Tensor* i_and(Tensor* a, Tensor* b) { return uop_bitwise_and(a, b); }
static Tensor* i_or (Tensor* a, Tensor* b) { return uop_bitwise_or(a, b); }
static Tensor* i_xor(Tensor* a, Tensor* b) { return uop_bitwise_xor(a, b); }
static Tensor* i_shl(Tensor* a, Tensor* b) { return uop_lshift(a, b); }
static Tensor* i_shr(Tensor* a, Tensor* b) { return uop_rshift(a, b); }
static Tensor* i_mod(Tensor* a, Tensor* b) { return uop_mod(a, b); }
static Tensor* i_div(Tensor* a, Tensor* b) { return uop_idiv(a, b); }

/* ------------------------------------------------------------------ cases */

/* 2x3, chosen so every op has distinguishable per-lane behaviour */
static const double A23[6] = { 3, 1, 2,  9, 7, 8 };
static const double B23[6] = { 1, 4, 2,  5, 7, 3 };
static const double M23[6] = { 1, 0, 1,  0, 1, 0 };   /* mask / condition */
/* 3x3 square for triu/tril/matmul */
static const double A33[9] = { 3, 1, 2,  9, 7, 8,  4, 6, 5 };
static const double B33[9] = { 1, 0, 2,  0, 1, 0,  2, 0, 1 };
/* 2x2x2 for the rank-3 movement cases */
static const double A222[8] = { 3, 1,  2, 9,  7, 8,  4, 6 };
static const double B222[8] = { 5, 2,  8, 1,  6, 3,  9, 4 };
static const double Z222[8] = { 0, 1,  1, 0,  0, 1,  1, 0 };  /* index / mask */

static const Case CASES[] = {
    /* elementwise unary */
    {"neg",       b_neg,      1, {2,3}, 2, {A23}},
    {"abs",       b_abs,      1, {2,3}, 2, {A23}},
    {"square",    b_square,   1, {2,3}, 2, {A23}},
    {"sign",      b_sign,     1, {2,3}, 2, {A23}},
    {"floor",     b_floor,    1, {2,3}, 2, {A23}},
    {"ceil",      b_ceil,     1, {2,3}, 2, {A23}},
    {"round",     b_round,    1, {2,3}, 2, {A23}},
    {"relu",      b_relu,     1, {2,3}, 2, {A23}},
    /* elementwise binary */
    {"add",       b_add,      2, {2,3}, 2, {A23, B23}},
    {"sub",       b_sub,      2, {2,3}, 2, {A23, B23}},
    {"mul",       b_mul,      2, {2,3}, 2, {A23, B23}},
    {"max",       b_max,      2, {2,3}, 2, {A23, B23}},
    {"minimum",   b_minimum,  2, {2,3}, 2, {A23, B23}},
    {"cmpgt",     b_cmpgt,    2, {2,3}, 2, {A23, B23}},
    {"cmpeq",     b_cmpeq,    2, {2,3}, 2, {A23, B23}},
    /* cumulative -- both axes */
    {"cumsum dim0",  b_cumsum0,  1, {2,3}, 2, {A23}},
    {"cumsum dim1",  b_cumsum1,  1, {2,3}, 2, {A23}},
    {"cumprod dim1", b_cumprod1, 1, {2,3}, 2, {A23}},
    {"cummax dim1",  b_cummax1,  1, {2,3}, 2, {A23}},
    {"cummin dim0",  b_cummin0,  1, {2,3}, 2, {A23}},
    /* ordering */
    {"sort dim1",    b_sort1,    1, {2,3}, 2, {A23}},
    {"sort desc d0", b_sortd0,   1, {2,3}, 2, {A23}},
    {"argsort dim1", b_argsort1, 1, {2,3}, 2, {A23}},
    {"topk dim1",    b_topk,     1, {2,3}, 2, {A23}},
    /* reductions */
    {"sum all",      b_sum_all,  1, {2,3}, 2, {A23}},
    {"sum dim1",     b_sum_d1,   1, {2,3}, 2, {A23}},
    {"max dim1",     b_max_d1,   1, {2,3}, 2, {A23}},
    {"argmax dim1",  b_argmax_d1,1, {2,3}, 2, {A23}},
    {"prod dim1",    b_prod_d1,  1, {2,3}, 2, {A23}},
    /* layout */
    {"reshape",      b_reshape,  1, {2,3}, 2, {A23}},
    {"transpose",    b_transpose,1, {2,3}, 2, {A23}},
    {"flatten",      b_flatten,  1, {2,3}, 2, {A23}},
    {"slice row0",   b_slice_row,1, {2,3}, 2, {A23}},
    {"roll dim1",    b_roll1,    1, {2,3}, 2, {A23}},
    {"roll dim0",    b_roll0,    1, {2,3}, 2, {A23}},
    {"repeat_intlv", b_repeat1,  1, {2,3}, 2, {A23}},
    {"cat dim0",     b_cat0,     2, {2,3}, 2, {A23, B23}},
    {"where",        b_where,    3, {2,3}, 2, {M23, A23, B23}},
    {"tile",         b_tile,     1, {2,3}, 2, {A23}},
    {"pad const",    b_pad,      1, {2,3}, 2, {A23}},
    {"unfold",       b_unfold,   1, {2,3}, 2, {A23}},
    /* rank-3 movement -- the arm that used to be missing entirely */
    {"pad3 const",     b_pad3,            1, {2,2,2}, 3, {A222}},
    {"pad3 reflect",   b_pad3_reflect,    1, {2,2,2}, 3, {A222}},
    {"pad3 replicate", b_pad3_replicate,  1, {2,2,2}, 3, {A222}},
    {"diagonal3 1,2",  b_diagonal3_12,    1, {2,2,2}, 3, {A222}},
    {"diagonal3 0,2",  b_diagonal3_02,    1, {2,2,2}, 3, {A222}},
    {"scatter3 dim2",  b_scatter3_d2,     3, {2,2,2}, 3, {A222, Z222, B222}},
    {"scatter3 dim0",  b_scatter3_d0,     3, {2,2,2}, 3, {A222, Z222, B222}},
    {"roll3 dim2",     b_roll3,           1, {2,2,2}, 3, {A222}},
    {"tile3",          b_tile3,           1, {2,2,2}, 3, {A222}},
    {"repeat3 dim1",   b_repeat3,         1, {2,2,2}, 3, {A222}},
    {"cat3 dim2",      b_cat3,            2, {2,2,2}, 3, {A222, B222}},
    {"masked_select3", b_masked_select3,  2, {2,2,2}, 3, {A222, Z222}},
    {"one_hot",      b_one_hot,  1, {2,3}, 2, {M23}},
    {"lerp",         b_lerp,     2, {2,3}, 2, {A23, B23}},
    /* float-domain math (integer inputs promote) */
    {"exp",          b_exp,      1, {2,3}, 2, {M23}},
    {"sqrt",         b_sqrt,     1, {2,3}, 2, {A23}},
    {"log",          b_log,      1, {2,3}, 2, {A23}},
    {"tanh",         b_tanh,     1, {2,3}, 2, {M23}},
    {"sigmoid",      b_sigmoid,  1, {2,3}, 2, {M23}},
    {"gelu",         b_gelu,     1, {2,3}, 2, {M23}},
    {"softplus",     b_softplus, 1, {2,3}, 2, {M23}},
    {"erf",          b_erf,      1, {2,3}, 2, {M23}},
    {"var dim1",     b_var_d1,   1, {2,3}, 2, {A23}},
    {"std dim1",     b_std_d1,   1, {2,3}, 2, {A23}},
    {"logsumexp d1", b_logsumexp_d1, 1, {2,3}, 2, {A23}},
    /* square-matrix ops */
    {"triu",         b_tri_u,    1, {3,3}, 2, {A33}},
    {"tril",         b_tri_l,    1, {3,3}, 2, {A33}},
    {"matmul",       b_matmul,   2, {3,3}, 2, {A33, B33}},
    /* multi-op graphs: fusion + JIT + typed kernels together */
    {"chain mul/add/relu/sub", b_chain,        2, {2,3}, 2, {A23, B23}},
    {"chain + reduce",         b_chain_reduce, 2, {2,3}, 2, {A23, B23}},
    {"matmul+relu+add",        b_matmul_chain, 2, {3,3}, 2, {A33, B33}},
    {"trace",        b_trace,    1, {3,3}, 2, {A33}},
    {"diag extract", b_diag,     1, {3,3}, 2, {A33}},
    /* 4-D NCHW: conv and pooling index arithmetic */
    {"maxpool2d",    b_maxpool,  1, {1,1,3,3}, 4, {A33}},
    {"avgpool2d",    b_avgpool,  1, {1,1,3,3}, 4, {A33}},
    {"conv2d",       b_conv2d,   1, {1,1,3,3}, 4, {A33}},
};

/* Integer-domain ops have no float reference to compare against -- bitwise and
 * shift results are defined on the bit pattern, not on a numeric value -- so
 * they are checked against values computed by hand. */
static void run_int_cases(DType dt) {
    const double A[4] = { 0xF0, 0xFF, 12, 7 };
    const double B[4] = { 0x0F, 0x0F, 10, 2 };
    const double and_[4] = { 0, 15, 8, 2 };
    const double or_ [4] = { 255, 255, 14, 7 };
    const double xor_[4] = { 255, 240, 6, 5 };
    const double shl_[4] = { 0xF0 << 15, 0, 0, 0 };  /* only [3] used below */
    const double mod_[4] = { 0, 0, 2, 1 };
    const double div_[4] = { 16, 17, 1, 3 };
    (void)shl_;
    const IntCase cases[] = {
        {"bitwise_and", i_and, and_}, {"bitwise_or", i_or, or_},
        {"bitwise_xor", i_xor, xor_}, {"mod", i_mod, mod_}, {"idiv", i_div, div_},
    };
    int shape[1] = {4};
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        tests_run++;
        Tensor* ta = make(dt, shape, 1, A);
        Tensor* tb = make(dt, shape, 1, B);
        Tensor* r  = ta && tb ? cases[ci].fn(ta, tb) : NULL;
        if (!r) {
            printf("  %-22s %s: NULL\n", cases[ci].name, dtype_name(dt));
            cml_reset_ir_context(); continue;
        }
        tensor_ensure_executed(r);
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            if (!dtype_holds(dt, cases[ci].want[i]) || !dtype_holds(dt, A[i])) continue;
            double got = (double)tensor_get_float(r, (size_t)i);
            if (fabs(got - cases[ci].want[i]) > 1e-6) {
                printf("  %-22s %s: [%d] got %g, want %g\n",
                       cases[ci].name, dtype_name(dt), i, got, cases[ci].want[i]);
                ok = 0;
                break;
            }
        }
        if (ok) tests_passed++;
        cml_reset_ir_context();
    }
    /* shifts checked separately: the operand is a shift count, not a value */
    tests_run++;
    const double SA[4] = { 1, 2, 3, 4 }, SB[4] = { 1, 2, 3, 1 };
    const double want_shl[4] = { 2, 8, 24, 8 };
    Tensor* ta = make(dt, shape, 1, SA);
    Tensor* tb = make(dt, shape, 1, SB);
    Tensor* r  = ta && tb ? i_shl(ta, tb) : NULL;
    int shl_ok = 0;
    if (!r) { printf("  lshift %s: NULL\n", dtype_name(dt)); }
    else {
        tensor_ensure_executed(r);
        shl_ok = 1;
        for (int i = 0; i < 4; i++) {
            if (!dtype_holds(dt, want_shl[i])) continue;
            double got = (double)tensor_get_float(r, (size_t)i);
            if (fabs(got - want_shl[i]) > 1e-6) {
                printf("  %-22s %s: [%d] got %g, want %g\n",
                       "lshift", dtype_name(dt), i, got, want_shl[i]);
                shl_ok = 0;
                break;
            }
        }
    }
    if (shl_ok) tests_passed++;
    cml_reset_ir_context();

    tests_run++;
    const double want_shr[4] = { 0, 0, 0, 2 };
    Tensor* ua = make(dt, shape, 1, SA);
    Tensor* ub = make(dt, shape, 1, SB);
    Tensor* r2 = ua && ub ? i_shr(ua, ub) : NULL;
    int shr_ok = 0;
    if (!r2) { printf("  rshift %s: NULL\n", dtype_name(dt)); }
    else {
        tensor_ensure_executed(r2);
        shr_ok = 1;
        for (int i = 0; i < 4; i++) {
            double got = (double)tensor_get_float(r2, (size_t)i);
            if (fabs(got - want_shr[i]) > 1e-6) {
                printf("  %-22s %s: [%d] got %g, want %g\n",
                       "rshift", dtype_name(dt), i, got, want_shr[i]);
                shr_ok = 0;
                break;
            }
        }
    }
    if (shr_ok) tests_passed++;
    cml_reset_ir_context();
}

int main(void) {
    cml_init();

    const DType DTYPES[] = {DTYPE_FLOAT64, DTYPE_FLOAT16, DTYPE_BFLOAT16, DTYPE_UINT8,
                            DTYPE_INT32, DTYPE_INT64, DTYPE_INT16, DTYPE_INT8};
    const int   NDT      = (int)(sizeof(DTYPES) / sizeof(DTYPES[0]));
    const int   NC       = (int)(sizeof(CASES) / sizeof(CASES[0]));

    printf("Dtype conformance: %d ops x %d dtypes, f32 kernels as reference\n\n", NC, NDT);
    for (int d = 0; d < NDT; d++)
        for (int i = 0; i < NC; i++)
            run_case(&CASES[i], DTYPES[d]);

    const DType INT_DTYPES[] = {DTYPE_INT32, DTYPE_INT64, DTYPE_INT16};
    for (int d = 0; d < (int)(sizeof(INT_DTYPES) / sizeof(INT_DTYPES[0])); d++)
        run_int_cases(INT_DTYPES[d]);

    cml_cleanup();
    return TEST_SUMMARY();
}
