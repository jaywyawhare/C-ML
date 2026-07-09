/*
 * Numerical parity gate for the shape-specialized LLVM SIMD emission.
 *
 * For every op it compares THREE producers of the same result:
 *   (ref)  a naive scalar C loop           — the ground truth
 *   (jit)  the executed tensor-op path      — shape-specialized JIT kernels
 *          (llvm_backend.c) when the LLVM backend is enabled, else the CPU path
 *   (simd) the hand-rolled simd_* functions — validated before they are removed
 *
 * The size matrix deliberately straddles SSE/AVX2/AVX-512 vector-width
 * boundaries (…7,8,15,16,17,63,64,255,256…) so remainder handling and
 * alignment edge cases are exercised.  Binary ops additionally cover the
 * broadcast cases (equal / lhs-scalar / rhs-scalar).
 *
 * This is the correctness gate referenced by docs/shape_specific_simd_plan.md:
 * it must pass before the hand-rolled SIMD library is deleted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/simd_math.h"
#include "ops/simd_utils.h"

static int g_pass = 0, g_total = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
    .has_dtype = true, .has_device = true
};

/* Sizes chosen to hit sub-width, aligned, and remainder cases. */
static const int SIZES[] = { 1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 127, 256, 1024, 4097 };
static const int N_SIZES = (int)(sizeof(SIZES) / sizeof(SIZES[0]));

/* Deterministic PRNG so failures reproduce (no Date/rand global state). */
static uint32_t s_rng = 0x9e3779b9u;
static float randf(float lo, float hi) {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
    float u = (float)(s_rng & 0xffffff) / (float)0xffffff;
    return lo + u * (hi - lo);
}

static int close(float a, float b, float atol, float rtol) {
    float d = fabsf(a - b);
    if (d <= atol) return 1;
    return d <= rtol * (0.5f * (fabsf(a) + fabsf(b)) + 1e-8f);
}

/* ---- unary ------------------------------------------------------------ */
typedef Tensor* (*unary_op_fn)(Tensor*);
typedef float   (*unary_ref_fn)(float);
typedef void    (*unary_simd_fn)(const float*, float*, size_t);

static void check_unary(const char* name, unary_op_fn op, unary_ref_fn ref,
                        unary_simd_fn simd, float lo, float hi,
                        float atol, float rtol) {
    g_total++;
    int failures = 0;
    for (int si = 0; si < N_SIZES; si++) {
        int n = SIZES[si];
        float* in  = malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++) in[i] = randf(lo, hi);

        Tensor* x = tensor_from_data(in, (int[]){n}, 1, &cpu_f32);
        Tensor* y = op(x);
        tensor_ensure_executed(y);
        float* jit = (float*)y->data;

        float* sim = NULL;
        if (simd) { sim = malloc(sizeof(float) * n); simd(in, sim, (size_t)n); }

        for (int i = 0; i < n; i++) {
            float r = ref(in[i]);
            /* accuracy vs libm (loose for fast-approx transcendentals) */
            if (!close(jit[i], r, atol, rtol)) {
                if (failures < 3)
                    printf("    [%s n=%d i=%d] jit=%.6g ref=%.6g in=%.6g\n",
                           name, n, i, jit[i], r, in[i]);
                failures++;
            }
            /* PARITY: executed path must match the hand-rolled simd it replaces */
            if (sim && !close(jit[i], sim[i], 1e-5f, 1e-4f)) {
                if (failures < 3)
                    printf("    [%s n=%d i=%d] PARITY jit=%.6g simd=%.6g\n",
                           name, n, i, jit[i], sim[i]);
                failures++;
            }
        }
        free(in); free(sim);
        tensor_free(x); tensor_free(y);
    }
    if (failures == 0) { g_pass++; printf("  PASS: unary %-10s (%d sizes)\n", name, N_SIZES); }
    else               { printf("  FAIL: unary %-10s (%d mismatches)\n", name, failures); }
}

/* ---- binary (with broadcast modes) ------------------------------------ */
typedef Tensor* (*binary_op_fn)(Tensor*, Tensor*);
typedef float   (*binary_ref_fn)(float, float);
typedef void    (*binary_simd_fn)(const float*, const float*, float*, size_t);

/* mode: 0 equal, 1 lhs scalar, 2 rhs scalar */
static void check_binary(const char* name, binary_op_fn op, binary_ref_fn ref,
                         binary_simd_fn simd, float lo, float hi,
                         float atol, float rtol) {
    g_total++;
    int failures = 0;
    for (int mode = 0; mode < 3; mode++) {
        for (int si = 0; si < N_SIZES; si++) {
            int n = SIZES[si];
            int an = (mode == 1) ? 1 : n;
            int bn = (mode == 2) ? 1 : n;
            float* a = malloc(sizeof(float) * an);
            float* b = malloc(sizeof(float) * bn);
            for (int i = 0; i < an; i++) a[i] = randf(lo, hi);
            for (int i = 0; i < bn; i++) b[i] = randf(lo, hi);

            Tensor* ta = tensor_from_data(a, (int[]){an}, 1, &cpu_f32);
            Tensor* tb = tensor_from_data(b, (int[]){bn}, 1, &cpu_f32);
            Tensor* tc = op(ta, tb);
            tensor_ensure_executed(tc);
            float* jit = (float*)tc->data;

            /* hand-rolled simd only covers the equal-size contiguous case */
            float* sim = NULL;
            if (simd && mode == 0) { sim = malloc(sizeof(float) * n); simd(a, b, sim, (size_t)n); }

            for (int i = 0; i < n; i++) {
                float av = a[mode == 1 ? 0 : i];
                float bv = b[mode == 2 ? 0 : i];
                float r  = ref(av, bv);
                if (!close(jit[i], r, atol, rtol)) {
                    if (failures < 3)
                        printf("    [%s mode=%d n=%d i=%d] jit=%.6g ref=%.6g\n",
                               name, mode, n, i, jit[i], r);
                    failures++;
                }
                if (sim && !close(jit[i], sim[i], 1e-5f, 1e-4f)) {
                    if (failures < 3)
                        printf("    [%s n=%d i=%d] PARITY jit=%.6g simd=%.6g\n",
                               name, n, i, jit[i], sim[i]);
                    failures++;
                }
            }
            free(a); free(b); free(sim);
            tensor_free(ta); tensor_free(tb); tensor_free(tc);
        }
    }
    if (failures == 0) { g_pass++; printf("  PASS: binary %-10s (3 modes x %d sizes)\n", name, N_SIZES); }
    else               { printf("  FAIL: binary %-10s (%d mismatches)\n", name, failures); }
}

/* ---- reductions ------------------------------------------------------- */
static void check_reduction(const char* name, unary_op_fn op,
                            int is_max, int is_mean) {
    g_total++;
    int failures = 0;
    for (int si = 0; si < N_SIZES; si++) {
        int n = SIZES[si];
        float* in = malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++) in[i] = randf(-5.0f, 5.0f);

        Tensor* x = tensor_from_data(in, (int[]){n}, 1, &cpu_f32);
        Tensor* y = op(x);
        tensor_ensure_executed(y);
        if (!y || !y->data) {
            printf("    [%s n=%d] NULL output (y=%p data=%p numel=%zu)\n",
                   name, n, (void*)y, (void*)(y ? y->data : NULL),
                   (size_t)(y ? y->numel : 0));
            failures++;
            if (x) tensor_free(x);
            if (y) tensor_free(y);
            free(in);
            continue;
        }
        float jit = ((float*)y->data)[0];

        double acc = is_max ? -1e30 : 0.0;
        for (int i = 0; i < n; i++) {
            if (is_max) { if (in[i] > acc) acc = in[i]; }
            else acc += in[i];
        }
        if (is_mean) acc /= n;
        float ref = (float)acc;
        /* larger tol for sum: summation order differs from scalar */
        if (!close(jit, ref, 1e-3f, 5e-3f)) {
            printf("    [%s n=%d] jit=%.6g ref=%.6g\n", name, n, jit, ref);
            failures++;
        }
        free(in);
        tensor_free(x); tensor_free(y);
    }
    if (failures == 0) { g_pass++; printf("  PASS: reduce %-10s (%d sizes)\n", name, N_SIZES); }
    else               { printf("  FAIL: reduce %-10s (%d mismatches)\n", name, failures); }
}

/* ---- scalar reference functions (mirror kernel semantics) ------------- */
static float r_neg(float x)  { return -x; }
static float r_abs(float x)  { return fabsf(x); }
static float r_relu(float x) { return x > 0.0f ? x : 0.0f; }
static float r_exp(float x)  { return expf(x); }
static float r_log(float x)  { return logf(x + 1e-8f); }        /* kernel adds eps */
static float r_sqrt(float x) { return sqrtf(fabsf(x)); }        /* kernel abs()es  */
static float r_rsqrt(float x){ return 1.0f / (sqrtf(fabsf(x)) + 1e-8f); }
static float r_recip(float x){ return 1.0f / (x + 1e-8f); }
static float r_sig(float x)  { return 1.0f / (1.0f + expf(-x)); }
static float r_tanh(float x) { return tanhf(x); }
static float r_sin(float x)  { return sinf(x); }
static float r_cos(float x)  { return cosf(x); }

static float rb_add(float a, float b) { return a + b; }
static float rb_sub(float a, float b) { return a - b; }
static float rb_mul(float a, float b) { return a * b; }
static float rb_div(float a, float b) { return a / (b + 1e-8f); } /* kernel adds eps */
static float rb_max(float a, float b) { return a > b ? a : b; }

/* reduce whole tensor -> scalar (dims=NULL / num_dims=0 means "all axes") */
static Tensor* red_sum(Tensor* x)  { ReduceParams rp = {NULL, 0, false}; return uop_sum(x, &rp); }
static Tensor* red_mean(Tensor* x) { ReduceParams rp = {NULL, 0, false}; return uop_mean(x, &rp); }
static Tensor* red_max(Tensor* x)  { ReduceParams rp = {NULL, 0, false}; return uop_max_reduce(x, &rp); }

int main(void) {
    printf("=== SIMD / shape-specialized JIT parity ===\n");

    const float A0 = 0.0f, R0 = 1e-4f;   /* exact-op tolerance (IEEE, no fast-math) */
    const float AT = 2e-4f, RT = 2e-3f;  /* transcendental tolerance                */

    printf("Unary:\n");
    check_unary("neg",   uop_neg,     r_neg,   simd_neg_f32,     -5, 5, A0, R0);
    check_unary("abs",   uop_abs,     r_abs,   simd_abs_f32,     -5, 5, A0, R0);
    check_unary("relu",  uop_relu,    r_relu,  NULL,             -5, 5, A0, R0);
    check_unary("exp",   uop_exp,     r_exp,   simd_exp_f32,     -4, 4, AT, RT);
    /* log uses a coarse fast approximation (~4% off libm in places).  The
     * libm bound below is loose to match the kernel's real accuracy; the tight
     * jit-vs-simd PARITY check inside check_unary is the actual gate here. */
    check_unary("log",   uop_log,     r_log,   simd_log_f32,   0.05f, 8, 0.1f, 5e-2f);
    check_unary("sqrt",  uop_sqrt,    r_sqrt,  simd_sqrt_f32,     0, 20, AT, RT);
    check_unary("rsqrt", uop_rsqrt,   r_rsqrt, simd_rsqrt_f32, 0.05f, 20, AT, 5e-3f);
    check_unary("recip", uop_recip,   r_recip, simd_recip_f32,  0.1f, 8, AT, 5e-3f);
    check_unary("sigmoid", uop_sigmoid, r_sig, simd_sigmoid_f32, -6, 6, AT, RT);
    check_unary("tanh",  uop_tanh,    r_tanh,  simd_tanh_f32,    -4, 4, AT, RT);
    check_unary("sin",   uop_sin,     r_sin,   simd_sin_f32,     -3, 3, AT, RT);
    check_unary("cos",   uop_cos,     r_cos,   simd_cos_f32,     -3, 3, AT, RT);

    printf("Binary:\n");
    check_binary("add", uop_add, rb_add, simd_add_f32, -5, 5, A0, R0);
    check_binary("sub", uop_sub, rb_sub, simd_sub_f32, -5, 5, A0, R0);
    check_binary("mul", uop_mul, rb_mul, simd_mul_f32, -5, 5, A0, R0);
    check_binary("max", uop_max, rb_max, simd_max_f32, -5, 5, A0, R0);
    check_binary("div", uop_div, rb_div, simd_div_f32, 0.5f, 5, AT, 5e-3f);

    printf("Reductions:\n");
    check_reduction("sum",  red_sum,  0, 0);
    check_reduction("mean", red_mean, 0, 1);
    check_reduction("max",  red_max,  1, 0);

    printf("\nResults: %d/%d checks passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
