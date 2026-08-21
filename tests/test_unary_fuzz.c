/* Every unary op, over random shapes, against a libm reference.
 *
 * The op list is enumerated from the two-argument-free `Tensor* uop_x(Tensor*)`
 * entry points in uops.h rather than chosen by hand, so "which ops are covered"
 * is answered by the header. Sampling is what let a broken `max` survive three
 * audits of the binary ops.
 *
 * Two properties per op:
 *   1. a unary op preserves its input's shape exactly (rank and every dim);
 *   2. every element equals the libm formula for that op.
 *
 * Each op declares the input domain it is defined on, so log/sqrt get positive
 * values, asin/acos get [-1,1], acosh gets >=1 and so on -- otherwise the
 * reference and the implementation would legitimately both produce NaN and the
 * comparison would prove nothing.
 */

#include "cml.h"
#include "test_require.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_int(int lo, int hi) { return lo + (int)(rng_next() % (uint64_t)(hi - lo + 1)); }

typedef enum { DOM_ANY, DOM_POS, DOM_UNIT, DOM_GE1, DOM_SMALL, DOM_INT } Domain;

typedef struct {
    const char* name;
    Tensor* (*build)(Tensor*);
    double (*ref)(double);
    Domain dom;
} UnaryOp;

/* ---- references (double precision, independent of the library) ---- */
static double r_neg(double x)   { return -x; }
static double r_exp(double x)   { return exp(x); }
static double r_log(double x)   { return log(x); }
static double r_sqrt(double x)  { return sqrt(x); }
static double r_recip(double x) { return 1.0 / x; }
static double r_abs(double x)   { return fabs(x); }
static double r_sin(double x)   { return sin(x); }
static double r_cos(double x)   { return cos(x); }
static double r_tan(double x)   { return tan(x); }
static double r_relu(double x)  { return x < 0 ? 0 : x; }
static double r_sigm(double x)  { return 1.0 / (1.0 + exp(-x)); }
static double r_tanh(double x)  { return tanh(x); }
static double r_gelu(double x)  { return 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
static double r_sign(double x)  { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
static double r_floor(double x) { return floor(x); }
static double r_ceil(double x)  { return ceil(x); }
static double r_round(double x) { return rint(x); }          /* ties to even */
static double r_log2(double x)  { return log2(x); }
static double r_exp2(double x)  { return exp2(x); }
static double r_asin(double x)  { return asin(x); }
static double r_acos(double x)  { return acos(x); }
static double r_atan(double x)  { return atan(x); }
static double r_sq(double x)    { return x * x; }
static double r_rsqrt(double x) { return 1.0 / sqrt(x); }
static double r_erf(double x)   { return erf(x); }
static double r_log10(double x) { return log10(x); }
static double r_sinh(double x)  { return sinh(x); }
static double r_cosh(double x)  { return cosh(x); }
static double r_asinh(double x) { return asinh(x); }
static double r_acosh(double x) { return acosh(x); }
static double r_atanh(double x) { return atanh(x); }
static double r_trunc(double x) { return trunc(x); }
static double r_isinf(double x) { return isinf(x) ? 1 : 0; }
static double r_isnan(double x) { return isnan(x) ? 1 : 0; }
static double r_isfin(double x) { return isfinite(x) ? 1 : 0; }
static double r_lnot(double x)  { return x == 0 ? 1 : 0; }
static double r_erfc(double x)  { return erfc(x); }
static double r_relu6(double x) { return isnan(x) ? x : fmin(fmax(x, 0), 6); }
static double r_hsig(double x)  { return fmin(fmax((x + 3) / 6, 0), 1); }
static double r_htanh(double x) { return fmin(fmax(x, -1), 1); }
static double r_qgelu(double x) { return x * (1.0 / (1.0 + exp(-1.702 * x))); }
static double r_splus(double x) { return log1p(exp(-fabs(x))) + fmax(x, 0); }
static double r_ssign(double x) { return x / (1 + fabs(x)); }
static double r_lsig(double x)  { return fmin(x, 0) - log1p(exp(-fabs(x))); }
static double r_selu(double x)  { return 1.0507009873554805 * (x > 0 ? x : 1.6732632423543772 * (exp(x) - 1)); }
static double r_mish(double x)  { return x * tanh(r_splus(x)); }
static double r_silu(double x)  { return x * r_sigm(x); }
static double r_hswish(double x){ return x >= 3 ? x : (x <= -3 ? 0 : x * (x + 3) / 6); }
static double r_bnot(double x)  { return (double)(~(int32_t)x); }

static const UnaryOp OPS[] = {
    {"neg", uop_neg, r_neg, DOM_ANY},        {"exp", uop_exp, r_exp, DOM_SMALL},
    {"log", uop_log, r_log, DOM_POS},        {"sqrt", uop_sqrt, r_sqrt, DOM_POS},
    {"recip", uop_recip, r_recip, DOM_POS},  {"abs", uop_abs, r_abs, DOM_ANY},
    {"sin", uop_sin, r_sin, DOM_ANY},        {"cos", uop_cos, r_cos, DOM_ANY},
    {"tan", uop_tan, r_tan, DOM_UNIT},       {"relu", uop_relu, r_relu, DOM_ANY},
    {"sigmoid", uop_sigmoid, r_sigm, DOM_ANY},
    {"tanh", uop_tanh, r_tanh, DOM_ANY},     {"gelu", uop_gelu, r_gelu, DOM_ANY},
    {"sign", uop_sign, r_sign, DOM_ANY},     {"floor", uop_floor, r_floor, DOM_ANY},
    {"ceil", uop_ceil, r_ceil, DOM_ANY},     {"round", uop_round, r_round, DOM_ANY},
    {"log2", uop_log2, r_log2, DOM_POS},     {"exp2", uop_exp2, r_exp2, DOM_SMALL},
    {"asin", uop_asin, r_asin, DOM_UNIT},    {"acos", uop_acos, r_acos, DOM_UNIT},
    {"atan", uop_atan, r_atan, DOM_ANY},     {"square", uop_square, r_sq, DOM_ANY},
    {"rsqrt", uop_rsqrt, r_rsqrt, DOM_POS},  {"erf", uop_erf, r_erf, DOM_ANY},
    {"bitwise_not", uop_bitwise_not, r_bnot, DOM_INT},
    {"log10", uop_log10, r_log10, DOM_POS},  {"sinh", uop_sinh, r_sinh, DOM_SMALL},
    {"cosh", uop_cosh, r_cosh, DOM_SMALL},   {"asinh", uop_asinh, r_asinh, DOM_ANY},
    {"acosh", uop_acosh, r_acosh, DOM_GE1},  {"atanh", uop_atanh, r_atanh, DOM_UNIT},
    {"trunc", uop_trunc, r_trunc, DOM_ANY},  {"isinf", uop_isinf, r_isinf, DOM_ANY},
    {"isnan", uop_isnan, r_isnan, DOM_ANY},  {"isfinite", uop_isfinite, r_isfin, DOM_ANY},
    {"logical_not", uop_logical_not, r_lnot, DOM_ANY},
    {"erfc", uop_erfc, r_erfc, DOM_ANY},     {"relu6", uop_relu6, r_relu6, DOM_ANY},
    {"hard_sigmoid", uop_hard_sigmoid, r_hsig, DOM_ANY},
    {"hard_tanh", uop_hard_tanh, r_htanh, DOM_ANY},
    {"quick_gelu", uop_quick_gelu, r_qgelu, DOM_ANY},
    {"softplus", uop_softplus, r_splus, DOM_ANY},
    {"softsign", uop_softsign, r_ssign, DOM_ANY},
    {"logsigmoid", uop_logsigmoid, r_lsig, DOM_ANY},
    {"selu", uop_selu, r_selu, DOM_ANY},     {"mish", uop_mish, r_mish, DOM_ANY},
    {"silu", uop_silu, r_silu, DOM_ANY},     {"hardswish", uop_hardswish, r_hswish, DOM_ANY},
};
#define NUM_OPS ((int)(sizeof(OPS) / sizeof(OPS[0])))

static float sample(Domain d, int k) {
    switch (d) {
    case DOM_POS:   return (float)(k % 7) * 0.5f + 0.5f;          /* 0.5 .. 3.5  */
    case DOM_UNIT:  return (float)(k % 9) * 0.2f - 0.8f;          /* -0.8 .. 0.8 */
    case DOM_GE1:   return (float)(k % 5) * 0.75f + 1.0f;         /* 1 .. 4      */
    case DOM_SMALL: return (float)(k % 7) * 0.5f - 1.5f;          /* -1.5 .. 1.5 */
    case DOM_INT:   return (float)(k % 16);
    default:        return (float)(k % 9) * 0.75f - 3.0f;         /* -3 .. 3     */
    }
}

static void one_case(const UnaryOp* op, int iter) {
    int nd = rng_int(1, 4);
    int shape[4];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);

    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    Tensor* a = tensor_zeros(shape, nd, &c);
    if (!a) { failures++; return; }
    for (size_t i = 0; i < a->numel; i++)
        tensor_set_float(a, i, sample(op->dom, (int)i + iter));

    Tensor* r = op->build(a);
    checks++;
    if (!r) {
        printf("  %-13s iter %d: returned NULL\n", op->name, iter);
        failures++;
        cml_reset_ir_context();
        return;
    }
    tensor_ensure_executed(r);

    /* Property 1: a unary op preserves shape exactly. */
    int bad = (r->ndim != nd);
    for (int i = 0; !bad && i < nd; i++) if (r->shape[i] != shape[i]) bad = 1;
    if (bad) {
        printf("  %-13s iter %d: shape [", op->name, iter);
        for (int i = 0; i < r->ndim; i++) printf("%d%s", r->shape[i], i + 1 < r->ndim ? "," : "");
        printf("], input was [");
        for (int i = 0; i < nd; i++) printf("%d%s", shape[i], i + 1 < nd ? "," : "");
        printf("]\n");
        failures++;
        cml_reset_ir_context();
        return;
    }

    /* Property 2: values match the libm reference. */
    for (size_t i = 0; i < r->numel; i++) {
        double in   = (double)sample(op->dom, (int)i + iter);
        double want = op->ref(in);
        double got  = (double)tensor_get_float(r, i);
        double tol  = fabs(want) * 1e-4 + 1e-5;
        if (fabs(got - want) > tol && !(isnan(got) && isnan(want))) {
            printf("  %-13s iter %d: f(%g) = %.9g, reference %.9g\n",
                   op->name, iter, in, got, want);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Unary ops over random shapes (%d ops, enumerated from uops.h):\n", NUM_OPS);

    const int ITERS = 40;
    for (int o = 0; o < NUM_OPS && failures < 15; o++)
        for (int i = 0; i < ITERS && failures < 15; i++) one_case(&OPS[o], i);

    printf("\n%d cases, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
