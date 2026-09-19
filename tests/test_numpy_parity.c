/*
 * Numerical fidelity: ops must match numpy/torch IEEE semantics, not the old
 * "safe" epsilon hacks (x/(y+1e-8), log(x+1e-8), sqrt(fabs(x)), 1/(x+1e-8)).
 *
 *   - exact division:  6/2 == 3.0 exactly (eps made it 2.9999999...)
 *   - div by zero   -> +/-inf   (not a/1e-8)
 *   - log(0)        -> -inf,  log(neg) -> nan
 *   - sqrt(neg)     -> nan       (no fabs)
 *   - reciprocal(0) -> inf
 */
#include <stdio.h>
#include <math.h>

#include "tensor/tensor.h"
#include "test_harness.h"
#include "ops/uops.h"

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
    return ok;
}
static const TensorConfig f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float ev1(Tensor* (*op)(Tensor*), float x) {
    Tensor* t = tensor_from_data((float[]){x}, (int[]){1}, 1, &f32);
    Tensor* r = op(t);
    tensor_ensure_executed(r);
    float v = ((float*)r->data)[0];
    tensor_free(t);
    tensor_free(r);
    return v;
}
static float ev2(Tensor* (*op)(Tensor*, Tensor*), float a, float b) {
    Tensor* ta = tensor_from_data((float[]){a}, (int[]){1}, 1, &f32);
    Tensor* tb = tensor_from_data((float[]){b}, (int[]){1}, 1, &f32);
    Tensor* r  = op(ta, tb);
    tensor_ensure_executed(r);
    float v = tensor_get_float(r, 0); /* dtype-aware: comparisons return bool */
    tensor_free(ta);
    tensor_free(tb);
    tensor_free(r);
    return v;
}

int main(void) {
    printf("=== numpy/torch numerical parity (IEEE, no eps) ===\n");

    /* exact division: no epsilon offset */
    check("div_exact_6_2", ev2(uop_div, 6.0f, 2.0f) == 3.0f);
    check("div_exact_1_4", ev2(uop_div, 1.0f, 4.0f) == 0.25f);
    check("div_by_zero_inf", isinf(ev2(uop_div, 1.0f, 0.0f)));
    check("div_neg_zero", ev2(uop_div, -2.0f, 0.0f) < 0 && isinf(ev2(uop_div, -2.0f, 0.0f)));

    /* log: IEEE domain */
    check("log_exact_e", fabsf(ev1(uop_log, expf(1.0f)) - 1.0f) < 1e-5f);
    check("log_zero_neginf", isinf(ev1(uop_log, 0.0f)) && ev1(uop_log, 0.0f) < 0);
    check("log_neg_nan", isnan(ev1(uop_log, -1.0f)));

    /* sqrt: sqrt(neg) is nan (no fabs) */
    check("sqrt_exact_4", ev1(uop_sqrt, 4.0f) == 2.0f);
    check("sqrt_neg_nan", isnan(ev1(uop_sqrt, -1.0f)));

    /* reciprocal(0) -> inf */
    check("recip_exact_4", ev1(uop_recip, 4.0f) == 0.25f);
    check("recip_zero_inf", isinf(ev1(uop_recip, 0.0f)));

    /* inf/nan propagation (IEEE) */
    float inf = INFINITY, nan = NAN;
    check("inf_plus_one", isinf(ev2(uop_add, inf, 1.0f)));
    check("inf_minus_inf", isnan(ev2(uop_sub, inf, inf)));
    check("inf_times_zero", isnan(ev2(uop_mul, inf, 0.0f)));
    check("nan_add", isnan(ev2(uop_add, nan, 1.0f)));
    check("nan_mul", isnan(ev2(uop_mul, nan, 2.0f)));

    /* comparisons with nan (IEEE: all false except !=) — results are bool 0/1 */
    check("nan_lt_false", ev2(uop_cmplt, nan, 1.0f) == 0.0f);
    check("nan_eq_false", ev2(uop_cmpeq, nan, nan) == 0.0f);
    check("nan_ne_true", ev2(uop_cmpne, nan, nan) == 1.0f);

    return TEST_SUMMARY();
}
