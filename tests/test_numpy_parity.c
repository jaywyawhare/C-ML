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
#include "ops/uops.h"

static int g_pass = 0, g_total = 0;
static int check(const char* name, int ok) {
    g_total++;
    if (ok) { g_pass++; printf("  PASS: %s\n", name); }
    else    { printf("  FAIL: %s\n", name); }
    return ok;
}
static const TensorConfig f32 = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                                 .has_dtype = true, .has_device = true};

static float ev1(Tensor* (*op)(Tensor*), float x) {
    Tensor* t = tensor_from_data((float[]){x}, (int[]){1}, 1, &f32);
    Tensor* r = op(t);
    tensor_ensure_executed(r);
    float v = ((float*)r->data)[0];
    tensor_free(t); tensor_free(r);
    return v;
}
static float ev2(Tensor* (*op)(Tensor*, Tensor*), float a, float b) {
    Tensor* ta = tensor_from_data((float[]){a}, (int[]){1}, 1, &f32);
    Tensor* tb = tensor_from_data((float[]){b}, (int[]){1}, 1, &f32);
    Tensor* r  = op(ta, tb);
    tensor_ensure_executed(r);
    float v = ((float*)r->data)[0];
    tensor_free(ta); tensor_free(tb); tensor_free(r);
    return v;
}

int main(void) {
    printf("=== numpy/torch numerical parity (IEEE, no eps) ===\n");

    /* exact division: no epsilon offset */
    check("div_exact_6_2",   ev2(uop_div, 6.0f, 2.0f) == 3.0f);
    check("div_exact_1_4",   ev2(uop_div, 1.0f, 4.0f) == 0.25f);
    check("div_by_zero_inf", isinf(ev2(uop_div, 1.0f, 0.0f)));
    check("div_neg_zero",    ev2(uop_div, -2.0f, 0.0f) < 0 && isinf(ev2(uop_div, -2.0f, 0.0f)));

    /* log: IEEE domain */
    check("log_exact_e",     fabsf(ev1(uop_log, expf(1.0f)) - 1.0f) < 1e-5f);
    check("log_zero_neginf", isinf(ev1(uop_log, 0.0f)) && ev1(uop_log, 0.0f) < 0);
    check("log_neg_nan",     isnan(ev1(uop_log, -1.0f)));

    /* sqrt: sqrt(neg) is nan (no fabs) */
    check("sqrt_exact_4",    ev1(uop_sqrt, 4.0f) == 2.0f);
    check("sqrt_neg_nan",    isnan(ev1(uop_sqrt, -1.0f)));

    /* reciprocal(0) -> inf */
    check("recip_exact_4",   ev1(uop_recip, 4.0f) == 0.25f);
    check("recip_zero_inf",  isinf(ev1(uop_recip, 0.0f)));

    printf("\nResults: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
