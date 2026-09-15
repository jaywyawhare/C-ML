/* Double-backward coverage (tensor_backward create_graph=true).
 *
 * Under the graph engine the published gradients are lazy UOP nodes, so a
 * second backward can differentiate the gradient itself. A silent break here
 * would be invisible to first-order suites: every check below differentiates
 * the *gradient of the loss*, not the loss, and each analytic value is
 * cross-checked against a central finite difference of the full two-level
 * procedure (forward -> backward -> penalty on the grad -> backward again),
 * so the test cannot drift with the implementation.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "ops/ir/autodiff.h"
#include "ops/uops.h"
#include "test_harness.h"

static TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};

static void fill_tensor(Tensor* t, const float* xs, size_t n) {
    memcpy(tensor_data_ptr(t), xs, sizeof(float) * n);
}

/* ── quad: f(x) = sum(x*x) ─────────────────────────────────────────────── */
/* dpenalty/dx where penalty(x) = sum((df/dx)^2) = 4*sum(x^2) → exactly 8x. */
static int test_quad_double_grad_exact(void) {
    const int n = 8;
    float xs[8];
    for (int i = 0; i < n; i++) xs[i] = 0.3f + 0.11f * i;

    cml_reset_ir_context();
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    fill_tensor(x, xs, n);
    x->requires_grad = true;

    Tensor* y = uop_mul(x, x);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, true);

    if (!x->grad) { cml_reset_ir_context(); return 0; }
    if (!x->grad->requires_grad) {
        printf("\n    grad not differentiable after create_graph");
        cml_reset_ir_context();
        return 0;
    }

    Tensor* g  = x->grad;
    Tensor* gg = uop_mul(g, g);
    Tensor* pen = uop_sum(gg, &rp);
    if (!pen->requires_grad) {
        printf("\n    requires_grad did not propagate into penalty");
        cml_reset_ir_context();
        return 0;
    }
    tensor_backward(pen, NULL, false, false);

    if (!x->grad || x->grad == g) {
        printf("\n    second backward published no new grad");
        cml_reset_ir_context();
        return 0;
    }
    tensor_ensure_executed(x->grad);
    float* got = (float*)tensor_data_ptr(x->grad);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        float want = 8.0f * xs[i];
        if (fabsf(got[i] - want) > 1e-4f * fmaxf(1.0f, fabsf(want))) {
            printf("\n    i=%d got %.5f want %.5f", i, got[i], want);
            ok = 0;
            break;
        }
    }
    cml_reset_ir_context();
    return ok;
}

/* Penalty value under the full procedure, for finite differences. */
static float quad_penalty(const float* xs, int n) {
    cml_reset_ir_context();
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    if (!x) return NAN;
    fill_tensor(x, xs, n);
    x->requires_grad = true;

    Tensor* y = uop_mul(x, x);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, true);
    if (!x->grad) { cml_reset_ir_context(); return NAN; }

    Tensor* gg  = uop_mul(x->grad, x->grad);
    Tensor* pen = uop_sum(gg, &rp);
    if (!pen) { cml_reset_ir_context(); return NAN; }
    tensor_ensure_executed(pen);
    float v = ((float*)tensor_data_ptr(pen))[0];
    cml_reset_ir_context();
    return v;
}

static int test_quad_double_grad_finite_diff(void) {
    const int n = 6;
    float xs[6];
    for (int i = 0; i < n; i++) xs[i] = 0.45f + 0.09f * i;

    cml_reset_ir_context();
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    fill_tensor(x, xs, n);
    x->requires_grad = true;

    Tensor* y = uop_mul(x, x);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, true);
    Tensor* gg  = uop_mul(x->grad, x->grad);
    Tensor* pen = uop_sum(gg, &rp);
    tensor_backward(pen, NULL, false, false);
    tensor_ensure_executed(x->grad);
    float ana[6];
    memcpy(ana, (float*)tensor_data_ptr(x->grad), sizeof(float) * n);
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < n; i++) {
        float saved = xs[i];
        xs[i] = saved + eps; float fp = quad_penalty(xs, n);
        xs[i] = saved - eps; float fm = quad_penalty(xs, n);
        xs[i] = saved;
        if (isnan(fp) || isnan(fm)) return 0;
        float num = (fp - fm) / (2.0f * eps);
        float den = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana[i])));
        if (fabsf(num - ana[i]) / den > 0.02f) {
            printf("    MISMATCH i=%d num=%.4f ana=%.4f", i, num, ana[i]);
            return 0;
        }
    }
    return 1;
}

/* ── matmul: h = x@w, loss = sum(h*h) ──────────────────────────────────── */
/* Exercises double differentiation through the MATMUL/TRANSPOSE/ADD VJPs. */
static float matmul_penalty(const float* xs, const float* ws,
                            int m, int k, int p) {
    cml_reset_ir_context();
    int xsh[2] = {m, k}, wsh[2] = {k, p};
    Tensor* x = tensor_zeros(xsh, 2, &cfg);
    Tensor* w = tensor_zeros(wsh, 2, &cfg);
    if (!x || !w) { cml_reset_ir_context(); return NAN; }
    fill_tensor(x, xs, (size_t)m * k);
    fill_tensor(w, ws, (size_t)k * p);
    x->requires_grad = true;
    w->requires_grad = true;

    Tensor* h    = uop_matmul(x, w);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(uop_mul(h, h), &rp);
    tensor_backward(loss, NULL, false, true);
    if (!x->grad) { cml_reset_ir_context(); return NAN; }

    Tensor* gg  = uop_mul(x->grad, x->grad);
    Tensor* pen = uop_sum(gg, &rp);
    if (!pen) { cml_reset_ir_context(); return NAN; }
    tensor_ensure_executed(pen);
    float v = ((float*)tensor_data_ptr(pen))[0];
    cml_reset_ir_context();
    return v;
}

static int test_matmul_double_grad_finite_diff(void) {
    const int m = 2, k = 3, p = 2;
    float xs[6], ws[6];
    for (int i = 0; i < m * k; i++) xs[i] = 0.2f + 0.13f * i;
    for (int i = 0; i < k * p; i++) ws[i] = 0.35f + 0.07f * i;

    cml_reset_ir_context();
    int xsh[2] = {m, k}, wsh[2] = {k, p};
    Tensor* x = tensor_zeros(xsh, 2, &cfg);
    Tensor* w = tensor_zeros(wsh, 2, &cfg);
    fill_tensor(x, xs, 6);
    fill_tensor(w, ws, 6);
    x->requires_grad = true;
    w->requires_grad = true;

    Tensor* h    = uop_matmul(x, w);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(uop_mul(h, h), &rp);
    tensor_backward(loss, NULL, false, true);

    Tensor* gg  = uop_mul(x->grad, x->grad);
    Tensor* pen = uop_sum(gg, &rp);
    tensor_backward(pen, NULL, false, false);
    tensor_ensure_executed(x->grad);
    float ana[6];
    memcpy(ana, (float*)tensor_data_ptr(x->grad), sizeof(float) * 6);
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < m * k; i++) {
        float saved = xs[i];
        xs[i] = saved + eps; float fp = matmul_penalty(xs, ws, m, k, p);
        xs[i] = saved - eps; float fm = matmul_penalty(xs, ws, m, k, p);
        xs[i] = saved;
        if (isnan(fp) || isnan(fm)) return 0;
        float num = (fp - fm) / (2.0f * eps);
        float den = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana[i])));
        if (fabsf(num - ana[i]) / den > 0.03f) {
            printf("    MISMATCH i=%d num=%.4f ana=%.4f", i, num, ana[i]);
            return 0;
        }
    }
    return 1;
}

/* Double-differentiate a composite VJP: quick-gelu's derivative contains a
 * sigmoid, and its VJP chain contains SUB/MUL composites. Under strict mode
 * any composite without a direct VJP rule would fail the second pass, so
 * this also proves the backward-subgraph composite rules are complete for
 * this chain. */
static float qgelu_penalty(const float* xs, int n) {
    cml_reset_ir_context();
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    if (!x) return NAN;
    fill_tensor(x, xs, n);
    x->requires_grad = true;

    Tensor* y = uop_quick_gelu(x);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, true);
    if (!x->grad) { cml_reset_ir_context(); return NAN; }

    Tensor* gg  = uop_mul(x->grad, x->grad);
    Tensor* pen = uop_sum(gg, &rp);
    if (!pen) { cml_reset_ir_context(); return NAN; }
    tensor_ensure_executed(pen);
    float v = ((float*)tensor_data_ptr(pen))[0];
    cml_reset_ir_context();
    return v;
}

static int test_composite_vjp_double_grad(void) {
    const int n = 6;
    float xs[6];
    for (int i = 0; i < n; i++) xs[i] = -0.8f + 0.3f * i;

    cml_reset_ir_context();
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    fill_tensor(x, xs, n);
    x->requires_grad = true;

    Tensor* y = uop_quick_gelu(x);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, true);

    if (!x->grad || !x->grad->requires_grad) {
        printf("\n    grad missing or inert");
        cml_reset_ir_context();
        return 0;
    }
    Tensor* gg  = uop_mul(x->grad, x->grad);
    Tensor* pen = uop_sum(gg, &rp);
    tensor_backward(pen, NULL, false, false);
    if (!x->grad) {
        printf("\n    second backward published no grad");
        cml_reset_ir_context();
        return 0;
    }
    tensor_ensure_executed(x->grad);
    float ana[6];
    memcpy(ana, (float*)tensor_data_ptr(x->grad), sizeof(float) * n);
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < n; i++) {
        float saved = xs[i];
        xs[i] = saved + eps; float fp = qgelu_penalty(xs, n);
        xs[i] = saved - eps; float fm = qgelu_penalty(xs, n);
        xs[i] = saved;
        if (isnan(fp) || isnan(fm)) return 0;
        float num = (fp - fm) / (2.0f * eps);
        float den = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(ana[i])));
        if (fabsf(num - ana[i]) / den > 0.03f) {
            printf("    MISMATCH i=%d num=%.4f ana=%.4f", i, num, ana[i]);
            return 0;
        }
    }
    return 1;
}

/* Without create_graph the published grad stays a plain result: it must NOT
 * be marked differentiable (a second backward over it would silently track
 * nothing, which is exactly the ambiguity this flag exists to control). */
static int test_no_create_graph_grad_inert(void) {
    const int n = 4;
    float xs[4] = {0.5f, 0.7f, 0.9f, 1.1f};

    cml_reset_ir_context();
    int sh[1] = {n};
    Tensor* x = tensor_zeros(sh, 1, &cfg);
    fill_tensor(x, xs, n);
    x->requires_grad = true;

    Tensor* y = uop_mul(x, x);
    ReduceParams rp = {0};
    Tensor* loss = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, false);

    int ok = x->grad != NULL && !x->grad->requires_grad;
    if (!ok) printf("    grad missing or unexpectedly differentiable");
    cml_reset_ir_context();
    return ok;
}

int main(void) {
    printf("double-backward (create_graph):\n");
    TEST(quad_double_grad_exact);
    TEST(quad_double_grad_finite_diff);
    TEST(matmul_double_grad_finite_diff);
    TEST(composite_vjp_double_grad);
    TEST(no_create_graph_grad_inert);
    return TEST_SUMMARY();
}
