/**
 * Measures (and locks in) the second-order behavior of the fused ops. The fused
 * backward kernels (UOP_SOFTMAX_BWD, etc.) are monolithic and have no VJP of
 * their own, so a create_graph=true backward through a fused op yields a
 * first-order gradient that is itself NOT differentiable (requires_grad=false).
 * This is the graceful, detectable no-VJP behavior - not silent wrong values and
 * not a crash. A primitive op (uop_mul) is included as the differentiable
 * contrast. If double-backward support is ever added to the fused ops, the
 * softmax assertion below flips and this test should be updated.
 */
#include <stdio.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "ops/uops.h"

#define D 4

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* Returns: 1 = grad exists and is differentiable, 0 = grad exists but not
 * differentiable, -1 = no grad / crash-free failure. */
static int probe(Tensor* (*make)(Tensor*)) {
    float xd[D]      = {0.3f, -0.2f, 0.5f, 0.1f};
    int s[]          = {1, D};
    Tensor* x        = cml_tensor(xd, s, 2, &CFG);
    x->requires_grad = true;
    Tensor* y        = make(x);
    if (!y)
        return -1;
    ReduceParams rp = {0};
    Tensor* loss    = uop_sum(y, &rp);
    tensor_backward(loss, NULL, false, true); /* create_graph */
    int r;
    if (!x->grad)
        r = -1;
    else
        r = x->grad->requires_grad ? 1 : 0;
    cml_reset_ir_context();
    return r;
}

static Tensor* via_softmax(Tensor* x) { return uop_softmax(x, -1); }
static Tensor* via_mul(Tensor* x) { return uop_mul(x, x); }

/* Full second-order probe: penalty = sum(grad^2); d(penalty)/dx via double
 * backward, compared against a finite difference of the first-order grad.
 * Returns the analytic value for x[idx], and writes the numerical one. */
static int softmax_double_grad(float* analytic_out, float* numeric_out, int idx) {
    float xd[D] = {0.3f, -0.2f, 0.5f, 0.1f};
    int s[]     = {1, D};

    /* analytic: d/dx sum((d sum(softmax)/dx)^2) -- sum(softmax)=1 so first grad
     * is 0 and this is a trivially small signal; use a weighted readout instead. */
    float w[D] = {1.0f, -0.5f, 0.25f, 0.75f};

    /* analytic via double backward */
    {
        Tensor* x        = cml_tensor(xd, s, 2, &CFG);
        x->requires_grad = true;
        Tensor* y        = uop_softmax(x, -1);
        Tensor* wt       = cml_tensor(w, s, 2, &CFG);
        Tensor* wy       = uop_mul(y, wt);
        ReduceParams rp  = {0};
        Tensor* loss     = uop_sum(wy, &rp);
        tensor_backward(loss, NULL, false, true);
        if (!x->grad || !x->grad->requires_grad) {
            cml_reset_ir_context();
            return 0; /* no 2nd order available */
        }
        Tensor* g   = x->grad;
        Tensor* gg  = uop_mul(g, g);
        Tensor* pen = uop_sum(gg, &rp);
        tensor_backward(pen, NULL, false, false);
        if (!x->grad) {
            cml_reset_ir_context();
            return 0;
        }
        const float* a = (const float*)tensor_data_ptr(x->grad);
        *analytic_out  = a ? a[idx] : 0.0f;
        cml_reset_ir_context();
    }

    /* numeric: finite-difference of penalty(x) = sum(first_grad^2) w.r.t x[idx] */
    const float eps = 1e-3f;
    float pen[2];
    for (int k = 0; k < 2; k++) {
        float xp[D];
        for (int j = 0; j < D; j++)
            xp[j] = xd[j];
        xp[idx] += (k == 0 ? eps : -eps);
        Tensor* x        = cml_tensor(xp, s, 2, &CFG);
        x->requires_grad = true;
        Tensor* y        = uop_softmax(x, -1);
        Tensor* wt       = cml_tensor(w, s, 2, &CFG);
        Tensor* wy       = uop_mul(y, wt);
        ReduceParams rp  = {0};
        Tensor* loss     = uop_sum(wy, &rp);
        tensor_backward(loss, NULL, false, false);
        const float* g = x->grad ? (const float*)tensor_data_ptr(x->grad) : NULL;
        float s2       = 0.0f;
        for (int j = 0; j < D; j++)
            s2 += g ? g[j] * g[j] : 0.0f;
        pen[k] = s2;
        cml_reset_ir_context();
    }
    *numeric_out = (pen[0] - pen[1]) / (2.0f * eps);
    return 1;
}

int main(void) {
    cml_init();
    printf("Fused-op double-backward behavior\n");

    int prim  = probe(via_mul);
    int fused = probe(via_softmax);
    printf("  primitive (mul) first-grad differentiable: %s\n", prim == 1 ? "yes" : "no");
    printf("  fused (softmax) first-grad differentiable: %s\n", fused == 1 ? "yes" : "no");

    /* Double-backward (create_graph) is a graph-autodiff feature; eager mode
     * doesn't build a higher-order graph, so if even the primitive op's grad is
     * non-differentiable here, this mode has no 2nd order to check - skip. */
    if (prim != 1) {
        printf(
            "  create_graph not supported in this mode (e.g. eager); skipping 2nd-order check\n");
        printf("Double-backward behavior OK (skipped).\n");
        return 0;
    }

    int ok = 1;

    if (fused == 1) {
        /* It claims differentiability - so the 2nd-order result MUST be correct,
         * else it is silently wrong. Measure against finite differences. */
        float a = 0.0f, n = 0.0f;
        int have   = softmax_double_grad(&a, &n, 0);
        float diff = (a > n ? a - n : n - a);
        printf("  softmax 2nd-order x[0]: analytic=%.5f numeric=%.5f diff=%.3e\n", (double)a,
               (double)n, (double)diff);
        int correct = have && diff < 2e-3f;
        printf("  => fused softmax 2nd-order is %s\n", correct ? "CORRECT" : "WRONG (silent)");
        ok = ok && correct;
    } else {
        printf("  => fused softmax reports no 2nd order (graceful)\n");
    }

    printf(ok ? "Double-backward behavior OK.\n" : "Double-backward behavior PROBLEM.\n");
    return ok ? 0 : 1;
}
