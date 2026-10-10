/**
 * Finite-difference gradient check for the fused RMSNorm (uop_rmsnorm + its VJP).
 * Compares the analytic gradients w.r.t. the input and the weight (from one
 * backward) against central-difference numerical gradients of the same MSE loss,
 * validating the fused forward kernel and its hand-derived backward.
 */
#include <math.h>
#include <stdio.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "ops/uops.h"

#define ROWS 2
#define D 5
#define EPS 1e-5f

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float X0[ROWS * D]  = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.0f, 0.4f};
static float W0[D]         = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f};
static float TGT[ROWS * D] = {0.1f, 0.2f, -0.3f, 0.0f, 0.15f, -0.2f, 0.1f, 0.3f, -0.1f, 0.05f};

static float loss_value(void) {
    int xs[] = {ROWS, D}, ws[] = {D}, ts[] = {ROWS, D};
    Tensor* x    = cml_tensor(X0, xs, 2, &CFG);
    Tensor* w    = cml_tensor(W0, ws, 1, &CFG);
    Tensor* y    = cml_tensor(TGT, ts, 2, &CFG);
    Tensor* out  = uop_rmsnorm(x, w, EPS);
    Tensor* loss = cml_nn_mse_loss(out, y);
    float v      = tensor_get_float(loss, 0);
    cml_reset_ir_context();
    return v;
}

/* Central-difference numerical gradient over buffer `buf` (n elements). */
static float max_grad_diff(float* buf, int n, const float* analytic) {
    const float eps = 1e-3f;
    float maxdiff   = 0.0f;
    for (int i = 0; i < n; i++) {
        float orig = buf[i];
        buf[i]     = orig + eps;
        float lp   = loss_value();
        buf[i]     = orig - eps;
        float lm   = loss_value();
        buf[i]     = orig;
        float num  = (lp - lm) / (2.0f * eps);
        float d    = fabsf(num - analytic[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    return maxdiff;
}

static int check(void) {
    float analytic_x[ROWS * D], analytic_w[D];
    {
        int xs[] = {ROWS, D}, ws[] = {D}, ts[] = {ROWS, D};
        Tensor* x        = cml_tensor(X0, xs, 2, &CFG);
        x->requires_grad = true;
        Tensor* w        = cml_tensor(W0, ws, 1, &CFG);
        w->requires_grad = true;
        Tensor* y        = cml_tensor(TGT, ts, 2, &CFG);
        Tensor* out      = uop_rmsnorm(x, w, EPS);
        Tensor* loss     = cml_nn_mse_loss(out, y);
        cml_backward(loss, NULL, false, false);
        const float* gx = (const float*)tensor_data_ptr(x->grad);
        const float* gw = (const float*)tensor_data_ptr(w->grad);
        for (int i = 0; i < ROWS * D; i++)
            analytic_x[i] = gx ? gx[i] : 0.0f;
        for (int j = 0; j < D; j++)
            analytic_w[j] = gw ? gw[j] : 0.0f;
    }
    cml_reset_ir_context();

    float dx = max_grad_diff(X0, ROWS * D, analytic_x);
    float dw = max_grad_diff(W0, D, analytic_w);
    int pass = (dx < 1e-2f) && (dw < 1e-2f);
    printf("  input grad maxdiff=%.3e  weight grad maxdiff=%.3e  %s\n", (double)dx, (double)dw,
           pass ? "PASS" : "FAIL");
    return pass;
}

int main(void) {
    cml_init();
    printf("Fused RMSNorm gradient check\n");
    int ok = check();
    printf(ok ? "Fused RMSNorm gradient check passed.\n" : "FAILED.\n");
    return ok ? 0 : 1;
}
