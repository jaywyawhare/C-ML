/**
 * Finite-difference gradient check for the fused LayerNorm (uop_layernorm + its
 * VJP). Compares the analytic gradients w.r.t. the input and the affine weight
 * (from one backward) against central-difference numerical gradients of the same
 * MSE loss, validating the fused forward kernel and its hand-derived backward.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "nn/layers/layernorm.h"

#define ROWS 2
#define D 5

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float X0[ROWS * D]  = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.0f, 0.4f};
static float TGT[ROWS * D] = {0.1f, 0.2f, -0.3f, 0.0f, 0.15f, -0.2f, 0.1f, 0.3f, -0.1f, 0.05f};

static float loss_value(LayerNorm* ln) {
    int xs[] = {ROWS, D}, ts[] = {ROWS, D};
    Tensor* x    = cml_tensor(X0, xs, 2, &CFG);
    Tensor* y    = cml_tensor(TGT, ts, 2, &CFG);
    Tensor* out  = module_forward((Module*)ln, x);
    Tensor* loss = cml_nn_mse_loss(out, y);
    float v      = tensor_get_float(loss, 0);
    cml_reset_ir_context();
    return v;
}

/* Central-difference numerical gradient over the buffer `buf` (n elements). */
static float max_grad_diff(LayerNorm* ln, float* buf, int n, const float* analytic) {
    const float eps = 1e-3f;
    float maxdiff   = 0.0f;
    for (int i = 0; i < n; i++) {
        float orig = buf[i];
        buf[i]     = orig + eps;
        float lp   = loss_value(ln);
        buf[i]     = orig - eps;
        float lm   = loss_value(ln);
        buf[i]     = orig;
        float num  = (lp - lm) / (2.0f * eps);
        float d    = fabsf(num - analytic[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    return maxdiff;
}

static int check(void) {
    LayerNorm* ln = cml_nn_layernorm(D, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    /* deterministic affine params */
    float* gw = (float*)ln->weight->tensor->data;
    float* gb = (float*)ln->bias->tensor->data;
    for (int j = 0; j < D; j++) {
        gw[j] = 1.0f + 0.1f * (float)(j - 2);
        gb[j] = 0.05f * (float)(j - 2);
    }

    /* Analytic: one backward; snapshot grads for x and weight. */
    float analytic_x[ROWS * D], analytic_w[D];
    tensor_zero_grad(ln->weight->tensor);
    {
        int xs[] = {ROWS, D}, ts[] = {ROWS, D};
        Tensor* x        = cml_tensor(X0, xs, 2, &CFG);
        x->requires_grad = true;
        Tensor* y        = cml_tensor(TGT, ts, 2, &CFG);
        Tensor* out      = module_forward((Module*)ln, x);
        Tensor* loss     = cml_nn_mse_loss(out, y);
        cml_backward(loss, NULL, false, false);
        const float* gx  = (const float*)tensor_data_ptr(x->grad);
        const float* gw2 = (const float*)tensor_data_ptr(ln->weight->tensor->grad);
        for (int i = 0; i < ROWS * D; i++)
            analytic_x[i] = gx ? gx[i] : 0.0f;
        for (int j = 0; j < D; j++)
            analytic_w[j] = gw2 ? gw2[j] : 0.0f;
    }
    cml_reset_ir_context();

    float dx = max_grad_diff(ln, X0, ROWS * D, analytic_x);
    float dw = max_grad_diff(ln, (float*)ln->weight->tensor->data, D, analytic_w);
    int pass = (dx < 1e-2f) && (dw < 1e-2f);
    printf("  input grad maxdiff=%.3e  weight grad maxdiff=%.3e  %s\n", (double)dx, (double)dw,
           pass ? "PASS" : "FAIL");
    return pass;
}

int main(void) {
    cml_init();
    printf("Fused LayerNorm gradient check\n");
    int ok = check();
    printf(ok ? "Fused LayerNorm gradient check passed.\n" : "FAILED.\n");
    return ok ? 0 : 1;
}
