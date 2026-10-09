/**
 * Finite-difference gradient check for the fused last-dim softmax (UOP_SOFTMAX +
 * its VJP): analytic input gradient vs a central-difference numerical gradient
 * of the same MSE loss.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "ops/uops.h"

#define ROWS 2
#define D 5

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float X0[ROWS * D]  = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.0f, 0.4f};
static float TGT[ROWS * D] = {0.1f, 0.2f, 0.3f, 0.25f, 0.15f, 0.2f, 0.1f, 0.3f, 0.2f, 0.2f};

static float loss_value(void) {
    int s[]      = {ROWS, D};
    Tensor* x    = cml_tensor(X0, s, 2, &CFG);
    Tensor* y    = cml_tensor(TGT, s, 2, &CFG);
    Tensor* out  = uop_softmax(x, -1);
    Tensor* loss = cml_nn_mse_loss(out, y);
    float v      = tensor_get_float(loss, 0);
    cml_reset_ir_context();
    return v;
}

int main(void) {
    cml_init();
    printf("Fused softmax gradient check\n");

    float analytic[ROWS * D];
    {
        int s[]          = {ROWS, D};
        Tensor* x        = cml_tensor(X0, s, 2, &CFG);
        x->requires_grad = true;
        Tensor* y        = cml_tensor(TGT, s, 2, &CFG);
        Tensor* out      = uop_softmax(x, -1);
        Tensor* loss     = cml_nn_mse_loss(out, y);
        cml_backward(loss, NULL, false, false);
        const float* gx = (const float*)tensor_data_ptr(x->grad);
        for (int i = 0; i < ROWS * D; i++)
            analytic[i] = gx ? gx[i] : 0.0f;
    }
    cml_reset_ir_context();

    const float eps = 1e-3f;
    float maxdiff   = 0.0f;
    for (int i = 0; i < ROWS * D; i++) {
        float orig = X0[i];
        X0[i]      = orig + eps;
        float lp   = loss_value();
        X0[i]      = orig - eps;
        float lm   = loss_value();
        X0[i]      = orig;
        float num  = (lp - lm) / (2.0f * eps);
        float d    = fabsf(num - analytic[i]);
        if (d > maxdiff)
            maxdiff = d;
    }

    int pass = maxdiff < 1e-2f;
    printf("  input grad maxdiff=%.3e %s\n", (double)maxdiff, pass ? "PASS" : "FAIL");
    printf(pass ? "Fused softmax gradient check passed.\n" : "FAILED.\n");
    return pass ? 0 : 1;
}
