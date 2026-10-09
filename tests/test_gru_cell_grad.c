/**
 * Finite-difference gradient check for the fused GRU cell (uop_gru_cell +
 * its VJP). Trains nothing: it compares the analytic weight gradient from one
 * backward against a central-difference numerical gradient of the same MSE
 * loss, which validates both the fused forward kernel and its hand-derived
 * backward end-to-end (through the LINEAR projections to the weights).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "nn/layers/rnn.h"

#define IN 3
#define HID 4

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static const float X0[IN]   = {0.3f, -0.2f, 0.5f};
static const float H0[HID]  = {0.1f, -0.4f, 0.2f, 0.05f};
static const float TGT[HID] = {0.2f, 0.1f, -0.3f, 0.15f};

static void set_weights(GRUCell* c) {
    float* wih = (float*)c->weight_ih->tensor->data;
    for (size_t i = 0; i < c->weight_ih->tensor->numel; i++)
        wih[i] = 0.05f * (float)((int)(i % 7) - 3);
    float* whh = (float*)c->weight_hh->tensor->data;
    for (size_t i = 0; i < c->weight_hh->tensor->numel; i++)
        whh[i] = 0.04f * (float)((int)(i % 5) - 2);
    if (c->bias_ih) {
        float* b = (float*)c->bias_ih->tensor->data;
        for (size_t i = 0; i < c->bias_ih->tensor->numel; i++)
            b[i] = 0.01f * (float)(i % 3);
    }
    if (c->bias_hh) {
        float* b = (float*)c->bias_hh->tensor->data;
        for (size_t i = 0; i < c->bias_hh->tensor->numel; i++)
            b[i] = -0.02f * (float)(i % 3);
    }
}

/* MSE(gru_cell(x, h), target) as a plain scalar; resets the IR graph after. */
static float loss_value(GRUCell* c) {
    int xs[] = {1, IN}, hs[] = {1, HID}, ts[] = {1, HID};
    Tensor* x    = cml_tensor((float*)X0, xs, 2, &CFG);
    Tensor* h    = cml_tensor((float*)H0, hs, 2, &CFG);
    Tensor* y    = cml_tensor((float*)TGT, ts, 2, &CFG);
    Tensor* out  = gru_cell_forward(c, x, h);
    Tensor* loss = cml_nn_mse_loss(out, y);
    float v      = tensor_get_float(loss, 0);
    cml_reset_ir_context();
    return v;
}

int main(void) {
    cml_init();
    printf("Fused GRU cell gradient check\n");

    GRUCell* c = cml_nn_gru_cell(IN, HID, true, DTYPE_FLOAT32, DEVICE_CPU);
    set_weights(c);

    /* Analytic: one backward, snapshot weight_ih grad. */
    Tensor* wih = c->weight_ih->tensor;
    tensor_zero_grad(wih);
    {
        int xs[] = {1, IN}, hs[] = {1, HID}, ts[] = {1, HID};
        Tensor* x    = cml_tensor((float*)X0, xs, 2, &CFG);
        Tensor* h    = cml_tensor((float*)H0, hs, 2, &CFG);
        Tensor* y    = cml_tensor((float*)TGT, ts, 2, &CFG);
        Tensor* out  = gru_cell_forward(c, x, h);
        Tensor* loss = cml_nn_mse_loss(out, y);
        cml_backward(loss, NULL, false, false);
    }
    int n = (int)wih->numel;
    float analytic[3 * HID * IN];
    const float* gd = (const float*)tensor_data_ptr(wih->grad);
    for (int i = 0; i < n; i++)
        analytic[i] = gd[i];
    cml_reset_ir_context();

    /* Numerical: central difference of the loss w.r.t. each weight_ih entry. */
    const float eps = 1e-3f;
    float* w        = (float*)wih->data;
    float maxdiff   = 0.0f;
    for (int i = 0; i < n; i++) {
        float orig = w[i];
        w[i]       = orig + eps;
        float lp   = loss_value(c);
        w[i]       = orig - eps;
        float lm   = loss_value(c);
        w[i]       = orig;
        float num  = (lp - lm) / (2.0f * eps);
        float d    = fabsf(num - analytic[i]);
        if (d > maxdiff)
            maxdiff = d;
    }

    int pass = maxdiff < 1e-2f; /* finite-difference tolerance */
    printf("  weight_ih grad maxdiff=%.3e %s\n", (double)maxdiff, pass ? "PASS" : "FAIL");
    printf(pass ? "Fused GRU cell gradient check passed.\n" : "FAILED.\n");
    return pass ? 0 : 1;
}
