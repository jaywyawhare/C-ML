/* Soft Dice loss: forward value checked by hand (1 - (2*1.7+1)/(4+1) = 0.12) and the analytic
 * autograd gradient checked against central finite differences. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "autograd/autograd.h"
#include "test_harness.h"

#define N 4
static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float P0[N]  = {0.8f, 0.2f, 0.9f, 0.1f};
static float TGT[N] = {1.0f, 0.0f, 1.0f, 0.0f};

static float loss_value(void) {
    int s[]      = {1, N};
    Tensor* p    = cml_tensor(P0, s, 2, &CFG);
    Tensor* y    = cml_tensor(TGT, s, 2, &CFG);
    Tensor* loss = cml_nn_dice_loss(p, y, 1.0f);
    float v      = tensor_get_float(loss, 0);
    cml_reset_ir_context();
    return v;
}

static int test_forward(void) { return fabsf(loss_value() - 0.12f) < 1e-5f; }

static int test_gradient(void) {
    float analytic[N];
    {
        int s[]          = {1, N};
        Tensor* p        = cml_tensor(P0, s, 2, &CFG);
        p->requires_grad = true;
        Tensor* y        = cml_tensor(TGT, s, 2, &CFG);
        Tensor* loss     = cml_nn_dice_loss(p, y, 1.0f);
        cml_backward(loss, NULL, false, false);
        const float* gp = (const float*)tensor_data_ptr(p->grad);
        for (int i = 0; i < N; i++)
            analytic[i] = gp ? gp[i] : 0.0f;
    }
    cml_reset_ir_context();

    const float eps = 1e-3f;
    int ok          = 1;
    for (int i = 0; i < N; i++) {
        float orig = P0[i];
        P0[i]      = orig + eps;
        float lp   = loss_value();
        P0[i]      = orig - eps;
        float lm   = loss_value();
        P0[i]      = orig;
        float num  = (lp - lm) / (2.0f * eps);
        ok &= fabsf(num - analytic[i]) < 1e-2f;
    }
    return ok;
}

int main(void) {
    cml_init();
    printf("=== soft Dice loss ===\n");
    TEST(forward);
    TEST(gradient);
    cml_cleanup();
    return TEST_SUMMARY();
}
