#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "autograd/amp.h"
#include "tensor/realize.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

static int test_amp_env_default(void);

/* The dtype CML_AMP_DTYPE asks for; the value is cached on first read inside
 * the library, so every suite in this binary sees the same answer. */
static DType amp_env_expected_dtype(void) {
    const char* env = getenv("CML_AMP_DTYPE");
    return (env && (strcmp(env, "bf16") == 0 || strcmp(env, "bfloat16") == 0))
               ? DTYPE_BFLOAT16 : DTYPE_FLOAT16;
}

/* Tiny linear layer y = w·x trained under bf16 autocast. The forward runs in
 * bf16 (cast-in), gradients come back through the bf16 graph, and the fp32
 * master weight is updated manually — the canonical AMP loop without an
 * optimizer, so the test isolates autocast/scaler behavior. */
static int test_bf16_training(void) {
    if (autocast_is_enabled()) return 0;
    autocast_set_dtype(DTYPE_BFLOAT16);
    if (autocast_get_dtype() != DTYPE_BFLOAT16) return 0;

    TensorConfig bcfg = {.dtype = DTYPE_BFLOAT16, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};
    int xs[2] = {8, 1}, ws[2] = {1, 1};

    float xd[8];
    for (int i = 0; i < 8; i++) xd[i] = (float)i / 4.0f - 1.0f;

    Tensor* Xb = tensor_full(xs, 2, &bcfg, 0.0f);
    Tensor* Wb = tensor_full(ws, 2, &bcfg, 0.3f);
    if (!Xb || !Wb) { if (Xb) tensor_free(Xb); if (Wb) tensor_free(Wb); return 0; }
    for (size_t i = 0; i < Xb->numel; i++) tensor_set_float(Xb, i, xd[i]);

    float master_w = 0.3f;
    cml_set_requires_grad(Wb, true);

    float first_loss = 0.0f, last_loss = INFINITY;
    bool grads_finite = true;

    for (int epoch = 0; epoch < 300 && grads_finite; epoch++) {
        /* target y = 4·x: exactly representable by the bias-free layer, and
         * same-dtype so the loss stays inside the bf16 subgraph and the
         * gradient flows back through the matmul VJP */
        Tensor* Yb   = tensor_full_like(Xb, 1.0f);
        for (size_t i = 0; i < Yb->numel; i++)
            tensor_set_float(Yb, i, 4.0f * xd[i]);

        Tensor* out  = cml_matmul(Xb, Wb);
        Tensor* loss = out ? cml_nn_mse_loss(out, Yb) : NULL;
        if (!loss) { grads_finite = false; tensor_free(Yb); break; }
        tensor_ensure_executed(loss);
        float l = tensor_get_float(loss, 0);
        if (epoch == 0) first_loss = l;
        last_loss = l;

        tensor_backward(loss, NULL, false, false);

        Tensor* g = tensor_get_grad(Wb);
        if (!g) grads_finite = false;
        else {
            tensor_ensure_executed(g);
            float gv = tensor_get_float(g, 0);
            if (!isfinite(gv)) grads_finite = false;
            master_w -= 0.05f * gv;   /* fp32 master update, written back */
            tensor_set_float(Wb, 0, master_w);
        }

        tensor_free(Yb);
        tensor_free(loss);
        tensor_free(out);
    }

    bool ok = grads_finite && last_loss < first_loss && last_loss < 1e-3f
              && isfinite(master_w);

    tensor_free(Xb);
    tensor_free(Wb);
    autocast_set_dtype(DTYPE_FLOAT16);
    return ok;
}

static int test_amp_dtype_api(void) {
    if (autocast_is_enabled()) return 0;

    /* factory default (fp16 unless CML_AMP_DTYPE selected bf16) */
    if (autocast_default_dtype() != amp_env_expected_dtype()) return 0;
    autocast_enter(DTYPE_FLOAT16);
    if (autocast_get_dtype() != DTYPE_FLOAT16) { autocast_exit(); return 0; }
    autocast_exit();

    autocast_set_dtype(DTYPE_BFLOAT16);
    if (autocast_get_dtype() != DTYPE_BFLOAT16) return 0;

    AutocastContext* ctx = autocast_get_context();
    if (!ctx || ctx->target_dtype != DTYPE_BFLOAT16) return 0;

    autocast_enter(DTYPE_FLOAT32);
    if (autocast_get_dtype() != DTYPE_FLOAT32) { autocast_exit(); return 0; }
    autocast_exit();

    autocast_set_dtype(DTYPE_FLOAT16);
    return autocast_get_dtype() == DTYPE_FLOAT16;
}

/* CML_AMP_DTYPE selects the default target dtype; it is cached on first read,
 * so this suite validates whichever variant it was launched with and CTest
 * registers both (plain, and with CML_AMP_DTYPE=bf16). */
static int test_amp_env_default(void) {
    return autocast_default_dtype() == amp_env_expected_dtype();
}

static int test_scaler_noop_under_bf16(void) {
    if (autocast_is_enabled()) return 0;
    autocast_set_dtype(DTYPE_BFLOAT16);

    GradScaler* scaler = grad_scaler_create(65536.0f, 2.0f, 0.5f, 2000);
    if (!scaler) return 0;
    if (scaler->scale_factor != 65536.0f) { grad_scaler_free(scaler); return 0; }

    int shape[1] = {4};
    TensorConfig fcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};
    Tensor* loss = tensor_full(shape, 1, &fcfg, 2.0f);
    Tensor* grad = tensor_full(shape, 1, &fcfg, 8.0f);
    if (!loss || !grad) {
        if (loss) tensor_free(loss);
        if (grad) tensor_free(grad);
        grad_scaler_free(scaler);
        return 0;
    }

    /* bf16 has fp32 exponent range, so gradients cannot underflow: scaling is
     * a no-op returning the loss itself, not a 65536x copy */
    Tensor* scaled = grad_scaler_scale(scaler, loss);
    bool noop = (scaled == loss);

    /* unscale divides by 1.0 under bf16 — values unchanged */
    Parameter param = {.tensor = loss, .requires_grad = true, .name = NULL};
    Parameter* params[1] = {&param};
    loss->grad = grad;
    grad_scaler_unscale(scaler, params, 1);
    for (int i = 0; i < 4; i++)
        if (tensor_get_float(grad, i) != 8.0f) noop = false;
    loss->grad = NULL;

    tensor_free(grad);
    if (scaled) tensor_free(scaled);
    tensor_free(loss);
    grad_scaler_free(scaler);

    /* under fp16 scaling must still happen */
    autocast_set_dtype(DTYPE_FLOAT16);
    GradScaler* s16 = grad_scaler_create(65536.0f, 2.0f, 0.5f, 2000);
    if (!s16) return 0;
    Tensor* loss16 = tensor_full(shape, 1, &fcfg, 2.0f);
    if (!loss16) { grad_scaler_free(s16); return 0; }
    Tensor* scaled16 = grad_scaler_scale(s16, loss16);
    bool scales_fp16 = scaled16 && scaled16 != loss16
                       && fabsf(tensor_get_float(scaled16, 0) - 131072.0f) < 1.0f;
    if (scaled16) tensor_free(scaled16);
    tensor_free(loss16);
    grad_scaler_free(s16);

    return noop && scales_fp16;
}

int main(void) {
    printf("test_amp_bf16\n\n");

    TEST(amp_dtype_api);
    TEST(amp_env_default);
    TEST(bf16_training);
    TEST(scaler_noop_under_bf16);

    cml_reset_ir_context();

    return TEST_SUMMARY();
}
