/*
 * weight_norm / spectral_norm parametrizations, validated against closed-form
 * oracles: hand-computed w = g*v/||v|| for weight_norm, a diagonal matrix with a
 * known largest singular value for spectral_norm, plus finite-difference checks
 * that gradients flow through both.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "nn/parametrize.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static Tensor* mk(const float* data, int* shape, int ndim, bool grad) {
    Tensor* t = tensor_zeros(shape, ndim, &cfg);
    memcpy(tensor_data_ptr(t), data, tensor_numel(shape, ndim) * sizeof(float));
    t->requires_grad = grad;
    return t;
}

static int test_weight_norm_forward(void) {
    float vd[4] = {3, 4, 0, 5};
    float gd[2] = {2, 10};
    int vsh[2]  = {2, 2};
    int gsh[1]  = {2};
    cml_reset_ir_context();
    Tensor* v = mk(vd, vsh, 2, false);
    Tensor* g = mk(gd, gsh, 1, false);
    Tensor* w = cml_nn_weight_norm(v, g, 0);
    if (!w || tensor_ensure_executed(w) != 0) {
        cml_reset_ir_context();
        return 0;
    }
    float* got    = (float*)tensor_data_ptr(w);
    float want[4] = {1.2f, 1.6f, 0.0f, 10.0f}; /* g*v/||v|| per row (||v||=5 both) */
    int ok        = 1;
    for (int i = 0; i < 4; i++)
        if (fabsf(got[i] - want[i]) > 1e-5f)
            ok = 0;
    cml_reset_ir_context();
    return ok;
}

/* sum(weight_norm(v,g)) as a scalar, for finite differencing one g element. */
static float wn_loss(const float* vd, const float* gd) {
    int vsh[2] = {2, 2};
    int gsh[1] = {2};
    cml_reset_ir_context();
    Tensor* v    = mk(vd, vsh, 2, false);
    Tensor* g    = mk(gd, gsh, 1, false);
    Tensor* w    = cml_nn_weight_norm(v, g, 0);
    Tensor* loss = cml_sum(w, -1, false);
    tensor_ensure_executed(loss);
    float r = ((float*)tensor_data_ptr(loss))[0];
    cml_reset_ir_context();
    return r;
}

static int test_weight_norm_grad(void) {
    float vd[4] = {3, 4, 0, 5};
    float gd[2] = {2, 10};
    int vsh[2]  = {2, 2};
    int gsh[1]  = {2};
    cml_reset_ir_context();
    Tensor* v    = mk(vd, vsh, 2, false);
    Tensor* g    = mk(gd, gsh, 1, true);
    Tensor* w    = cml_nn_weight_norm(v, g, 0);
    Tensor* loss = cml_sum(w, -1, false);
    tensor_backward(loss, NULL, false, false);
    if (!g->grad) {
        cml_reset_ir_context();
        return 0;
    }
    tensor_ensure_executed(g->grad);
    float ana[2];
    memcpy(ana, tensor_data_ptr(g->grad), sizeof(ana));
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < 2; i++) {
        float s   = gd[i];
        gd[i]     = s + eps;
        float fp  = wn_loss(vd, gd);
        gd[i]     = s - eps;
        float fm  = wn_loss(vd, gd);
        gd[i]     = s;
        float num = (fp - fm) / (2 * eps);
        if (fabsf(num - ana[i]) > 1e-2f * fmaxf(1.0f, fabsf(num)))
            return 0;
    }
    return 1;
}

static int test_spectral_norm_sigma(void) {
    /* diag(3,2,1): largest singular value is 3, so w = W/3 has spectral norm 1. */
    float wd[9] = {3, 0, 0, 0, 2, 0, 0, 0, 1};
    float ud[3] = {1, 1, 1};
    int wsh[2]  = {3, 3};
    int ush[1]  = {3};
    cml_reset_ir_context();
    Tensor* W = mk(wd, wsh, 2, false);
    Tensor* u = mk(ud, ush, 1, false);
    Tensor* w = cml_nn_spectral_norm(W, u, 30);
    if (!w || tensor_ensure_executed(w) != 0) {
        cml_reset_ir_context();
        return 0;
    }
    float* got = (float*)tensor_data_ptr(w);
    /* W/sigma with sigma=3: diagonal becomes 1, 2/3, 1/3. */
    int ok = fabsf(got[0] - 1.0f) < 1e-3f && fabsf(got[4] - 2.0f / 3.0f) < 1e-3f &&
             fabsf(got[8] - 1.0f / 3.0f) < 1e-3f;
    cml_reset_ir_context();
    return ok;
}

static float sn_loss(const float* wd, const float* ud) {
    int wsh[2] = {3, 3};
    int ush[1] = {3};
    cml_reset_ir_context();
    Tensor* W    = mk(wd, wsh, 2, false);
    Tensor* u    = mk(ud, ush, 1, false);
    Tensor* w    = cml_nn_spectral_norm(W, u, 30);
    Tensor* loss = cml_sum(w, -1, false);
    tensor_ensure_executed(loss);
    float r = ((float*)tensor_data_ptr(loss))[0];
    cml_reset_ir_context();
    return r;
}

static int test_spectral_norm_grad(void) {
    float wd[9] = {3, 0.2f, 0, 0.1f, 2, 0, 0, 0, 1};
    float ud[3] = {1, 0, 0};
    int wsh[2]  = {3, 3};
    int ush[1]  = {3};
    cml_reset_ir_context();
    Tensor* W    = mk(wd, wsh, 2, true);
    Tensor* u    = mk(ud, ush, 1, false);
    Tensor* w    = cml_nn_spectral_norm(W, u, 30);
    Tensor* loss = cml_sum(w, -1, false);
    tensor_backward(loss, NULL, false, false);
    if (!W->grad) {
        cml_reset_ir_context();
        return 0;
    }
    tensor_ensure_executed(W->grad);
    float ana[9];
    memcpy(ana, tensor_data_ptr(W->grad), sizeof(ana));
    cml_reset_ir_context();

    const float eps = 1e-3f;
    /* Check a few entries; u is converged so sigma's dependence on W is smooth. */
    int idxs[3] = {0, 4, 1};
    for (int k = 0; k < 3; k++) {
        int i     = idxs[k];
        float s   = wd[i];
        wd[i]     = s + eps;
        float fp  = sn_loss(wd, ud);
        wd[i]     = s - eps;
        float fm  = sn_loss(wd, ud);
        wd[i]     = s;
        float num = (fp - fm) / (2 * eps);
        if (fabsf(num - ana[i]) > 2e-2f * fmaxf(1.0f, fabsf(num)))
            return 0;
    }
    return 1;
}

int main(void) {
    printf("test_parametrize\n\n");
    TEST(weight_norm_forward);
    TEST(weight_norm_grad);
    TEST(spectral_norm_sigma);
    TEST(spectral_norm_grad);
    cml_reset_ir_context();
    return TEST_SUMMARY();
}
