/*
 * Forward-mode AD (jvp / jacfwd) via the double-VJP identity. The forward map is
 * y = (x @ W)^2 (elementwise). jvp(v) and the full jacfwd Jacobian are both
 * cross-checked against central finite differences. The double-VJP needs the
 * graph autodiff engine, so the cases are skipped under the eager engine.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "ops/ir/autodiff.h"
#include "ops/uops.h"
#include "autograd/forward_mode.h"
#include "test_harness.h"

#define K 3
#define P 2

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static const float WD[K * P] = {0.5f, -0.3f, 0.2f, 0.8f, -0.4f, 0.1f};

static Tensor* mk(const float* data, int* shape, int ndim) {
    Tensor* t = tensor_zeros(shape, ndim, &cfg);
    memcpy(tensor_data_ptr(t), data, tensor_numel(shape, ndim) * sizeof(float));
    return t;
}

/* y = (x @ W)^2, elementwise. ctx is W. */
static Tensor* f_sq(Tensor* x, void* ctx) {
    Tensor* W = (Tensor*)ctx;
    Tensor* h = uop_matmul(x, W);
    return uop_mul(h, h);
}

/* Evaluate f at xd into out[P]. */
static void eval_f(const float* xd, float* out) {
    cml_reset_ir_context();
    int xsh[2] = {1, K}, wsh[2] = {K, P};
    Tensor* x = mk(xd, xsh, 2);
    Tensor* W = mk(WD, wsh, 2);
    Tensor* y = f_sq(x, W);
    tensor_ensure_executed(y);
    memcpy(out, tensor_data_ptr(y), P * sizeof(float));
    cml_reset_ir_context();
}

static int test_jvp_matches_finite_diff(void) {
    float xd[K] = {0.7f, -0.2f, 0.5f};
    float vd[K] = {0.3f, 1.0f, -0.5f};

    cml_reset_ir_context();
    int xsh[2] = {1, K}, wsh[2] = {K, P};
    Tensor* x  = mk(xd, xsh, 2);
    Tensor* W  = mk(WD, wsh, 2);
    Tensor* v  = mk(vd, xsh, 2);
    Tensor* jv = cml_jvp(f_sq, W, x, v);
    if (!jv) {
        cml_reset_ir_context();
        return 0;
    }
    float ana[P];
    memcpy(ana, tensor_data_ptr(jv), sizeof(ana));
    cml_reset_ir_context();

    const float eps = 1e-3f;
    float xp[K], xm[K], fp[P], fm[P];
    for (int i = 0; i < K; i++) {
        xp[i] = xd[i] + eps * vd[i];
        xm[i] = xd[i] - eps * vd[i];
    }
    eval_f(xp, fp);
    eval_f(xm, fm);
    for (int i = 0; i < P; i++) {
        float num = (fp[i] - fm[i]) / (2 * eps);
        if (fabsf(num - ana[i]) > 1e-2f * fmaxf(1.0f, fabsf(num)))
            return 0;
    }
    return 1;
}

static int test_jacfwd_matches_finite_diff(void) {
    float xd[K] = {0.4f, 0.9f, -0.6f};

    cml_reset_ir_context();
    int xsh[2] = {1, K}, wsh[2] = {K, P};
    Tensor* x   = mk(xd, xsh, 2);
    Tensor* W   = mk(WD, wsh, 2);
    Tensor* jac = cml_jacfwd(f_sq, W, x);
    if (!jac) {
        cml_reset_ir_context();
        return 0;
    }
    float ana[P * K];
    memcpy(ana, tensor_data_ptr(jac), sizeof(ana));
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int j = 0; j < K; j++) {
        float xp[K], xm[K], fp[P], fm[P];
        memcpy(xp, xd, sizeof(xp));
        memcpy(xm, xd, sizeof(xm));
        xp[j] += eps;
        xm[j] -= eps;
        eval_f(xp, fp);
        eval_f(xm, fm);
        for (int i = 0; i < P; i++) {
            float num = (fp[i] - fm[i]) / (2 * eps);
            if (fabsf(num - ana[i * K + j]) > 1e-2f * fmaxf(1.0f, fabsf(num)))
                return 0;
        }
    }
    return 1;
}

int main(void) {
    printf("test_forward_mode\n\n");
    if (!cml_autodiff_use_graph()) {
        printf("  (skipping jvp/jacfwd: double-VJP needs the graph engine)\n");
        return TEST_SUMMARY();
    }
    TEST(jvp_matches_finite_diff);
    TEST(jacfwd_matches_finite_diff);
    cml_reset_ir_context();
    return TEST_SUMMARY();
}
