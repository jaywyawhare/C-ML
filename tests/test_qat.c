/*
 * QAT primitives: min/max observers, fake-quant, and the straight-through
 * estimator. Modeled on tests/test_quant_matmul.c.
 *
 * Covers (a) observer convergence to a known range for both modes,
 * (b) the fake-quant error bound |fq(x) - x| <= scale/2,
 * (c) STE gradient identity via tensor_backward + finite differences of the
 *     surrogate (NOT the raw forward — fake-quant is piecewise constant, so a
 *     literal forward finite difference is ~0 almost everywhere and says
 *     nothing about the estimator), and (d) a tiny end-to-end regression in
 *     which a linear layer trained through weight fake-quant reduces its loss.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/context.h"
#include "core/quantization.h"
#include "autograd/autograd.h"
#include "autograd/loss_functions.h"
#include "test_harness.h"

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static uint32_t s_rng = 0xca7f00du;
static float randf(float lo, float hi) {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
    return lo + ((float)(s_rng & 0xffffff) / (float)0xffffff) * (hi - lo);
}

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) { tests_passed++; printf("  PASS: %s\n", name); }
    else    { printf("  FAIL: %s\n", name); }
    return ok;
}

/* ---- observers ---- */

/* Plain MINMAX mode must track the exact running extremes across batches,
 * ignore later smaller values, and clear on reset(). */
static int test_observer_minmax(void) {
    QatObserver* obs = cml_qat_observer_create(CML_QAT_OBS_MINMAX, 0.0f);
    if (!obs) return 0;

    int ok = 1;
    float batches[3][16];
    /* b0 spans [-1.5, 2.5]; b1/b2 stay strictly inside it */
    for (int j = 0; j < 16; j++) {
        batches[0][j] = randf(-1.4f, 2.4f);
        batches[1][j] = randf(-1.0f, 2.0f);
        batches[2][j] = randf(-0.5f, 1.0f);
    }
    batches[0][3]  = -1.5f;
    batches[0][10] =  2.5f;

    for (int b = 0; b < 3 && ok; b++) {
        Tensor* t = tensor_from_data(batches[b], (int[]){16}, 1, &cpu_f32);
        if (cml_qat_observer_update(obs, t) != 0) ok = 0;
        tensor_free(t);
    }

    ok = ok && obs->initialized && obs->num_updates == 3;
    ok = ok && fabsf(obs->running_min - (-1.5f)) < 1e-6f;
    ok = ok && fabsf(obs->running_max -   2.5f ) < 1e-6f;

    QuantParams qp = cml_qat_observer_params(obs);
    ok = ok && fabsf(qp.scale - (2.5f / 127.0f)) < 1e-6f && qp.zero_point == 0;

    cml_qat_observer_reset(obs);
    ok = ok && !obs->initialized && obs->num_updates == 0;

    cml_qat_observer_free(obs);
    return ok;
}

/* Moving-average mode: an outlier batch must decay away geometrically. Each
 * batch here spans its range deterministically (linspace), so after enough
 * updates the running stats sit exactly on the in-range extremes up to the
 * vanishing outlier residue 4 * 0.5^49. */
static int test_observer_ema(void) {
    QatObserver* obs =
        cml_qat_observer_create(CML_QAT_OBS_MOVING_AVG_MINMAX, 0.5f);
    if (!obs) return 0;

    int ok = 1;
    float batch[64];

    /* NOTE: tensor_from_data copies its buffer, so each batch gets a fresh
     * tensor rather than mutating one shared tensor's snapshot */
    Tensor* t = NULL;
    for (int b = 0; b <= 50 && ok; b++) {
        float lo = (b == 0) ? -4.0f : -1.0f;
        float hi = (b == 0) ?  4.0f :  1.0f;
        for (int j = 0; j < 64; j++)
            batch[j] = lo + (hi - lo) * (float)j / 63.0f;
        tensor_free(t);
        t = tensor_from_data(batch, (int[]){64}, 1, &cpu_f32);
        if (cml_qat_observer_update(obs, t) != 0) ok = 0;
    }
    tensor_free(t);

    ok = ok && obs->num_updates == 51;
    ok = ok && fabsf(obs->running_min - (-1.0f)) < 1e-3f;
    ok = ok && fabsf(obs->running_max -   1.0f ) < 1e-3f;

    cml_qat_observer_free(obs);
    return ok;
}

/* ---- fake quant ---- */

/* Max abs error against the unquantized values must be <= scale/2, and every
 * output must sit exactly on the quantization grid. */
static int test_fake_quant_error_bound(int n) {
    float* xd = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) xd[i] = randf(-3.0f, 3.0f);
    Tensor* x = tensor_from_data(xd, (int[]){n}, 1, &cpu_f32);

    QuantParams qp = cml_quantize_compute_params(x, true); /* symmetric int8 */
    Tensor* fq = cml_qat_fake_quant(x, &qp);
    tensor_ensure_executed(fq);

    int ok = fq && fq->data;
    float max_err = 0.0f;
    for (int i = 0; i < n && ok; i++) {
        float v = tensor_get_float(fq, i);
        float q = v / qp.scale;
        if (fabsf(q - roundf(q)) > 1e-4f) { ok = 0; break; }   /* on-grid */
        float err = fabsf(v - xd[i]);
        if (err > max_err) max_err = err;
    }
    ok = ok && max_err <= 0.5f * qp.scale + 1e-6f;

    printf("    [n=%d scale=%.5f max_err=%.5f bound=%.5f] ", n,
           (double)qp.scale, (double)max_err, (double)(0.5f * qp.scale));

    free(xd);
    tensor_free(x);
    tensor_free(fq);
    return ok;
}

/* STE gradient: with loss L = sum(w * fq(x)), dL/dx must equal w exactly
 * (identity pass-through), including for elements outside the calibrated
 * range. Cross-checked against central differences of the STE surrogate
 * f(x) = sum(w * (x + c)) with c = fq - x frozen from the base evaluation —
 * differencing the raw forward would probe the piecewise-constant rounding,
 * not the estimator. */
static int test_ste_gradient_identity(void) {
    const int n = 8;
    float xd[8], wd[8];
    for (int i = 0; i < n; i++) {
        xd[i] = randf(-1.0f, 1.0f);
        wd[i] = randf(-2.0f, 2.0f);
    }
    xd[6] = 25.0f;   /* far above any calibrated grid point */
    xd[7] = -25.0f;  /* far below */

    Tensor* x = tensor_from_data(xd, (int[]){n}, 1, &cpu_f32);
    Tensor* w = tensor_from_data(wd, (int[]){n}, 1, &cpu_f32);
    tensor_set_requires_grad(x, true);

    QuantParams qp = cml_quantize_compute_params(x, true);
    Tensor* fq   = cml_qat_fake_quant(x, &qp);
    ReduceParams rp = {NULL, 0, false};
    Tensor* loss = uop_sum(uop_mul(fq, w), &rp);
    tensor_backward(loss, NULL, false, false);

    Tensor* grad = x->grad;
    int ok = grad && grad->data;

    /* analytic STE gradient == upstream weight vector */
    for (int i = 0; i < n && ok; i++)
        if (fabsf(tensor_get_float(grad, i) - wd[i]) > 1e-5f) ok = 0;

    /* finite-difference cross-check on the surrogate */
    float corr[8], base[8];
    for (int i = 0; i < n; i++) {
        base[i] = tensor_get_float(x, i);
        corr[i] = tensor_get_float(fq, i) - base[i];
    }
    static const int probes[4] = {0, 3, 6, 7}; /* includes out-of-range idx */
    const float h = 1e-3f;
    for (int p = 0; p < 4 && ok; p++) {
        int i = probes[p];
        double fp = 0.0, fm = 0.0; /* f32 accumulation loses the small
                                      central difference to rounding */
        for (int j = 0; j < n; j++) {
            double xp = (double)base[j] + ((j == i) ?  h : 0.0f);
            double xm = (double)base[j] + ((j == i) ? -h : 0.0f);
            fp += (double)wd[j] * (xp + corr[j]);
            fm += (double)wd[j] * (xm + corr[j]);
        }
        double fd  = (fp - fm) / (2.0 * h);
        float ana = tensor_get_float(grad, i);
        if (fabs(fd - ana) > 1e-4) ok = 0;
    }

    cml_ir_reset_global_context();
    tensor_free(x); tensor_free(w); tensor_free(fq);
    return ok;
}

/* ---- tiny end-to-end: linear layer trained through weight fake-quant ---- */

/* y = X @ W with W fake-quantized each step (observer recalibrating as SGD
 * moves it). The loss must drop well below its starting value. */
static int test_linear_training_with_weight_fake_quant(void) {
    const int N = 16, K = 3;
    float xd[N * K], td[N];
    const float true_w[3] = {0.5f, -1.25f, 2.0f};
    for (int i = 0; i < N * K; i++) xd[i] = randf(-1.0f, 1.0f);
    for (int n = 0; n < N; n++) {
        td[n] = 0.0f;
        for (int k = 0; k < K; k++) td[n] += xd[n * K + k] * true_w[k];
    }

    Tensor* X = tensor_from_data(xd, (int[]){N, K}, 2, &cpu_f32);
    Tensor* T = tensor_from_data(td, (int[]){N, 1}, 2, &cpu_f32);

    float w_init[3];
    for (int i = 0; i < K; i++) w_init[i] = randf(-0.1f, 0.1f);
    Tensor* W = tensor_from_data(w_init, (int[]){K, 1}, 2, &cpu_f32);
    tensor_set_requires_grad(W, true);

    QatObserver* obs = cml_qat_observer_create(CML_QAT_OBS_MINMAX, 0.0f);
    const float lr = 0.05f;
    float first_loss = INFINITY, final_loss = INFINITY;

    for (int step = 0; step < 120; step++) {
        tensor_zero_grad(W);

        Tensor* Wq   = cml_qat_fake_quant_observed(W, obs);
        Tensor* y    = uop_matmul(X, Wq);
        Tensor* loss = tensor_mse_loss(y, T);
        tensor_backward(loss, NULL, false, false);

        float lv = tensor_get_float(loss, 0);
        if (step == 0) first_loss = lv;
        final_loss = lv;

        /* manual SGD on the underlying weight data */
        for (int i = 0; i < K; i++) {
            float g = tensor_get_float(W->grad, i);
            tensor_set_float(W, i, tensor_get_float(W, i) - lr * g);
        }

        tensor_free(y); tensor_free(loss); tensor_free(Wq);
        cml_ir_reset_global_context();
    }

    int ok = final_loss < first_loss * 0.25f;
    printf("    [loss %.6f -> %.6f] ", (double)first_loss, (double)final_loss);

    cml_qat_observer_free(obs);
    tensor_free(X); tensor_free(T); tensor_free(W);
    return ok;
}

int main(void) {
    printf("=== QAT: min/max observers + fake-quant STE ===\n");
    check("observer_minmax_convergence",      test_observer_minmax());
    check("observer_moving_avg_convergence",  test_observer_ema());
    check("fake_quant_error_bound_256",       test_fake_quant_error_bound(256));
    check("fake_quant_error_bound_1000",      test_fake_quant_error_bound(1000));
    check("ste_gradient_identity",            test_ste_gradient_identity());
    check("linear_weight_fake_quant_trains",  test_linear_training_with_weight_fake_quant());
    return TEST_SUMMARY();
}
