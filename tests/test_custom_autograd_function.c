#include <stdio.h>
#include <math.h>

#include "cml.h"
#include "autograd/custom_function.h"
#include "test_harness.h"
#include "test_require.h"

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-3f)

/* ── custom op: elementwise square, with save_for_backward ──────────────────
 * forward:  y = x^2   (saves x for the backward)
 * backward: dx = grad_out * 2x
 */
static int g_square_saw_saved = 0;

static Tensor* square_forward(CMLAutogradContext* ctx, Tensor** inputs, int n_inputs,
                              void* user_data) {
    (void)user_data;
    REQUIRE(n_inputs == 1);
    Tensor* x   = inputs[0];
    Tensor* out = tensor_clone(x);
    REQUIRE(out != NULL);
    for (size_t i = 0; i < out->numel; i++) {
        float v = tensor_get_float(x, i);
        tensor_set_float(out, i, v * v);
    }
    cml_autograd_ctx_save(ctx, &x, 1); /* stash x for backward */
    return out;
}

static void square_backward(CMLAutogradContext* ctx, Tensor* grad_output, Tensor** grad_inputs,
                            int n_inputs, void* user_data) {
    (void)user_data;
    REQUIRE(n_inputs == 1);
    int n_saved     = 0;
    Tensor** saved  = cml_autograd_ctx_saved(ctx, &n_saved);
    g_square_saw_saved = (saved && n_saved == 1);
    REQUIRE(g_square_saw_saved);
    Tensor* x  = saved[0];
    Tensor* gi = tensor_clone(grad_output);
    for (size_t i = 0; i < gi->numel; i++) {
        float g = tensor_get_float(grad_output, i);
        float v = tensor_get_float(x, i);
        tensor_set_float(gi, i, g * 2.0f * v);
    }
    grad_inputs[0] = gi;
}

/* ── custom op: out = a*b + a  (two inputs, scratch via set_data) ────────────
 * backward: da = g*(b+1), db = g*a
 */
static Tensor* muladd_forward(CMLAutogradContext* ctx, Tensor** inputs, int n_inputs,
                              void* user_data) {
    (void)user_data;
    REQUIRE(n_inputs == 2);
    Tensor* a   = inputs[0];
    Tensor* b   = inputs[1];
    Tensor* out = tensor_clone(a);
    REQUIRE(out != NULL);
    for (size_t i = 0; i < out->numel; i++) {
        float av = tensor_get_float(a, i);
        float bv = tensor_get_float(b, i);
        tensor_set_float(out, i, av * bv + av);
    }
    Tensor* saved[2] = {a, b};
    cml_autograd_ctx_save(ctx, saved, 2);
    return out;
}

static void muladd_backward(CMLAutogradContext* ctx, Tensor* grad_output, Tensor** grad_inputs,
                            int n_inputs, void* user_data) {
    (void)user_data;
    REQUIRE(n_inputs == 2);
    int n_saved    = 0;
    Tensor** saved = cml_autograd_ctx_saved(ctx, &n_saved);
    REQUIRE(saved && n_saved == 2);
    Tensor* a   = saved[0];
    Tensor* b   = saved[1];
    Tensor* da  = tensor_clone(grad_output);
    Tensor* db  = tensor_clone(grad_output);
    for (size_t i = 0; i < da->numel; i++) {
        float g  = tensor_get_float(grad_output, i);
        float av = tensor_get_float(a, i);
        float bv = tensor_get_float(b, i);
        tensor_set_float(da, i, g * (bv + 1.0f));
        tensor_set_float(db, i, g * av);
    }
    grad_inputs[0] = da;
    grad_inputs[1] = db;
}

/* y = square(x) then z = y * w, loss = sum(z).
 * dL/dy = w  (z = y*w, loss = Σz)   -> dL/dx = w * 2x (via custom backward)
 * dL/dw = y = x^2                    (via the built-in engine directly)
 */
static int test_custom_square_mixed(void) {
    float x_data[] = {1.0f, 2.0f, 3.0f};
    float w_data[] = {0.5f, 1.5f, 2.0f};
    Tensor* x      = cml_tensor_1d(x_data, 3);
    Tensor* w      = cml_tensor_1d(w_data, 3);
    REQUIRE(x && w);
    cml_set_requires_grad(x, true);
    cml_set_requires_grad(w, true);

    Tensor* inputs[] = {x};
    Tensor* y        = cml_autograd_function_apply(square_forward, square_backward, inputs, 1, NULL);
    REQUIRE(y != NULL);

    Tensor* z    = cml_mul(y, w);
    Tensor* loss = cml_sum(z, 0, false);
    REQUIRE(loss != NULL);

    cml_backward(loss, NULL, false, false);

    REQUIRE(x->grad && w->grad);
    int ok = 1;
    for (int i = 0; i < 3; i++) {
        ok &= APPROX_EQ(tensor_get_float(x->grad, i), w_data[i] * 2.0f * x_data[i]);
        ok &= APPROX_EQ(tensor_get_float(w->grad, i), x_data[i] * x_data[i]);
    }
    ok &= g_square_saw_saved;

    tensor_free(x);
    tensor_free(w);
    tensor_free(y);
    tensor_free(z);
    tensor_free(loss);
    return ok;
}

/* Finite-difference check of the custom square gradient against a scalar loss
 * L = Σ square(x). Analytic dL/dx_i = 2 x_i. */
static int test_custom_square_finite_diff(void) {
    float x_data[] = {0.7f, -1.3f, 2.1f, 0.4f};
    const int n    = 4;
    Tensor* x      = cml_tensor_1d(x_data, n);
    REQUIRE(x);
    cml_set_requires_grad(x, true);

    Tensor* inputs[] = {x};
    Tensor* y        = cml_autograd_function_apply(square_forward, square_backward, inputs, 1, NULL);
    Tensor* loss     = cml_sum(y, 0, false);
    cml_backward(loss, NULL, false, false);
    REQUIRE(x->grad);

    int ok       = 1;
    const float h = 1e-3f;
    for (int i = 0; i < n; i++) {
        float fp = 0.0f, fm = 0.0f;
        for (int k = 0; k < n; k++) {
            float xp = x_data[k] + (k == i ? h : 0.0f);
            float xm = x_data[k] - (k == i ? h : 0.0f);
            fp += xp * xp;
            fm += xm * xm;
        }
        float fd = (fp - fm) / (2.0f * h);
        ok &= APPROX_EQ(tensor_get_float(x->grad, i), fd);
    }

    tensor_free(x);
    tensor_free(y);
    tensor_free(loss);
    return ok;
}

/* Two-input custom op mixed with a built-in reduction. */
static int test_custom_muladd(void) {
    float a_data[] = {2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f};
    Tensor* a      = cml_tensor_1d(a_data, 2);
    Tensor* b      = cml_tensor_1d(b_data, 2);
    REQUIRE(a && b);
    cml_set_requires_grad(a, true);
    cml_set_requires_grad(b, true);

    Tensor* inputs[] = {a, b};
    Tensor* out      = cml_autograd_function_apply(muladd_forward, muladd_backward, inputs, 2, NULL);
    Tensor* loss     = cml_sum(out, 0, false);
    cml_backward(loss, NULL, false, false);

    REQUIRE(a->grad && b->grad);
    int ok = 1;
    for (int i = 0; i < 2; i++) {
        /* loss = Σ (a*b + a); dL/da = b+1, dL/db = a */
        ok &= APPROX_EQ(tensor_get_float(a->grad, i), b_data[i] + 1.0f);
        ok &= APPROX_EQ(tensor_get_float(b->grad, i), a_data[i]);
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
    tensor_free(loss);
    return ok;
}

int main(void) {
    printf("Custom Autograd Function Tests\n");
    cml_init();

    TEST(custom_square_mixed);
    TEST(custom_square_finite_diff);
    TEST(custom_muladd);

    cml_autograd_custom_reset();
    return TEST_SUMMARY();
}
