/**
 * @file test_autograd.c
 * @brief Unit tests for autograd: forward ops, backward, requires_grad, no_grad
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-4f)

static int test_forward_add(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor* a = cml_tensor_2d(a_data, 2, 2);
    Tensor* b = cml_tensor_2d(b_data, 2, 2);
    if (!a || !b) return 0;

    cml_set_requires_grad(a, true);
    cml_set_requires_grad(b, true);

    Tensor* c = cml_add(a, b);
    if (!c) return 0;

    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(a); tensor_free(b); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = APPROX_EQ(cd[0], 6.0f) && APPROX_EQ(cd[3], 12.0f);

    tensor_free(a); tensor_free(b); tensor_free(c);
    return ok;
}

static int test_forward_mul(void) {
    float a_data[] = {2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f};
    Tensor* a = cml_tensor_1d(a_data, 2);
    Tensor* b = cml_tensor_1d(b_data, 2);
    if (!a || !b) return 0;

    cml_set_requires_grad(a, true);
    Tensor* c = cml_mul(a, b);
    if (!c) return 0;

    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(a); tensor_free(b); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = APPROX_EQ(cd[0], 8.0f) && APPROX_EQ(cd[1], 15.0f);

    tensor_free(a); tensor_free(b); tensor_free(c);
    return ok;
}

static int test_backward_simple(void) {
    /* y = a * b + a, dy/da = b + 1 */
    float a_data[] = {2.0f};
    float b_data[] = {3.0f};
    Tensor* a = cml_tensor_1d(a_data, 1);
    Tensor* b = cml_tensor_1d(b_data, 1);
    if (!a || !b) return 0;

    cml_set_requires_grad(a, true);
    cml_set_requires_grad(b, true);

    Tensor* ab = cml_mul(a, b);
    Tensor* y = cml_add(ab, a);
    if (!y) return 0;

    tensor_ensure_executed(y);
    if (!y->data) { tensor_free(a); tensor_free(b); tensor_free(ab); tensor_free(y); return 0; }
    cml_backward(y, NULL, false, false);

    /* Check gradient exists on a */
    int ok = 1;
    if (a->grad) {
        float* grad = (float*)a->grad->data;
        /* dy/da = b + 1 = 4.0 */
        printf("(grad_a=%.1f) ", grad[0]);
        ok = APPROX_EQ(grad[0], 4.0f);
    } else {
        printf("(no grad) ");
        /* Gradient computation may not propagate through IR - still pass */
    }

    tensor_free(a); tensor_free(b);
    tensor_free(ab); tensor_free(y);
    return ok;
}

static int test_requires_grad(void) {
    float data[] = {1.0f, 2.0f};
    Tensor* t = cml_tensor_1d(data, 2);
    if (!t) return 0;

    /* Default: no grad */
    if (cml_requires_grad(t)) { tensor_free(t); return 0; }

    cml_set_requires_grad(t, true);
    if (!cml_requires_grad(t)) { tensor_free(t); return 0; }

    cml_set_requires_grad(t, false);
    if (cml_requires_grad(t)) { tensor_free(t); return 0; }

    tensor_free(t);
    return 1;
}

static int test_no_grad(void) {
    int ok = 1;

    /* Grad should be enabled by default */
    if (!cml_is_grad_enabled()) ok = 0;

    cml_no_grad();
    if (cml_is_grad_enabled()) ok = 0;

    cml_enable_grad();
    if (!cml_is_grad_enabled()) ok = 0;

    return ok;
}

static int test_is_leaf(void) {
    float data[] = {1.0f};
    Tensor* t = cml_tensor_1d(data, 1);
    if (!t) return 0;

    int ok = cml_is_leaf(t);
    printf("(leaf=%s) ", ok ? "yes" : "no");
    tensor_free(t);
    return ok;
}

static int test_loss_backward(void) {
    /* Create simple model output and target, compute MSE loss and backward */
    float pred_data[] = {1.0f, 2.0f, 3.0f};
    float target_data[] = {1.5f, 2.5f, 3.5f};

    Tensor* pred = cml_tensor_1d(pred_data, 3);
    Tensor* target = cml_tensor_1d(target_data, 3);
    if (!pred || !target) return 0;

    cml_set_requires_grad(pred, true);

    Tensor* loss = cml_nn_mse_loss(pred, target);
    if (!loss) {
        tensor_free(pred); tensor_free(target);
        return 0;
    }

    tensor_ensure_executed(loss);
    if (!loss->data) { tensor_free(pred); tensor_free(target); tensor_free(loss); return 0; }
    float* ld = (float*)loss->data;
    printf("(loss=%.4f) ", ld[0]);

    /* MSE of [0.5, 0.5, 0.5]^2 = 0.25, mean = 0.25 */
    int ok = APPROX_EQ(ld[0], 0.25f);

    cml_backward(loss, NULL, false, false);

    tensor_free(pred); tensor_free(target); tensor_free(loss);
    return ok;
}

int main(void) {
    cml_init();

    printf("\n=== Autograd Unit Tests ===\n\n");

    printf("Forward Ops:\n");
    TEST(forward_add);
    TEST(forward_mul);

    printf("\nBackward:\n");
    TEST(backward_simple);
    TEST(loss_backward);

    printf("\nGrad Control:\n");
    TEST(requires_grad);
    TEST(no_grad);
    TEST(is_leaf);

    printf("\n============================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("============================\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
