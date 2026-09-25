/* Gradient checkpointing: does it actually trade memory for compute?
 *
 * autograd_checkpoint() saved the IR linkage and detached the node but never
 * released the activation buffer, so the feature cost recompute and saved
 * nothing -- the one thing it exists to do. These tests pin both halves: the
 * activation is really gone after checkpointing, and autograd_recompute rebuilds
 * the same values from the saved graph.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "autograd/checkpointing.h"
#include "test_harness.h"
#include "test_require.h"

#define EPS 1e-5f

static Tensor* leaf(const float* d, int n) {
    int shape[1]     = {n};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return tensor_from_data(d, shape, 1, &cfg);
}

static bool values_match(Tensor* t, const float* want, int n) {
    if (!t || (int)t->numel != n || !t->data)
        return false;
    const float* got = (const float*)t->data;
    for (int i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > EPS) {
            printf("(elem %d: got %g want %g) ", i, (double)got[i], (double)want[i]);
            return false;
        }
    }
    return true;
}

static bool test_enable_disable(void) {
    autograd_set_checkpointing(false);
    bool off = !autograd_is_checkpointing_enabled();
    autograd_set_checkpointing(true);
    bool on = autograd_is_checkpointing_enabled();
    autograd_set_checkpointing(false);
    return off && on && !autograd_is_checkpointing_enabled();
}

/* Checkpointing while disabled must be a no-op that reports failure rather than
 * silently dropping an activation nothing will recompute. */
static bool test_checkpoint_refused_when_disabled(void) {
    autograd_set_checkpointing(false);

    const float ad[3] = {1.0f, 2.0f, 3.0f};
    const float bd[3] = {4.0f, 5.0f, 6.0f};
    Tensor* a         = leaf(ad, 3);
    Tensor* b         = leaf(bd, 3);
    Tensor* y         = uop_add(a, b);
    REQUIRE(y && tensor_ensure_executed(y) == 0);

    bool ok = autograd_checkpoint(y) == -1 && y->data != NULL;
    ok      = ok && autograd_checkpoint(NULL) == -1;

    cml_reset_ir_context();
    return ok;
}

/* The core claim: after checkpointing, the activation buffer is released. */
static bool test_checkpoint_releases_activation(void) {
    autograd_set_checkpointing(true);

    const float ad[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float bd[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    Tensor* a         = leaf(ad, 4);
    Tensor* b         = leaf(bd, 4);
    Tensor* y         = uop_mul(a, b);
    REQUIRE(y && tensor_ensure_executed(y) == 0);
    REQUIRE(y->data != NULL); /* realized before checkpointing */

    bool ok = autograd_checkpoint(y) == 0;
    if (ok && y->data != NULL) {
        printf("(activation still resident) ");
        ok = false;
    }
    if (ok && y->is_executed) {
        printf("(still marked executed) ");
        ok = false;
    }
    /* Detached from the graph so the forward graph can be torn down. */
    if (ok && y->ir_node != NULL) {
        printf("(still attached to IR) ");
        ok = false;
    }

    autograd_checkpointing_cleanup();
    autograd_set_checkpointing(false);
    cml_reset_ir_context();
    return ok;
}

/* ...and recompute must bring the values back, from the saved graph alone. */
static bool test_recompute_restores_values(void) {
    autograd_set_checkpointing(true);

    const float ad[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    const float bd[4]   = {2.0f, 3.0f, 4.0f, 5.0f};
    const float want[4] = {3.0f, 5.0f, 7.0f, 9.0f}; /* a + b */

    Tensor* a = leaf(ad, 4);
    Tensor* b = leaf(bd, 4);
    Tensor* y = uop_add(a, b);
    REQUIRE(y && tensor_ensure_executed(y) == 0);
    REQUIRE(values_match(y, want, 4)); /* forward is right to begin with */

    REQUIRE(autograd_checkpoint(y) == 0);
    REQUIRE(y->data == NULL); /* genuinely dropped */

    Tensor* back = autograd_recompute(y);
    bool ok      = back == y && values_match(y, want, 4);

    autograd_checkpointing_cleanup();
    autograd_set_checkpointing(false);
    cml_reset_ir_context();
    return ok;
}

/* A multiply, to show the recomputation follows the recorded op rather than
 * defaulting to something that happens to match. */
static bool test_recompute_uses_the_recorded_op(void) {
    autograd_set_checkpointing(true);

    const float ad[3]   = {2.0f, 3.0f, 4.0f};
    const float bd[3]   = {5.0f, 6.0f, 7.0f};
    const float want[3] = {10.0f, 18.0f, 28.0f}; /* a * b, not a + b */

    Tensor* a = leaf(ad, 3);
    Tensor* b = leaf(bd, 3);
    Tensor* y = uop_mul(a, b);
    REQUIRE(y && tensor_ensure_executed(y) == 0);
    REQUIRE(autograd_checkpoint(y) == 0);

    Tensor* back = autograd_recompute(y);
    bool ok      = back == y && values_match(y, want, 3);

    autograd_checkpointing_cleanup();
    autograd_set_checkpointing(false);
    cml_reset_ir_context();
    return ok;
}

/* Several checkpoints at once, each recomputed independently. */
static bool test_multiple_checkpoints(void) {
    autograd_set_checkpointing(true);

    const float ad[2] = {1.0f, 2.0f};
    const float bd[2] = {10.0f, 20.0f};
    Tensor* a         = leaf(ad, 2);
    Tensor* b         = leaf(bd, 2);

    Tensor* sum  = uop_add(a, b);
    Tensor* prod = uop_mul(a, b);
    REQUIRE(sum && prod);
    REQUIRE(tensor_ensure_executed(sum) == 0 && tensor_ensure_executed(prod) == 0);
    REQUIRE(autograd_checkpoint(sum) == 0 && autograd_checkpoint(prod) == 0);

    bool ok = sum->data == NULL && prod->data == NULL;

    const float want_sum[2]  = {11.0f, 22.0f};
    const float want_prod[2] = {10.0f, 40.0f};
    ok = ok && autograd_recompute(sum) == sum && values_match(sum, want_sum, 2);
    ok = ok && autograd_recompute(prod) == prod && values_match(prod, want_prod, 2);

    autograd_checkpointing_cleanup();
    autograd_set_checkpointing(false);
    cml_reset_ir_context();
    return ok;
}

/* Recomputing something that was never checkpointed hands the tensor back
 * unchanged rather than inventing a value. */
static bool test_recompute_unknown_tensor(void) {
    autograd_set_checkpointing(true);

    const float ad[2] = {7.0f, 8.0f};
    Tensor* a         = leaf(ad, 2);
    REQUIRE(a && tensor_ensure_executed(a) == 0);

    bool ok = autograd_recompute(a) == a && values_match(a, ad, 2);
    ok      = ok && autograd_recompute(NULL) == NULL;

    autograd_checkpointing_cleanup();
    autograd_set_checkpointing(false);
    cml_reset_ir_context();
    return ok;
}

/* cleanup must be safe to call twice and with nothing registered -- it releases
 * the references autograd_checkpoint took on the saved inputs. */
static bool test_cleanup_is_idempotent(void) {
    autograd_checkpointing_cleanup();
    autograd_checkpointing_cleanup();

    autograd_set_checkpointing(true);
    const float ad[2] = {1.0f, 1.0f};
    Tensor* a         = leaf(ad, 2);
    Tensor* y         = uop_add(a, a);
    REQUIRE(y && tensor_ensure_executed(y) == 0);
    REQUIRE(autograd_checkpoint(y) == 0);
    autograd_checkpointing_cleanup();
    autograd_checkpointing_cleanup();

    autograd_set_checkpointing(false);
    cml_reset_ir_context();
    return true;
}

int main(void) {
    printf("Gradient Checkpointing Tests\n\n");

    printf("Toggle:\n");
    TEST(enable_disable);
    TEST(checkpoint_refused_when_disabled);

    printf("\nMemory is actually reclaimed:\n");
    TEST(checkpoint_releases_activation);

    printf("\nRecomputation:\n");
    TEST(recompute_restores_values);
    TEST(recompute_uses_the_recorded_op);
    TEST(multiple_checkpoints);
    TEST(recompute_unknown_tensor);

    printf("\nTeardown:\n");
    TEST(cleanup_is_idempotent);

    return TEST_SUMMARY();
}
