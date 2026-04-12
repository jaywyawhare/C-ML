#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cml.h"
#include "autograd/loss_functions.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "tensor/tensor.h"

static void tensor_free_executed(Tensor* t) {
    if (!t)
        return;
    tensor_ensure_executed(t);
    tensor_free(t);
}

static int tests_run    = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_smooth_zero_delegates(void) {
    cml_init();

    float logits_data[] = {2.0f, 1.0f, 0.1f, 0.5f, 1.5f, 0.3f};
    Tensor* logits = tensor_from_array_2d(logits_data, 2, 3);
    float target_data[] = {0.0f, 2.0f};
    Tensor* targets_raw = tensor_from_array_2d(target_data, 1, 2);
    int new_shape[] = {2};
    ReshapeParams rp = {.new_shape = new_shape, .new_ndim = 1};
    Tensor* targets = uop_reshape(targets_raw, &rp);
    tensor_free(targets_raw);

    if (!logits || !targets) {
        if (logits) tensor_free(logits);
        if (targets) tensor_free(targets);
        cml_reset_ir_context();
        cml_cleanup();
        return 0;
    }

    Tensor* loss_smooth = tensor_cross_entropy_loss_smooth(logits, targets, 0.0f);
    Tensor* loss_plain = tensor_cross_entropy_loss(logits, targets);

    int ok = (loss_smooth != NULL && loss_plain != NULL);

    if (loss_smooth) tensor_free_executed(loss_smooth);
    if (loss_plain) tensor_free_executed(loss_plain);
    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

static int test_smooth_nonzero_returns(void) {
    cml_init();

    float logits_data[] = {2.0f, 1.0f, 0.1f, 0.5f, 1.5f, 0.3f};
    Tensor* logits = tensor_from_array_2d(logits_data, 2, 3);
    float target_data[] = {0.0f, 2.0f};
    Tensor* targets_raw = tensor_from_array_2d(target_data, 1, 2);
    int new_shape[] = {2};
    ReshapeParams rp = {.new_shape = new_shape, .new_ndim = 1};
    Tensor* targets = uop_reshape(targets_raw, &rp);
    tensor_free(targets_raw);

    if (!logits || !targets) {
        if (logits) tensor_free(logits);
        if (targets) tensor_free(targets);
        cml_reset_ir_context();
        cml_cleanup();
        return 0;
    }

    Tensor* loss = tensor_cross_entropy_loss_smooth(logits, targets, 0.1f);
    int ok = (loss != NULL);

    if (loss) tensor_free_executed(loss);
    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

static int test_smooth_null_input(void) {
    Tensor* r1 = tensor_cross_entropy_loss_smooth(NULL, NULL, 0.1f);
    Tensor* r2 = tensor_sparse_cross_entropy_loss_smooth(NULL, NULL, 0.1f);
    return (r1 == NULL && r2 == NULL);
}

static int test_smooth_invalid_epsilon(void) {
    cml_init();

    float logits_data[] = {1.0f, 2.0f, 3.0f};
    Tensor* logits = tensor_from_array_2d(logits_data, 1, 3);
    float target_data[] = {1.0f};
    Tensor* targets_raw = tensor_from_array_2d(target_data, 1, 1);
    int new_shape[] = {1};
    ReshapeParams rp = {.new_shape = new_shape, .new_ndim = 1};
    Tensor* targets = uop_reshape(targets_raw, &rp);
    tensor_free(targets_raw);

    Tensor* r = tensor_cross_entropy_loss_smooth(logits, targets, -0.5f);
    int ok = (r == NULL);

    r = tensor_cross_entropy_loss_smooth(logits, targets, 1.5f);
    ok = ok && (r == NULL);

    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

static int test_sparse_smooth_zero_delegates(void) {
    cml_init();

    float logits_data[] = {2.0f, 1.0f, 0.1f, 0.5f, 1.5f, 0.3f};
    Tensor* logits = tensor_from_array_2d(logits_data, 2, 3);
    float target_data[] = {0.0f, 2.0f};
    Tensor* targets_raw = tensor_from_array_2d(target_data, 1, 2);
    int new_shape[] = {2};
    ReshapeParams rp = {.new_shape = new_shape, .new_ndim = 1};
    Tensor* targets = uop_reshape(targets_raw, &rp);
    tensor_free(targets_raw);

    if (!logits || !targets) {
        if (logits) tensor_free(logits);
        if (targets) tensor_free(targets);
        cml_reset_ir_context();
        cml_cleanup();
        return 0;
    }

    Tensor* loss = tensor_sparse_cross_entropy_loss_smooth(logits, targets, 0.0f);
    int ok = (loss != NULL);

    if (loss) tensor_free_executed(loss);
    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

static int test_sparse_smooth_nonzero(void) {
    cml_init();

    float logits_data[] = {2.0f, 1.0f, 0.1f, 0.5f, 1.5f, 0.3f};
    Tensor* logits = tensor_from_array_2d(logits_data, 2, 3);
    float target_data[] = {0.0f, 2.0f};
    Tensor* targets_raw = tensor_from_array_2d(target_data, 1, 2);
    int new_shape[] = {2};
    ReshapeParams rp = {.new_shape = new_shape, .new_ndim = 1};
    Tensor* targets = uop_reshape(targets_raw, &rp);
    tensor_free(targets_raw);

    if (!logits || !targets) {
        if (logits) tensor_free(logits);
        if (targets) tensor_free(targets);
        cml_reset_ir_context();
        cml_cleanup();
        return 0;
    }

    Tensor* loss = tensor_sparse_cross_entropy_loss_smooth(logits, targets, 0.2f);
    int ok = (loss != NULL);

    if (loss) tensor_free_executed(loss);
    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

static int test_smooth_batch_mismatch(void) {
    cml_init();

    float logits_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor* logits = tensor_from_array_2d(logits_data, 2, 3);
    float target_data[] = {0.0f, 1.0f, 2.0f};
    Tensor* targets_raw = tensor_from_array_2d(target_data, 1, 3);
    int new_shape[] = {3};
    ReshapeParams rp = {.new_shape = new_shape, .new_ndim = 1};
    Tensor* targets = uop_reshape(targets_raw, &rp);
    tensor_free(targets_raw);

    Tensor* r = tensor_cross_entropy_loss_smooth(logits, targets, 0.1f);
    int ok = (r == NULL);

    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

static int test_smooth_1d_target_required(void) {
    cml_init();

    float logits_data[] = {1.0f, 2.0f, 3.0f};
    Tensor* logits = tensor_from_array_2d(logits_data, 1, 3);
    float target_data[] = {0.0f, 1.0f};
    Tensor* targets = tensor_from_array_2d(target_data, 1, 2);

    Tensor* r = tensor_cross_entropy_loss_smooth(logits, targets, 0.1f);
    int ok = (r == NULL);

    tensor_free(logits);
    tensor_free(targets);
    cml_reset_ir_context();
    cml_cleanup();
    return ok;
}

int main(void) {
    printf("=== Label Smoothing Tests ===\n");

    RUN_TEST(test_smooth_zero_delegates);
    RUN_TEST(test_smooth_nonzero_returns);
    RUN_TEST(test_smooth_null_input);
    RUN_TEST(test_smooth_invalid_epsilon);
    RUN_TEST(test_sparse_smooth_zero_delegates);
    RUN_TEST(test_sparse_smooth_nonzero);
    RUN_TEST(test_smooth_batch_mismatch);
    RUN_TEST(test_smooth_1d_target_required);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
