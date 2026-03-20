#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-5f)

static int test_assign_basic(void) {
    int shape[] = {2, 3};
    Tensor* dst = cml_zeros(shape, 2, NULL);
    Tensor* src = cml_ones(shape, 2, NULL);
    if (!dst || !src) return 0;

    Tensor* saved_ptr = dst;
    int ret = tensor_assign(dst, src);
    if (ret != 0) return 0;
    if (dst != saved_ptr) return 0;

    float* data = (float*)dst->data;
    for (size_t i = 0; i < dst->numel; i++) {
        if (!APPROX_EQ(data[i], 1.0f)) {
            tensor_free(src);
            tensor_free(dst);
            return 0;
        }
    }
    tensor_free(src);
    tensor_free(dst);
    return 1;
}

static int test_assign_preserves_grad_flag(void) {
    int shape[] = {4};
    Tensor* dst = cml_zeros(shape, 1, NULL);
    dst->requires_grad = true;
    Tensor* src = cml_ones(shape, 1, NULL);

    tensor_assign(dst, src);
    int ok = dst->requires_grad == true;

    tensor_free(src);
    tensor_free(dst);
    return ok;
}

static int test_assign_shape_mismatch(void) {
    int shape1[] = {2, 3};
    int shape2[] = {3, 2};
    Tensor* dst = cml_zeros(shape1, 2, NULL);
    Tensor* src = cml_ones(shape2, 2, NULL);

    int ret = tensor_assign(dst, src);
    tensor_free(dst);
    tensor_free(src);
    return ret != 0;
}

static int test_assign_ndim_mismatch(void) {
    int shape1[] = {6};
    int shape2[] = {2, 3};
    Tensor* dst = cml_zeros(shape1, 1, NULL);
    Tensor* src = cml_ones(shape2, 2, NULL);

    int ret = tensor_assign(dst, src);
    tensor_free(dst);
    tensor_free(src);
    return ret != 0;
}

static int test_assign_null(void) {
    int shape[] = {2};
    Tensor* t = cml_zeros(shape, 1, NULL);
    int ret1 = tensor_assign(NULL, t);
    int ret2 = tensor_assign(t, NULL);
    tensor_free(t);
    return ret1 != 0 && ret2 != 0;
}

static int test_assign_data_basic(void) {
    int shape[] = {4};
    Tensor* t = cml_zeros(shape, 1, NULL);
    if (!t) return 0;

    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int ret = tensor_assign_data(t, vals, sizeof(vals));
    if (ret != 0) { tensor_free(t); return 0; }

    float* data = (float*)t->data;
    for (int i = 0; i < 4; i++) {
        if (!APPROX_EQ(data[i], vals[i])) { tensor_free(t); return 0; }
    }
    tensor_free(t);
    return 1;
}

static int test_assign_data_marks_executed(void) {
    int shape[] = {3};
    Tensor* t = cml_zeros(shape, 1, NULL);
    t->is_executed = false;

    float vals[] = {1.0f, 2.0f, 3.0f};
    tensor_assign_data(t, vals, sizeof(vals));

    int ok = t->is_executed == true;
    tensor_free(t);
    return ok;
}

static int test_assign_data_null(void) {
    int shape[] = {2};
    Tensor* t = cml_zeros(shape, 1, NULL);
    int ret = tensor_assign_data(t, NULL, 8);
    tensor_free(t);
    return ret != 0;
}

static int test_assign_data_overflow(void) {
    int shape[] = {2};
    Tensor* t = cml_zeros(shape, 1, NULL);
    float big[100];
    int ret = tensor_assign_data(t, big, sizeof(big));
    tensor_free(t);
    return ret != 0;
}

static int test_assign_identity_preserved(void) {
    int shape[] = {3, 3};
    Tensor* dst = cml_zeros(shape, 2, NULL);
    Tensor* src = cml_ones(shape, 2, NULL);
    if (!dst || !src) return 0;

    void* original_ptr = (void*)dst;
    int original_ndim = dst->ndim;

    tensor_assign(dst, src);

    int ok = ((void*)dst == original_ptr) && (dst->ndim == original_ndim);
    tensor_free(src);
    tensor_free(dst);
    return ok;
}

int main(void) {
    printf("Tensor Assign Tests\n");

    TEST(assign_basic);
    TEST(assign_preserves_grad_flag);
    TEST(assign_shape_mismatch);
    TEST(assign_ndim_mismatch);
    TEST(assign_null);
    TEST(assign_data_basic);
    TEST(assign_data_marks_executed);
    TEST(assign_data_null);
    TEST(assign_data_overflow);
    TEST(assign_identity_preserved);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
