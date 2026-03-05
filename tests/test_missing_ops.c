/**
 * @file test_missing_ops.c
 * @brief Tests for sort, argsort, topk, cumprod, bitwise ops, nonzero, masked_fill
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor/tensor.h"
#include "ops/uops.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

#define APPROX(a, b) (fabsf((a) - (b)) < 1e-5f)

static int test_sort_1d(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 2.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* sorted = uop_sort(t, 0, false);
    tensor_ensure_executed(sorted);
    float* out = sorted->data;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 1.0f)) return 0;
    if (!APPROX(out[2], 2.0f)) return 0;
    if (!APPROX(out[3], 3.0f)) return 0;
    if (!APPROX(out[4], 4.0f)) return 0;
    if (!APPROX(out[5], 5.0f)) return 0;
    tensor_free(t);
    tensor_free(sorted);
    return 1;
}

static int test_sort_descending(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* sorted = uop_sort(t, 0, true);
    tensor_ensure_executed(sorted);
    float* out = sorted->data;
    if (!APPROX(out[0], 4.0f)) return 0;
    if (!APPROX(out[1], 3.0f)) return 0;
    if (!APPROX(out[2], 2.0f)) return 0;
    if (!APPROX(out[3], 1.0f)) return 0;
    tensor_free(t);
    tensor_free(sorted);
    return 1;
}

static int test_argsort_1d(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* indices = uop_argsort(t, 0, false);
    tensor_ensure_executed(indices);
    float* out = indices->data;
    // sorted order: 1.0(idx1), 2.0(idx3), 3.0(idx0), 4.0(idx2)
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 3.0f)) return 0;
    if (!APPROX(out[2], 0.0f)) return 0;
    if (!APPROX(out[3], 2.0f)) return 0;
    tensor_free(t);
    tensor_free(indices);
    return 1;
}

static int test_topk(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* idx_out = NULL;
    Tensor* values = uop_topk(t, 3, 0, true, &idx_out);
    tensor_ensure_executed(values);
    float* out = values->data;
    // top 3 largest: 9, 5, 4
    if (!APPROX(out[0], 9.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    if (!APPROX(out[2], 4.0f)) return 0;
    tensor_free(t);
    tensor_free(values);
    if (idx_out) tensor_free(idx_out);
    return 1;
}

static int test_cumprod(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* result = uop_cumprod(t, 0);
    tensor_ensure_executed(result);
    float* out = result->data;
    // cumprod: 1, 1*2=2, 2*3=6, 6*4=24
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 2.0f)) return 0;
    if (!APPROX(out[2], 6.0f)) return 0;
    if (!APPROX(out[3], 24.0f)) return 0;
    tensor_free(t);
    tensor_free(result);
    return 1;
}

static int test_bitwise_and(void) {
    float data_a[] = {7.0f, 5.0f, 3.0f, 15.0f};
    float data_b[] = {3.0f, 6.0f, 1.0f, 10.0f};
    int shape[] = {4};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_and(a, b);
    tensor_ensure_executed(result);
    float* out = result->data;
    // 7&3=3, 5&6=4, 3&1=1, 15&10=10
    if (!APPROX(out[0], 3.0f)) return 0;
    if (!APPROX(out[1], 4.0f)) return 0;
    if (!APPROX(out[2], 1.0f)) return 0;
    if (!APPROX(out[3], 10.0f)) return 0;
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_bitwise_or(void) {
    float data_a[] = {5.0f, 3.0f};
    float data_b[] = {3.0f, 6.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_or(a, b);
    tensor_ensure_executed(result);
    float* out = result->data;
    // 5|3=7, 3|6=7
    if (!APPROX(out[0], 7.0f)) return 0;
    if (!APPROX(out[1], 7.0f)) return 0;
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_bitwise_xor(void) {
    float data_a[] = {5.0f, 3.0f};
    float data_b[] = {3.0f, 6.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_xor(a, b);
    tensor_ensure_executed(result);
    float* out = result->data;
    // 5^3=6, 3^6=5
    if (!APPROX(out[0], 6.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_bitwise_not(void) {
    float data[] = {0.0f, 5.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_not(a);
    tensor_ensure_executed(result);
    float* out = result->data;
    // ~0 = -1, ~5 = -6
    if (!APPROX(out[0], -1.0f)) return 0;
    if (!APPROX(out[1], -6.0f)) return 0;
    tensor_free(a);
    tensor_free(result);
    return 1;
}

static int test_nonzero(void) {
    float data[] = {0.0f, 3.0f, 0.0f, 5.0f, 7.0f, 0.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* result = uop_nonzero(t);
    tensor_ensure_executed(result);
    float* out = result->data;
    // nonzero indices: 1, 3, 4
    if (result->shape[0] < 3) return 0;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 3.0f)) return 0;
    if (!APPROX(out[2], 4.0f)) return 0;
    tensor_free(t);
    tensor_free(result);
    return 1;
}

static int test_masked_fill(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float mask_data[] = {1.0f, 0.0f, 1.0f, 0.0f};
    int shape[] = {4};
    Tensor* a = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* mask = tensor_from_data(mask_data, shape, 1, &cpu_f32);
    Tensor* result = uop_masked_fill(a, mask, -999.0f);
    tensor_ensure_executed(result);
    float* out = result->data;
    // where mask=1: fill with -999, where mask=0: keep original
    if (!APPROX(out[0], -999.0f)) return 0;
    if (!APPROX(out[1], 2.0f)) return 0;
    if (!APPROX(out[2], -999.0f)) return 0;
    if (!APPROX(out[3], 4.0f)) return 0;
    tensor_free(a);
    tensor_free(mask);
    tensor_free(result);
    return 1;
}

int main(void) {
    printf("=== Missing Ops Tests ===\n");
    TEST(sort_1d);
    TEST(sort_descending);
    TEST(argsort_1d);
    TEST(topk);
    TEST(cumprod);
    TEST(bitwise_and);
    TEST(bitwise_or);
    TEST(bitwise_xor);
    TEST(bitwise_not);
    TEST(nonzero);
    TEST(masked_fill);
    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
