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

#define APPROX(a, b) (fabsf((a) - (b)) < 1e-4f)

static int test_sort_ascending(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_sort(t, 0, false);
    tensor_ensure_executed(out);
    float* d = out->data;
    if (!APPROX(d[0], 1.0f)) return 0;
    if (!APPROX(d[1], 1.0f)) return 0;
    if (!APPROX(d[2], 3.0f)) return 0;
    if (!APPROX(d[3], 4.0f)) return 0;
    if (!APPROX(d[4], 5.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_sort_descending(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_sort(t, 0, true);
    tensor_ensure_executed(out);
    float* d = out->data;
    // Descending: 5, 4, 3, 1, 1
    if (!APPROX(d[0], 5.0f)) return 0;
    if (!APPROX(d[1], 4.0f)) return 0;
    if (!APPROX(d[2], 3.0f)) return 0;
    if (!APPROX(d[3], 1.0f)) return 0;
    if (!APPROX(d[4], 1.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_topk(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    int shape[] = {8};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_topk(t, 3, 0, true, NULL);
    tensor_ensure_executed(out);
    if (out->numel != 3) return 0;
    float* d = out->data;
    // Top-3 largest: 9, 6, 5
    if (!APPROX(d[0], 9.0f)) return 0;
    if (!APPROX(d[1], 6.0f)) return 0;
    if (!APPROX(d[2], 5.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_any_all(void) {
    // Test ANY: at least one non-zero
    {
        float data[] = {0.0f, 0.0f, 1.0f, 0.0f};
        int shape[] = {4};
        Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
        Tensor* out = uop_any(t, NULL);
        tensor_ensure_executed(out);
        if (!APPROX(((float*)out->data)[0], 1.0f)) { tensor_free(t); tensor_free(out); return 0; }
        tensor_free(t);
        tensor_free(out);
    }
    // Test ANY: all zeros
    {
        float data[] = {0.0f, 0.0f, 0.0f};
        int shape[] = {3};
        Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
        Tensor* out = uop_any(t, NULL);
        tensor_ensure_executed(out);
        if (!APPROX(((float*)out->data)[0], 0.0f)) { tensor_free(t); tensor_free(out); return 0; }
        tensor_free(t);
        tensor_free(out);
    }
    // Test ALL: all non-zero
    {
        float data[] = {1.0f, 2.0f, 3.0f};
        int shape[] = {3};
        Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
        Tensor* out = uop_all(t, NULL);
        tensor_ensure_executed(out);
        if (!APPROX(((float*)out->data)[0], 1.0f)) { tensor_free(t); tensor_free(out); return 0; }
        tensor_free(t);
        tensor_free(out);
    }
    // Test ALL: one zero
    {
        float data[] = {1.0f, 0.0f, 3.0f};
        int shape[] = {3};
        Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
        Tensor* out = uop_all(t, NULL);
        tensor_ensure_executed(out);
        if (!APPROX(((float*)out->data)[0], 0.0f)) { tensor_free(t); tensor_free(out); return 0; }
        tensor_free(t);
        tensor_free(out);
    }
    return 1;
}

static int test_hard_sigmoid(void) {
    float data[] = {-4.0f, -3.0f, 0.0f, 3.0f, 4.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_hard_sigmoid(t);
    tensor_ensure_executed(out);
    float* d = out->data;
    // hard_sigmoid(x) = clamp((x + 3) / 6, 0, 1)
    // x=-4: (-4+3)/6 = -1/6 -> clamp to 0
    // x=-3: (-3+3)/6 = 0 -> 0
    // x=0: (0+3)/6 = 0.5 -> 0.5
    // x=3: (3+3)/6 = 1 -> 1
    // x=4: (4+3)/6 = 7/6 -> clamp to 1
    if (!APPROX(d[0], 0.0f)) return 0;
    if (!APPROX(d[1], 0.0f)) return 0;
    if (!APPROX(d[2], 0.5f)) return 0;
    if (!APPROX(d[3], 1.0f)) return 0;
    if (!APPROX(d[4], 1.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_hard_tanh(void) {
    float data[] = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_hard_tanh(t);
    tensor_ensure_executed(out);
    float* d = out->data;
    // hard_tanh(x) = clamp(x, -1, 1)
    if (!APPROX(d[0], -1.0f)) return 0;
    if (!APPROX(d[1], -1.0f)) return 0;
    if (!APPROX(d[2], 0.0f)) return 0;
    if (!APPROX(d[3], 0.5f)) return 0;
    if (!APPROX(d[4], 1.0f)) return 0;
    if (!APPROX(d[5], 1.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_quick_gelu(void) {
    float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_quick_gelu(t);
    tensor_ensure_executed(out);
    float* d = out->data;
    // quick_gelu(x) = x * sigmoid(1.702 * x)
    for (int i = 0; i < 4; i++) {
        float x = data[i];
        float expected = x * (1.0f / (1.0f + expf(-1.702f * x)));
        if (!APPROX(d[i], expected)) return 0;
    }
    tensor_free(t);
    tensor_free(out);
    return 1;
}

int main(void) {
    printf("Missing Ops Tests\n");

    TEST(sort_ascending);
    TEST(sort_descending);
    TEST(topk);
    TEST(any_all);
    TEST(hard_sigmoid);
    TEST(hard_tanh);
    TEST(quick_gelu);

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
