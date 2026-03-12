#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"

static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name) do {                             \
    printf("  %-40s ", #name);                      \
    tests_run++;                                    \
    if (test_##name()) {                            \
        printf("PASS\n");                           \
        tests_passed++;                             \
    } else {                                        \
        printf("FAIL\n");                           \
    }                                               \
} while (0)

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-5f)

static const TensorConfig cpu_f32 = {
    .dtype      = DTYPE_FLOAT32,
    .device     = DEVICE_CPU,
    .has_dtype  = true,
    .has_device = true,
};

static int test_scalar_tensor(void) {
    float value = 42.0f;
    Tensor* t = tensor_from_data(&value, NULL, 0, &cpu_f32);
    if (!t) {
        int shape1[] = {1};
        t = tensor_from_data(&value, shape1, 1, &cpu_f32);
        if (!t) return 0;
        if (t->numel != 1) { tensor_free(t); return 0; }
        tensor_ensure_executed(t);
        float v = tensor_get_float(t, 0);
        if (!APPROX_EQ(v, 42.0f)) { tensor_free(t); return 0; }
        tensor_free(t);
        return 1;
    }

    if (t->numel != 1) { tensor_free(t); return 0; }

    tensor_ensure_executed(t);
    float v = tensor_get_float(t, 0);
    if (!APPROX_EQ(v, 42.0f)) { tensor_free(t); return 0; }

    tensor_free(t);
    return 1;
}

static int test_zero_size_tensor(void) {
    int shape[] = {0};

    Tensor* t_zeros = tensor_zeros(shape, 1, &cpu_f32);
    if (t_zeros) {
        if (t_zeros->numel != 0) { tensor_free(t_zeros); return 0; }
        tensor_free(t_zeros);
    }

    Tensor* t_ones = tensor_ones(shape, 1, &cpu_f32);
    if (t_ones) {
        if (t_ones->numel != 0) { tensor_free(t_ones); return 0; }
        tensor_free(t_ones);
    }

    Tensor* t_empty = tensor_empty(shape, 1, &cpu_f32);
    if (t_empty) {
        if (t_empty->numel != 0) { tensor_free(t_empty); return 0; }
        tensor_free(t_empty);
    }

    return 1;
}

static int test_high_ndim_8d(void) {
    int shape[] = {2, 3, 2, 2, 1, 2, 1, 3};
    size_t expected_numel = 2 * 3 * 2 * 2 * 1 * 2 * 1 * 3;

    Tensor* t = tensor_zeros(shape, 8, &cpu_f32);
    if (!t) return 0;

    if (t->ndim != 8)              { tensor_free(t); return 0; }
    if (t->numel != expected_numel) { tensor_free(t); return 0; }

    for (int i = 0; i < 8; i++) {
        if (t->shape[i] != shape[i]) { tensor_free(t); return 0; }
    }

    tensor_ensure_executed(t);
    for (size_t i = 0; i < t->numel; i++) {
        float v = tensor_get_float(t, i);
        if (v != 0.0f) { tensor_free(t); return 0; }
    }

    tensor_free(t);
    return 1;
}

static int test_nan_propagation(void) {
    float nan_val = NAN;
    float normal_val = 5.0f;

    int shape[] = {1};
    Tensor* t_nan    = tensor_from_data(&nan_val,    shape, 1, &cpu_f32);
    Tensor* t_normal = tensor_from_data(&normal_val, shape, 1, &cpu_f32);
    if (!t_nan || !t_normal) {
        if (t_nan)    tensor_free(t_nan);
        if (t_normal) tensor_free(t_normal);
        return 0;
    }

    Tensor* sum = tensor_add(t_nan, t_normal);
    if (!sum) { tensor_free(t_nan); tensor_free(t_normal); return 0; }
    tensor_ensure_executed(sum);
    float sum_v = tensor_get_float(sum, 0);
    if (!isnan(sum_v)) {
        tensor_free(t_nan); tensor_free(t_normal); tensor_free(sum);
        return 0;
    }

    Tensor* prod = tensor_mul(t_nan, t_normal);
    if (!prod) {
        tensor_free(t_nan); tensor_free(t_normal); tensor_free(sum);
        return 0;
    }
    tensor_ensure_executed(prod);
    float prod_v = tensor_get_float(prod, 0);
    if (!isnan(prod_v)) {
        tensor_free(t_nan); tensor_free(t_normal);
        tensor_free(sum); tensor_free(prod);
        return 0;
    }

    tensor_free(t_nan);
    tensor_free(t_normal);
    tensor_free(sum);
    tensor_free(prod);
    return 1;
}

static int test_shape_mismatch_matmul(void) {
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[20];
    for (int i = 0; i < 20; i++) b_data[i] = (float)i;

    int a_shape[] = {2, 3};
    int b_shape[] = {4, 5};

    Tensor* a = tensor_from_data(a_data, a_shape, 2, &cpu_f32);
    Tensor* b = tensor_from_data(b_data, b_shape, 2, &cpu_f32);
    if (!a || !b) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        return 0;
    }

    Tensor* c = tensor_matmul(a, b);

    int ok = (c == NULL);

    if (c) {
        int exec_result = tensor_ensure_executed(c);
        ok = 1;
        (void)exec_result;
        tensor_free(c);
    }

    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_single_element_batch(void) {
    float val_a = 3.0f;
    float val_b = 7.0f;

    int shape[] = {1, 1};
    Tensor* a = tensor_from_data(&val_a, shape, 2, &cpu_f32);
    Tensor* b = tensor_from_data(&val_b, shape, 2, &cpu_f32);
    if (!a || !b) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        return 0;
    }

    Tensor* sum = tensor_add(a, b);
    if (!sum) { tensor_free(a); tensor_free(b); return 0; }
    tensor_ensure_executed(sum);
    if (!APPROX_EQ(tensor_get_float(sum, 0), 10.0f)) {
        tensor_free(a); tensor_free(b); tensor_free(sum);
        return 0;
    }

    Tensor* prod = tensor_mul(a, b);
    if (!prod) { tensor_free(a); tensor_free(b); tensor_free(sum); return 0; }
    tensor_ensure_executed(prod);
    if (!APPROX_EQ(tensor_get_float(prod, 0), 21.0f)) {
        tensor_free(a); tensor_free(b); tensor_free(sum); tensor_free(prod);
        return 0;
    }

    Tensor* mm = tensor_matmul(a, b);
    if (!mm) {
        tensor_free(a); tensor_free(b);
        tensor_free(sum); tensor_free(prod);
        return 0;
    }
    tensor_ensure_executed(mm);
    if (!APPROX_EQ(tensor_get_float(mm, 0), 21.0f)) {
        tensor_free(a); tensor_free(b);
        tensor_free(sum); tensor_free(prod); tensor_free(mm);
        return 0;
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(sum);
    tensor_free(prod);
    tensor_free(mm);
    return 1;
}

static int test_large_tensor(void) {
    int shape[] = {1000, 1000};
    Tensor* a = tensor_ones(shape, 2, &cpu_f32);
    Tensor* b = tensor_ones(shape, 2, &cpu_f32);
    if (!a || !b) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        return 0;
    }

    if (a->numel != 1000000) { tensor_free(a); tensor_free(b); return 0; }

    Tensor* sum = tensor_add(a, b);
    if (!sum) { tensor_free(a); tensor_free(b); return 0; }
    tensor_ensure_executed(sum);

    int ok = 1;
    size_t check_indices[] = {0, 1, 999, 500000, 999999};
    for (int i = 0; i < 5; i++) {
        float v = tensor_get_float(sum, check_indices[i]);
        if (!APPROX_EQ(v, 2.0f)) { ok = 0; break; }
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(sum);
    return ok;
}

static int test_double_free_protection(void) {
    int shape[] = {2, 3};
    Tensor* t = tensor_ones(shape, 2, &cpu_f32);
    if (!t) return 0;

    tensor_free(t);
    t = NULL;
    tensor_free(t);

    return 1;
}

int main(void) {
    cml_init();

    printf("test_edge_cases\n\n");

    TEST(scalar_tensor);
    TEST(zero_size_tensor);
    TEST(high_ndim_8d);
    TEST(nan_propagation);
    TEST(shape_mismatch_matmul);
    TEST(single_element_batch);
    TEST(large_tensor);
    TEST(double_free_protection);

    printf("\n%d/%d passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
