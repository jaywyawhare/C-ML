#include <stdio.h>
#include <stdlib.h>
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

#define APPROX_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))

static int test_dtype_size(void) {
    if (cml_dtype_size(DTYPE_FLOAT8_E4M3_FNUZ) != 1) return 0;
    if (cml_dtype_size(DTYPE_FLOAT8_E5M2_FNUZ) != 1) return 0;
    return 1;
}

static int test_e4m3fnuz_roundtrip(void) {
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -0.25f, 100.0f};
    int n = sizeof(values) / sizeof(values[0]);

    int shape[] = {n};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT8_E4M3_FNUZ, .has_dtype = true};
    Tensor* t = tensor_empty(shape, 1, &cfg);
    if (!t) return 0;

    for (int i = 0; i < n; i++)
        tensor_set_float(t, i, values[i]);

    for (int i = 0; i < n; i++) {
        float got = tensor_get_float(t, i);
        if (values[i] == 0.0f) {
            if (got != 0.0f) { tensor_free(t); return 0; }
        } else {
            float rel_err = fabsf(got - values[i]) / fabsf(values[i]);
            if (rel_err > 0.3f) { tensor_free(t); return 0; }
        }
    }

    tensor_free(t);
    return 1;
}

static int test_e5m2fnuz_roundtrip(void) {
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 4.0f, -8.0f};
    int n = sizeof(values) / sizeof(values[0]);

    int shape[] = {n};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT8_E5M2_FNUZ, .has_dtype = true};
    Tensor* t = tensor_empty(shape, 1, &cfg);
    if (!t) return 0;

    for (int i = 0; i < n; i++)
        tensor_set_float(t, i, values[i]);

    for (int i = 0; i < n; i++) {
        float got = tensor_get_float(t, i);
        if (values[i] == 0.0f) {
            if (got != 0.0f) { tensor_free(t); return 0; }
        } else {
            float rel_err = fabsf(got - values[i]) / fabsf(values[i]);
            if (rel_err > 0.5f) { tensor_free(t); return 0; }
        }
    }

    tensor_free(t);
    return 1;
}

static int test_fnuz_no_negative_zero(void) {
    int shape[] = {1};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT8_E4M3_FNUZ, .has_dtype = true};
    Tensor* t = tensor_empty(shape, 1, &cfg);
    if (!t) return 0;

    tensor_set_float(t, 0, -0.0f);
    float got = tensor_get_float(t, 0);
    if (got != 0.0f) { tensor_free(t); return 0; }

    tensor_free(t);
    return 1;
}

static int test_fnuz_0x80_is_nan(void) {
    int shape[] = {1};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT8_E4M3_FNUZ, .has_dtype = true};
    Tensor* t = tensor_empty(shape, 1, &cfg);
    if (!t) return 0;

    ((uint8_t*)t->data)[0] = 0x80;
    float got = tensor_get_float(t, 0);
    if (!isnan(got)) { tensor_free(t); return 0; }

    tensor_free(t);
    return 1;
}

static int test_cast_to_fnuz(void) {
    float data[] = {1.0f, 2.0f, -3.0f, 0.0f};
    int shape[] = {4};
    Tensor* src = tensor_from_data(data, shape, 1, NULL);
    if (!src) return 0;

    Tensor* e4 = tensor_fp8e4m3fnuz(src);
    Tensor* e5 = tensor_fp8e5m2fnuz(src);
    if (!e4 || !e5) {
        tensor_free(src); tensor_free(e4); tensor_free(e5);
        return 0;
    }

    if (e4->dtype != DTYPE_FLOAT8_E4M3_FNUZ) { tensor_free(src); tensor_free(e4); tensor_free(e5); return 0; }
    if (e5->dtype != DTYPE_FLOAT8_E5M2_FNUZ) { tensor_free(src); tensor_free(e4); tensor_free(e5); return 0; }

    float v4 = tensor_get_float(e4, 0);
    float v5 = tensor_get_float(e5, 0);
    if (!APPROX_EQ(v4, 1.0f, 0.2f)) { tensor_free(src); tensor_free(e4); tensor_free(e5); return 0; }
    if (!APPROX_EQ(v5, 1.0f, 0.5f)) { tensor_free(src); tensor_free(e4); tensor_free(e5); return 0; }

    tensor_free(src);
    tensor_free(e4);
    tensor_free(e5);
    return 1;
}

static int test_ones_fnuz(void) {
    int shape[] = {4};
    TensorConfig cfg4 = {.dtype = DTYPE_FLOAT8_E4M3_FNUZ, .has_dtype = true};
    TensorConfig cfg5 = {.dtype = DTYPE_FLOAT8_E5M2_FNUZ, .has_dtype = true};

    Tensor* t4 = tensor_ones(shape, 1, &cfg4);
    Tensor* t5 = tensor_ones(shape, 1, &cfg5);
    if (!t4 || !t5) { tensor_free(t4); tensor_free(t5); return 0; }

    for (int i = 0; i < 4; i++) {
        float v4 = tensor_get_float(t4, i);
        float v5 = tensor_get_float(t5, i);
        if (!APPROX_EQ(v4, 1.0f, 0.01f)) { tensor_free(t4); tensor_free(t5); return 0; }
        if (!APPROX_EQ(v5, 1.0f, 0.01f)) { tensor_free(t4); tensor_free(t5); return 0; }
    }

    tensor_free(t4);
    tensor_free(t5);
    return 1;
}

static int test_promote_fnuz(void) {
    DType r1 = cml_promote_dtype(DTYPE_FLOAT8_E4M3_FNUZ, DTYPE_FLOAT32);
    if (r1 != DTYPE_FLOAT32) return 0;

    DType r2 = cml_promote_dtype(DTYPE_FLOAT8_E5M2_FNUZ, DTYPE_INT32);
    if (r2 != DTYPE_FLOAT8_E5M2_FNUZ) return 0;

    DType r3 = cml_promote_dtype(DTYPE_FLOAT8_E4M3_FNUZ, DTYPE_FLOAT8_E5M2_FNUZ);
    if (r3 != DTYPE_FLOAT8_E4M3_FNUZ && r3 != DTYPE_FLOAT8_E5M2_FNUZ) return 0;

    return 1;
}

int main(void) {
    printf("=== FP8 FNUZ Tests ===\n");
    TEST(dtype_size);
    TEST(e4m3fnuz_roundtrip);
    TEST(e5m2fnuz_roundtrip);
    TEST(fnuz_no_negative_zero);
    TEST(fnuz_0x80_is_nan);
    TEST(cast_to_fnuz);
    TEST(ones_fnuz);
    TEST(promote_fnuz);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
