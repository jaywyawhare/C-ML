#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor/tensor.h"
#include "ops/uops.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static const TensorConfig cpu_i32 = {
    .dtype = DTYPE_INT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

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

static int test_where(void) {
    float cond_data[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float x_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float y_data[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    int shape[] = {4};

    Tensor* cond = tensor_from_data(cond_data, shape, 1, &cpu_f32);
    Tensor* x = tensor_from_data(x_data, shape, 1, &cpu_f32);
    Tensor* y = tensor_from_data(y_data, shape, 1, &cpu_f32);

    Tensor* result = tensor_where(cond, x, y);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    float* out = (float*)result->data;
    if (!APPROX(out[0], 10.0f)) return 0;
    if (!APPROX(out[1], -2.0f)) return 0;
    if (!APPROX(out[2], 30.0f)) return 0;
    if (!APPROX(out[3], -4.0f)) return 0;

    tensor_free(cond);
    tensor_free(x);
    tensor_free(y);
    tensor_free(result);
    return 1;
}

static int test_one_hot(void) {
    float idx_data[] = {0.0f, 2.0f, 1.0f};
    int shape[] = {3};
    Tensor* indices = tensor_from_data(idx_data, shape, 1, &cpu_f32);

    Tensor* result = tensor_one_hot(indices, 4);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->ndim != 2) return 0;
    if (result->shape[0] != 3 || result->shape[1] != 4) return 0;

    float* out = (float*)result->data;
    // [1,0,0,0], [0,0,1,0], [0,1,0,0]
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 0.0f)) return 0;
    if (!APPROX(out[4], 0.0f)) return 0;
    if (!APPROX(out[6], 1.0f)) return 0;
    if (!APPROX(out[9], 1.0f)) return 0;

    tensor_free(indices);
    tensor_free(result);
    return 1;
}

static int test_roll(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);

    Tensor* result = tensor_roll(t, 2, 0);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    float* out = (float*)result->data;
    // roll by 2: [4,5,1,2,3]
    if (!APPROX(out[0], 4.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    if (!APPROX(out[2], 1.0f)) return 0;
    if (!APPROX(out[3], 2.0f)) return 0;
    if (!APPROX(out[4], 3.0f)) return 0;

    tensor_free(t);
    tensor_free(result);
    return 1;
}

static int test_nonzero(void) {
    float data[] = {0.0f, 3.0f, 0.0f, 5.0f, 7.0f, 0.0f};
    int shape[] = {2, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);

    Tensor* result = tensor_nonzero(t);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->ndim != 2) return 0;
    if (result->shape[0] != 3) return 0;
    if (result->shape[1] != 2) return 0;

    float* out = (float*)result->data;
    // nonzero elements at (0,1), (1,0), (1,1)
    if (!APPROX(out[0], 0.0f)) return 0;
    if (!APPROX(out[1], 1.0f)) return 0;
    if (!APPROX(out[2], 1.0f)) return 0;
    if (!APPROX(out[3], 0.0f)) return 0;
    if (!APPROX(out[4], 1.0f)) return 0;
    if (!APPROX(out[5], 1.0f)) return 0;

    tensor_free(t);
    tensor_free(result);
    return 1;
}

static int test_copysign(void) {
    float a_data[] = {1.0f, -2.0f, 3.0f, -4.0f};
    float b_data[] = {-1.0f, 1.0f, -1.0f, 1.0f};
    int shape[] = {4};

    Tensor* a = tensor_from_data(a_data, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(b_data, shape, 1, &cpu_f32);

    Tensor* result = tensor_copysign(a, b);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    float* out = (float*)result->data;
    if (!APPROX(out[0], -1.0f)) return 0;
    if (!APPROX(out[1], 2.0f)) return 0;
    if (!APPROX(out[2], -3.0f)) return 0;
    if (!APPROX(out[3], 4.0f)) return 0;

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_logaddexp(void) {
    float a_data[] = {1.0f, 2.0f};
    float b_data[] = {3.0f, 4.0f};
    int shape[] = {2};

    Tensor* a = tensor_from_data(a_data, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(b_data, shape, 1, &cpu_f32);

    Tensor* result = tensor_logaddexp(a, b);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    float* out = (float*)result->data;
    float expected0 = logf(expf(1.0f) + expf(3.0f));
    float expected1 = logf(expf(2.0f) + expf(4.0f));
    if (!APPROX(out[0], expected0)) return 0;
    if (!APPROX(out[1], expected1)) return 0;

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_multinomial(void) {
    float probs_data[] = {0.0f, 0.0f, 1.0f, 0.0f};
    int shape[] = {4};
    Tensor* probs = tensor_from_data(probs_data, shape, 1, &cpu_f32);

    Tensor* result = tensor_multinomial(probs, 3, true);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->shape[0] != 3) return 0;

    int32_t* out = (int32_t*)result->data;
    for (int i = 0; i < 3; i++) {
        if (out[i] != 2) return 0;
    }

    tensor_free(probs);
    tensor_free(result);
    return 1;
}

static int test_einsum_matmul(void) {
    // 2x3 @ 3x2 = 2x2
    float a_data[] = {1,2,3, 4,5,6};
    float b_data[] = {7,8, 9,10, 11,12};
    int a_shape[] = {2, 3};
    int b_shape[] = {3, 2};

    Tensor* a = tensor_from_data(a_data, a_shape, 2, &cpu_f32);
    Tensor* b = tensor_from_data(b_data, b_shape, 2, &cpu_f32);

    Tensor* tensors[] = {a, b};
    Tensor* result = tensor_einsum("ij,jk->ik", tensors, 2);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->ndim != 2 || result->shape[0] != 2 || result->shape[1] != 2) return 0;

    float* out = (float*)result->data;
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    if (!APPROX(out[0], 58.0f)) return 0;
    if (!APPROX(out[1], 64.0f)) return 0;
    if (!APPROX(out[2], 139.0f)) return 0;
    if (!APPROX(out[3], 154.0f)) return 0;

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_einsum_trace(void) {
    // trace: "ii->"
    float data[] = {1,2, 3,4};
    int shape[] = {2, 2};
    Tensor* a = tensor_from_data(data, shape, 2, &cpu_f32);

    Tensor* tensors[] = {a};
    Tensor* result = tensor_einsum("ii->", tensors, 1);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    float* out = (float*)result->data;
    if (!APPROX(out[0], 5.0f)) return 0;

    tensor_free(a);
    tensor_free(result);
    return 1;
}

static int test_einsum_diagonal(void) {
    // diagonal: "ii->i"
    float data[] = {1,2,3, 4,5,6, 7,8,9};
    int shape[] = {3, 3};
    Tensor* a = tensor_from_data(data, shape, 2, &cpu_f32);

    Tensor* tensors[] = {a};
    Tensor* result = tensor_einsum("ii->i", tensors, 1);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->ndim != 1 || result->shape[0] != 3) return 0;

    float* out = (float*)result->data;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    if (!APPROX(out[2], 9.0f)) return 0;

    tensor_free(a);
    tensor_free(result);
    return 1;
}

static int test_einsum_outer(void) {
    // outer product: "i,j->ij"
    float a_data[] = {1, 2, 3};
    float b_data[] = {4, 5};
    int a_shape[] = {3};
    int b_shape[] = {2};

    Tensor* a = tensor_from_data(a_data, a_shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(b_data, b_shape, 1, &cpu_f32);

    Tensor* tensors[] = {a, b};
    Tensor* result = tensor_einsum("i,j->ij", tensors, 2);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->ndim != 2 || result->shape[0] != 3 || result->shape[1] != 2) return 0;

    float* out = (float*)result->data;
    if (!APPROX(out[0], 4.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    if (!APPROX(out[2], 8.0f)) return 0;
    if (!APPROX(out[3], 10.0f)) return 0;
    if (!APPROX(out[4], 12.0f)) return 0;
    if (!APPROX(out[5], 15.0f)) return 0;

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_einsum_sum_all(void) {
    // sum all: "ij->"
    float data[] = {1, 2, 3, 4, 5, 6};
    int shape[] = {2, 3};
    Tensor* a = tensor_from_data(data, shape, 2, &cpu_f32);

    Tensor* tensors[] = {a};
    Tensor* result = tensor_einsum("ij->", tensors, 1);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    float* out = (float*)result->data;
    if (!APPROX(out[0], 21.0f)) return 0;

    tensor_free(a);
    tensor_free(result);
    return 1;
}

static int test_einsum_transpose(void) {
    // transpose: "ij->ji"
    float data[] = {1, 2, 3, 4, 5, 6};
    int shape[] = {2, 3};
    Tensor* a = tensor_from_data(data, shape, 2, &cpu_f32);

    Tensor* tensors[] = {a};
    Tensor* result = tensor_einsum("ij->ji", tensors, 1);
    if (!result) return 0;
    tensor_ensure_executed(result);
    if (!result->data) return 0;

    if (result->ndim != 2 || result->shape[0] != 3 || result->shape[1] != 2) return 0;

    float* out = (float*)result->data;
    // original: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 4.0f)) return 0;
    if (!APPROX(out[2], 2.0f)) return 0;
    if (!APPROX(out[3], 5.0f)) return 0;
    if (!APPROX(out[4], 3.0f)) return 0;
    if (!APPROX(out[5], 6.0f)) return 0;

    tensor_free(a);
    tensor_free(result);
    return 1;
}

int main(void) {
    printf("=== Tensor Ops Extra Tests ===\n");

    TEST(where);
    TEST(one_hot);
    TEST(roll);
    TEST(nonzero);
    TEST(copysign);
    TEST(logaddexp);
    TEST(multinomial);
    TEST(einsum_matmul);
    TEST(einsum_trace);
    TEST(einsum_diagonal);
    TEST(einsum_outer);
    TEST(einsum_sum_all);
    TEST(einsum_transpose);

    printf("\n%d / %d tests passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
