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

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-5f)

static int test_empty_tensor(void) {
    int shape[] = {2, 3};
    Tensor* t = cml_empty(shape, 2, NULL);
    if (!t) return 0;
    if (t->ndim != 2 || t->shape[0] != 2 || t->shape[1] != 3) return 0;
    if (t->numel != 6) return 0;
    tensor_free(t);
    return 1;
}

static int test_zeros_tensor(void) {
    int shape[] = {3, 4};
    Tensor* t = cml_zeros(shape, 2, NULL);
    if (!t) return 0;
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        if (data[i] != 0.0f) { tensor_free(t); return 0; }
    }
    tensor_free(t);
    return 1;
}

static int test_ones_tensor(void) {
    int shape[] = {2, 2};
    Tensor* t = cml_ones(shape, 2, NULL);
    if (!t) return 0;
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        if (data[i] != 1.0f) { tensor_free(t); return 0; }
    }
    tensor_free(t);
    return 1;
}

static int test_from_data(void) {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* t = cml_tensor_2d(values, 2, 2);
    if (!t) return 0;
    float* data = (float*)t->data;
    for (int i = 0; i < 4; i++) {
        if (!APPROX_EQ(data[i], values[i])) { tensor_free(t); return 0; }
    }
    tensor_free(t);
    return 1;
}

static int test_1d_creation(void) {
    Tensor* z = cml_zeros_1d(5);
    Tensor* o = cml_ones_1d(5);
    if (!z || !o) return 0;
    if (z->numel != 5 || o->numel != 5) return 0;
    float* zd = (float*)z->data;
    float* od = (float*)o->data;
    if (zd[0] != 0.0f || od[0] != 1.0f) return 0;
    tensor_free(z);
    tensor_free(o);
    return 1;
}

static int test_2d_creation(void) {
    Tensor* z = cml_zeros_2d(3, 4);
    Tensor* o = cml_ones_2d(3, 4);
    if (!z || !o) return 0;
    if (z->numel != 12 || o->numel != 12) return 0;
    tensor_free(z);
    tensor_free(o);
    return 1;
}

static int test_add(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor* a = cml_tensor_2d(a_data, 2, 2);
    Tensor* b = cml_tensor_2d(b_data, 2, 2);
    if (!a || !b) return 0;
    Tensor* c = cml_add(a, b);
    if (!c) return 0;
    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(a); tensor_free(b); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = APPROX_EQ(cd[0], 6.0f) && APPROX_EQ(cd[1], 8.0f) &&
             APPROX_EQ(cd[2], 10.0f) && APPROX_EQ(cd[3], 12.0f);
    tensor_free(a); tensor_free(b); tensor_free(c);
    return ok;
}

static int test_sub(void) {
    float a_data[] = {5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f};
    Tensor* a = cml_tensor_1d(a_data, 2);
    Tensor* b = cml_tensor_1d(b_data, 2);
    if (!a || !b) return 0;
    Tensor* c = cml_sub(a, b);
    if (!c) return 0;
    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(a); tensor_free(b); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = APPROX_EQ(cd[0], 4.0f) && APPROX_EQ(cd[1], 4.0f);
    tensor_free(a); tensor_free(b); tensor_free(c);
    return ok;
}

static int test_mul(void) {
    float a_data[] = {2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f};
    Tensor* a = cml_tensor_1d(a_data, 2);
    Tensor* b = cml_tensor_1d(b_data, 2);
    if (!a || !b) return 0;
    Tensor* c = cml_mul(a, b);
    if (!c) return 0;
    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(a); tensor_free(b); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = APPROX_EQ(cd[0], 8.0f) && APPROX_EQ(cd[1], 15.0f);
    tensor_free(a); tensor_free(b); tensor_free(c);
    return ok;
}

static int test_matmul(void) {
    /* [1, 2; 3, 4] @ [5, 6; 7, 8] = [19, 22; 43, 50] */
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor* a = cml_tensor_2d(a_data, 2, 2);
    Tensor* b = cml_tensor_2d(b_data, 2, 2);
    if (!a || !b) return 0;
    Tensor* c = cml_matmul(a, b);
    if (!c) return 0;
    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(a); tensor_free(b); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = APPROX_EQ(cd[0], 19.0f) && APPROX_EQ(cd[1], 22.0f) &&
             APPROX_EQ(cd[2], 43.0f) && APPROX_EQ(cd[3], 50.0f);
    tensor_free(a); tensor_free(b); tensor_free(c);
    return ok;
}

static int test_reshape(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor* t = cml_tensor_1d(data, 6);
    if (!t) return 0;
    int new_shape[] = {2, 3};
    Tensor* r = cml_reshape(t, new_shape, 2);
    if (!r) { tensor_free(t); return 0; }
    int ok = (r->ndim == 2 && r->shape[0] == 2 && r->shape[1] == 3 && r->numel == 6);
    tensor_free(t);
    if (r != t) tensor_free(r);
    return ok;
}

static int test_transpose(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor* t = cml_tensor_2d(data, 2, 3);
    if (!t) return 0;
    Tensor* tr = cml_transpose(t, 0, 1);
    if (!tr) { tensor_free(t); return 0; }
    int ok = (tr->shape[0] == 3 && tr->shape[1] == 2);
    tensor_free(t);
    if (tr != t) tensor_free(tr);
    return ok;
}

static int test_clone(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = cml_tensor_1d(data, 3);
    if (!t) return 0;
    Tensor* c = cml_clone(t);
    if (!c) { tensor_free(t); return 0; }
    tensor_ensure_executed(c);
    if (!c->data) { tensor_free(t); tensor_free(c); return 0; }
    float* cd = (float*)c->data;
    int ok = (c != t) && APPROX_EQ(cd[0], 1.0f) && APPROX_EQ(cd[1], 2.0f) &&
             APPROX_EQ(cd[2], 3.0f);
    tensor_free(t); tensor_free(c);
    return ok;
}

static int test_null_safety(void) {
    /* These should return NULL and not crash */
    Tensor* r = cml_add(NULL, NULL);
    if (r != NULL) return 0;
    r = cml_matmul(NULL, NULL);
    if (r != NULL) return 0;
    r = cml_clone(NULL);
    if (r != NULL) return 0;
    return 1;
}

int main(void) {
    cml_init();

    printf("\n=== Tensor Unit Tests ===\n\n");

    printf("Creation:\n");
    TEST(empty_tensor);
    TEST(zeros_tensor);
    TEST(ones_tensor);
    TEST(from_data);
    TEST(1d_creation);
    TEST(2d_creation);

    printf("\nBasic Ops:\n");
    TEST(add);
    TEST(sub);
    TEST(mul);
    TEST(matmul);

    printf("\nShape Manipulation:\n");
    TEST(reshape);
    TEST(transpose);
    TEST(clone);

    printf("\nEdge Cases:\n");
    TEST(null_safety);

    printf("\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
