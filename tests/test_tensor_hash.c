#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cml.h"
#include "tensor/tensor.h"
#include "ops/ir/ir.h"
#include "ops/ir/context.h"

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

static int test_hash_basic(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {2, 2};
    Tensor* t = tensor_from_data(data, shape, 2, NULL);
    if (!t) { cml_ir_free(ir); return 0; }

    uint64_t h = tensor_hash(t);
    int ok = (h != 0);

    tensor_free(t);
    cml_ir_free(ir);
    return ok;
}

static int test_hash_deterministic(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};

    Tensor* t1 = tensor_from_data(data, shape, 1, NULL);
    Tensor* t2 = tensor_from_data(data, shape, 1, NULL);
    if (!t1 || !t2) {
        if (t1) tensor_free(t1);
        if (t2) tensor_free(t2);
        cml_ir_free(ir);
        return 0;
    }

    uint64_t h1 = tensor_hash(t1);
    uint64_t h2 = tensor_hash(t2);

    int ok = (h1 == h2);
    tensor_free(t1);
    tensor_free(t2);
    cml_ir_free(ir);
    return ok;
}

static int test_hash_different_data(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data1[] = {1.0f, 2.0f};
    float data2[] = {3.0f, 4.0f};
    int shape[] = {2};

    Tensor* t1 = tensor_from_data(data1, shape, 1, NULL);
    Tensor* t2 = tensor_from_data(data2, shape, 1, NULL);
    if (!t1 || !t2) {
        if (t1) tensor_free(t1);
        if (t2) tensor_free(t2);
        cml_ir_free(ir);
        return 0;
    }

    uint64_t h1 = tensor_hash(t1);
    uint64_t h2 = tensor_hash(t2);

    int ok = (h1 != h2);
    tensor_free(t1);
    tensor_free(t2);
    cml_ir_free(ir);
    return ok;
}

static int test_hash_null(void) {
    return tensor_hash(NULL) == 0;
}

static int test_keccak_basic(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, NULL);
    if (!t) { cml_ir_free(ir); return 0; }

    uint8_t hash[32];
    int rc = tensor_keccak(t, hash, 32);
    if (rc != 0) {
        tensor_free(t);
        cml_ir_free(ir);
        return 0;
    }

    int all_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (hash[i] != 0) { all_zero = 0; break; }
    }

    tensor_free(t);
    cml_ir_free(ir);
    return !all_zero;
}

static int test_keccak_deterministic(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data[] = {5.0f, 6.0f, 7.0f};
    int shape[] = {3};

    Tensor* t1 = tensor_from_data(data, shape, 1, NULL);
    Tensor* t2 = tensor_from_data(data, shape, 1, NULL);
    if (!t1 || !t2) {
        if (t1) tensor_free(t1);
        if (t2) tensor_free(t2);
        cml_ir_free(ir);
        return 0;
    }

    uint8_t h1[32], h2[32];
    tensor_keccak(t1, h1, 32);
    tensor_keccak(t2, h2, 32);

    int ok = (memcmp(h1, h2, 32) == 0);
    tensor_free(t1);
    tensor_free(t2);
    cml_ir_free(ir);
    return ok;
}

static int test_keccak_different_data(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data1[] = {1.0f};
    float data2[] = {2.0f};
    int shape[] = {1};

    Tensor* t1 = tensor_from_data(data1, shape, 1, NULL);
    Tensor* t2 = tensor_from_data(data2, shape, 1, NULL);
    if (!t1 || !t2) {
        if (t1) tensor_free(t1);
        if (t2) tensor_free(t2);
        cml_ir_free(ir);
        return 0;
    }

    uint8_t h1[32], h2[32];
    tensor_keccak(t1, h1, 32);
    tensor_keccak(t2, h2, 32);

    int ok = (memcmp(h1, h2, 32) != 0);
    tensor_free(t1);
    tensor_free(t2);
    cml_ir_free(ir);
    return ok;
}

static int test_keccak_short_output(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data[] = {42.0f};
    int shape[] = {1};
    Tensor* t = tensor_from_data(data, shape, 1, NULL);
    if (!t) { cml_ir_free(ir); return 0; }

    uint8_t full[32], partial[16];
    tensor_keccak(t, full, 32);
    tensor_keccak(t, partial, 16);

    int ok = (memcmp(full, partial, 16) == 0);
    tensor_free(t);
    cml_ir_free(ir);
    return ok;
}

static int test_keccak_null(void) {
    uint8_t out[32];
    return tensor_keccak(NULL, out, 32) != 0;
}

static int test_keccak_large_tensor(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    int n = 1024;
    float* data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)i;
    int shape[] = {n};

    Tensor* t = tensor_from_data(data, shape, 1, NULL);
    free(data);
    if (!t) { cml_ir_free(ir); return 0; }

    uint8_t hash[32];
    int rc = tensor_keccak(t, hash, 32);

    tensor_free(t);
    cml_ir_free(ir);
    return rc == 0;
}

static int test_keccak_known_empty(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;
    cml_ir_set_global_context(ir);

    float data[] = {0.0f};
    int shape[] = {1};
    Tensor* t = tensor_from_data(data, shape, 1, NULL);
    if (!t) { cml_ir_free(ir); return 0; }

    uint8_t hash[32];
    int rc = tensor_keccak(t, hash, 32);

    tensor_free(t);
    cml_ir_free(ir);
    return rc == 0;
}

int main(void) {
    printf("=== Tensor Hash Tests ===\n");

    TEST(hash_basic);
    TEST(hash_deterministic);
    TEST(hash_different_data);
    TEST(hash_null);
    TEST(keccak_basic);
    TEST(keccak_deterministic);
    TEST(keccak_different_data);
    TEST(keccak_short_output);
    TEST(keccak_null);
    TEST(keccak_large_tensor);
    TEST(keccak_known_empty);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
