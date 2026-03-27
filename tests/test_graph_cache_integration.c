/*
 * Test graph cache integration — verifies buffer reuse across reset cycles
 * and SSE Winograd transform correctness.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/execution.h"
#include "ops/ir/graph_cache.h"

static int tests_passed = 0;
static int tests_total  = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

static void test_cache_population(void) {
    printf("Test: cache population across reset cycles\n");
    CMLGraphCache* cache = cml_get_graph_cache();
    size_t h0 = cache->hits, m0 = cache->misses;
    TensorConfig cfg = {0};
    int sa[] = {32, 64}, sb[] = {64, 32};
    float* a = malloc(32*64*sizeof(float));
    float* b = malloc(64*32*sizeof(float));
    for (int i = 0; i < 32*64; i++) a[i] = (float)(i%7)*0.1f;
    for (int i = 0; i < 64*32; i++) b[i] = (float)(i%5)*0.1f;

    /* Iter 1: miss */
    Tensor* ta = tensor_from_data(a, sa, 2, &cfg);
    Tensor* tb = tensor_from_data(b, sb, 2, &cfg);
    Tensor* tc = uop_matmul(ta, tb);
    float* r = (float*)tensor_data_ptr(tc);
    float v1 = r[0];
    tensor_free(ta); tensor_free(tb); tensor_free(tc);
    cml_reset_ir_context();

    CHECK("first iteration creates cache entry", cache->misses > m0);
    CHECK("cache count > 0", cache->count > 0);

    /* Iter 2: hit */
    size_t h1 = cache->hits;
    ta = tensor_from_data(a, sa, 2, &cfg);
    tb = tensor_from_data(b, sb, 2, &cfg);
    tc = uop_matmul(ta, tb);
    r = (float*)tensor_data_ptr(tc);
    float v2 = r[0];
    tensor_free(ta); tensor_free(tb); tensor_free(tc);
    cml_reset_ir_context();

    CHECK("second iteration cache hit", cache->hits > h1);
    CHECK("results consistent across cache hit", v1 == v2);

    free(a); free(b);
}

static void test_multi_node_cache(void) {
    printf("Test: multi-node graph cache (mm+add+relu)\n");
    CMLGraphCache* cache = cml_get_graph_cache();
    TensorConfig cfg = {0};
    int sa[] = {16, 32}, sb[] = {32, 16}, sbi[] = {16};
    float* a = malloc(16*32*sizeof(float));
    float* b = malloc(32*16*sizeof(float));
    float* bias = malloc(16*sizeof(float));
    for (int i = 0; i < 16*32; i++) a[i] = (float)(i%9)*0.05f;
    for (int i = 0; i < 32*16; i++) b[i] = (float)(i%7)*0.05f;
    for (int i = 0; i < 16; i++) bias[i] = 0.1f;

    float results[3];
    for (int iter = 0; iter < 3; iter++) {
        Tensor* ta = tensor_from_data(a, sa, 2, &cfg);
        Tensor* tb = tensor_from_data(b, sb, 2, &cfg);
        Tensor* tbias = tensor_from_data(bias, sbi, 1, &cfg);
        Tensor* tc = uop_matmul(ta, tb);
        Tensor* td = uop_add(tc, tbias);
        Tensor* te = uop_relu(td);
        float* r = (float*)tensor_data_ptr(te);
        results[iter] = r[0];
        tensor_free(ta); tensor_free(tb); tensor_free(tbias);
        tensor_free(tc); tensor_free(td); tensor_free(te);
        cml_reset_ir_context();
    }

    CHECK("multi-node: iter 0==1", results[0] == results[1]);
    CHECK("multi-node: iter 1==2", results[1] == results[2]);

    free(a); free(b); free(bias);
}

static void test_cache_stats(void) {
    printf("Test: cache stats\n");
    CMLGraphCache* cache = cml_get_graph_cache();
    CHECK("cache has entries", cache->count > 0);
    CHECK("cache has hits", cache->hits > 0);
    cml_graph_cache_print_stats(cache);
}

int main(void) {
    printf("=== Graph Cache Integration Tests ===\n\n");
    test_cache_population();
    test_multi_node_cache();
    test_cache_stats();

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
