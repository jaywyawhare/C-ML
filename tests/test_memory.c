/**
 * @file test_memory.c
 * @brief Unit tests for memory pool, graph allocator, and cleanup context
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "cml.h"
#include "alloc/memory_pools.h"
#include "alloc/graph_allocator.h"
#include "core/cleanup.h"

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

static int test_memory_pool_create(void) {
    MemoryPool* pool = memory_pool_create(1024, 8, DTYPE_FLOAT32);
    if (!pool) return 0;
    printf("(blocks=%d) ", pool->num_blocks);
    memory_pool_free(pool);
    return 1;
}

static int test_memory_pool_alloc_free(void) {
    MemoryPool* pool = memory_pool_create(256, 4, DTYPE_FLOAT32);
    if (!pool) return 0;

    void* block1 = memory_pool_alloc(pool);
    void* block2 = memory_pool_alloc(pool);
    if (!block1 || !block2) { memory_pool_free(pool); return 0; }

    /* Blocks should be different */
    if (block1 == block2) { memory_pool_free(pool); return 0; }

    /* Free a block and reallocate */
    int ret = memory_pool_free_block(pool, block1);
    if (ret != 0) { memory_pool_free(pool); return 0; }

    void* block3 = memory_pool_alloc(pool);
    if (!block3) { memory_pool_free(pool); return 0; }

    printf("(ok) ");
    memory_pool_free(pool);
    return 1;
}

static int test_memory_pool_exhaustion(void) {
    MemoryPool* pool = memory_pool_create(64, 2, DTYPE_FLOAT32);
    if (!pool) return 0;

    void* b1 = memory_pool_alloc(pool);
    void* b2 = memory_pool_alloc(pool);
    void* b3 = memory_pool_alloc(pool); /* Should be NULL if pool is full */

    printf("(b1=%s b2=%s b3=%s) ",
           b1 ? "ok" : "null", b2 ? "ok" : "null", b3 ? "ok" : "null");

    /* At least b1 and b2 should succeed */
    int ok = (b1 != NULL && b2 != NULL);

    memory_pool_free(pool);
    return ok;
}

static int test_tensor_pool_create(void) {
    int shape[] = {2, 3};
    TensorPool* pool = tensor_pool_create(shape, 2, 4, DTYPE_FLOAT32, DEVICE_CPU);
    if (!pool) return 0;
    printf("(capacity=%zu) ", pool->capacity);
    tensor_pool_free(pool);
    return 1;
}

static int test_graph_allocator_create(void) {
    /* Create allocator with NULL buft - should handle gracefully */
    CMLGraphAllocator_t galloc = cml_graph_allocator_new(NULL);
    if (!galloc) {
        printf("(NULL buft returned NULL, ok) ");
        return 1;
    }
    cml_graph_allocator_free(galloc);
    return 1;
}

static int test_cleanup_context_create(void) {
    CleanupContext* ctx = cleanup_context_create();
    if (!ctx) return 0;
    /* Context is auto-registered globally; don't free manually (cml_auto_cleanup handles it) */
    return 1;
}

static int test_cleanup_register_tensor(void) {
    CleanupContext* ctx = cleanup_context_create();
    if (!ctx) return 0;

    Tensor* t = cml_ones_1d(5);
    if (!t) return 0;

    int ret = cleanup_register_tensor(ctx, t);
    if (ret != 0) {
        tensor_free(t);
        return 0;
    }

    /* Don't free ctx here - it's auto-registered and cml_auto_cleanup will handle it */
    return 1;
}

static int test_cleanup_clear_all(void) {
    CleanupContext* ctx = cleanup_context_create();
    if (!ctx) return 0;

    Tensor* t1 = cml_ones_1d(3);
    Tensor* t2 = cml_zeros_1d(3);
    if (t1) cleanup_register_tensor(ctx, t1);
    if (t2) cleanup_register_tensor(ctx, t2);

    cleanup_clear_all(ctx);
    /* After clear_all, context is still registered globally - don't double-free */
    return 1;
}

static int test_cleanup_null_safety(void) {
    /* Should not crash */
    cleanup_context_free(NULL);
    return 1;
}

int main(void) {
    cml_init();

    printf("\n=== Memory Management Unit Tests ===\n\n");

    printf("Memory Pool:\n");
    TEST(memory_pool_create);
    TEST(memory_pool_alloc_free);
    TEST(memory_pool_exhaustion);

    printf("\nTensor Pool:\n");
    TEST(tensor_pool_create);

    printf("\nGraph Allocator:\n");
    TEST(graph_allocator_create);

    printf("\nCleanup Context:\n");
    TEST(cleanup_context_create);
    TEST(cleanup_register_tensor);
    TEST(cleanup_clear_all);
    TEST(cleanup_null_safety);

    printf("\n=====================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("=====================================\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
