/**
 * @file test_trace.c
 * @brief Tests for trace-and-replay JIT system
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/trace.h"

static void test_trace_create_free(void) {
    printf("  test_trace_create_free...");

    CMLTrace* trace = cml_trace_create();
    assert(trace != NULL);
    assert(trace->num_entries == 0);
    assert(trace->num_slots == 0);
    assert(trace->is_recording == false);
    assert(trace->is_complete == false);

    cml_trace_free(trace);
    printf(" PASS\n");
}

static void test_trace_begin_end(void) {
    printf("  test_trace_begin_end...");

    CMLTrace* trace = cml_trace_create();
    assert(trace != NULL);

    uint64_t graph_hash = 0x12345678;
    int ret = cml_trace_begin(trace, graph_hash);
    assert(ret == 0);
    assert(trace->is_recording == true);
    assert(trace->graph_hash == graph_hash);

    ret = cml_trace_end(trace);
    assert(ret == 0);
    assert(trace->is_recording == false);
    assert(trace->is_complete == true);

    cml_trace_free(trace);
    printf(" PASS\n");
}

static void test_trace_record_kernel(void) {
    printf("  test_trace_record_kernel...");

    CMLTrace* trace = cml_trace_create();
    assert(trace != NULL);

    int ret = cml_trace_begin(trace, 0xABCD);
    assert(ret == 0);

    /* Record a kernel entry */
    size_t grid[3] = { 64, 1, 1 };
    size_t block[3] = { 256, 1, 1 };
    int arg_indices[] = { 0, 1, 2 };
    void* fake_kernel = (void*)0xDEAD;

    ret = cml_trace_record_kernel(trace, 0x1111, fake_kernel,
                                   grid, block, arg_indices, 3);
    assert(ret == 0);
    assert(trace->num_entries == 1);

    /* Verify the recorded entry */
    assert(trace->entries[0].type == CML_TRACE_KERNEL);
    assert(trace->entries[0].kernel_hash == 0x1111);
    assert(trace->entries[0].compiled_kernel == fake_kernel);
    assert(trace->entries[0].grid[0] == 64);
    assert(trace->entries[0].block[0] == 256);
    assert(trace->entries[0].num_args == 3);
    assert(trace->entries[0].arg_indices[0] == 0);
    assert(trace->entries[0].arg_indices[1] == 1);
    assert(trace->entries[0].arg_indices[2] == 2);

    /* Record a second kernel */
    size_t grid2[3] = { 32, 32, 1 };
    size_t block2[3] = { 16, 16, 1 };
    int arg_indices2[] = { 3, 4 };

    ret = cml_trace_record_kernel(trace, 0x2222, (void*)0xBEEF,
                                   grid2, block2, arg_indices2, 2);
    assert(ret == 0);
    assert(trace->num_entries == 2);

    ret = cml_trace_end(trace);
    assert(ret == 0);
    assert(trace->is_complete == true);

    cml_trace_free(trace);
    printf(" PASS\n");
}

static void test_cache_create_free(void) {
    printf("  test_cache_create_free...");

    CMLTraceCache* cache = cml_trace_cache_create();
    assert(cache != NULL);
    assert(cache->count == 0);

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

static void test_cache_miss(void) {
    printf("  test_cache_miss...");

    CMLTraceCache* cache = cml_trace_cache_create();
    assert(cache != NULL);

    /* Lookup on empty cache should return NULL */
    CMLTrace* result = cml_trace_cache_lookup(cache, 0xDEADBEEF);
    assert(result == NULL);

    result = cml_trace_cache_lookup(cache, 0);
    assert(result == NULL);

    result = cml_trace_cache_lookup(cache, 999);
    assert(result == NULL);

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

static void test_cache_insert_lookup(void) {
    printf("  test_cache_insert_lookup...");

    CMLTraceCache* cache = cml_trace_cache_create();
    assert(cache != NULL);

    /* Create a trace and record some entries */
    CMLTrace* trace = cml_trace_create();
    assert(trace != NULL);

    uint64_t hash = 0xCAFE;
    int ret = cml_trace_begin(trace, hash);
    assert(ret == 0);

    size_t grid[3] = { 128, 1, 1 };
    size_t block[3] = { 64, 1, 1 };
    int args[] = { 0, 1 };
    ret = cml_trace_record_kernel(trace, 0x5555, (void*)0x1, grid, block, args, 2);
    assert(ret == 0);

    ret = cml_trace_end(trace);
    assert(ret == 0);

    /* Insert into cache */
    ret = cml_trace_cache_insert(cache, hash, trace);
    assert(ret == 0);
    assert(cache->count == 1);

    /* Lookup should find the trace */
    CMLTrace* found = cml_trace_cache_lookup(cache, hash);
    assert(found != NULL);
    assert(found == trace);
    assert(found->graph_hash == hash);
    assert(found->num_entries == 1);
    assert(found->is_complete == true);

    /* Lookup with different hash should miss */
    CMLTrace* not_found = cml_trace_cache_lookup(cache, 0xBEEF);
    assert(not_found == NULL);

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

static void test_cache_multiple_entries(void) {
    printf("  test_cache_multiple_entries...");

    CMLTraceCache* cache = cml_trace_cache_create();
    assert(cache != NULL);

    /* Insert multiple traces */
    for (int i = 0; i < 10; i++) {
        CMLTrace* trace = cml_trace_create();
        assert(trace != NULL);

        uint64_t hash = (uint64_t)(1000 + i);
        int ret = cml_trace_begin(trace, hash);
        assert(ret == 0);
        ret = cml_trace_end(trace);
        assert(ret == 0);

        ret = cml_trace_cache_insert(cache, hash, trace);
        assert(ret == 0);
    }

    assert(cache->count == 10);

    /* Verify all can be found */
    for (int i = 0; i < 10; i++) {
        uint64_t hash = (uint64_t)(1000 + i);
        CMLTrace* found = cml_trace_cache_lookup(cache, hash);
        assert(found != NULL);
        assert(found->graph_hash == hash);
    }

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

int main(void) {
    printf("=== Trace-and-Replay Tests ===\n");

    test_trace_create_free();
    test_trace_begin_end();
    test_trace_record_kernel();
    test_cache_create_free();
    test_cache_miss();
    test_cache_insert_lookup();
    test_cache_multiple_entries();

    printf("All trace-and-replay tests passed.\n");
    return 0;
}
