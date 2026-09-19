#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_require.h"

#include "ops/ir/trace.h"

static void test_trace_create_free(void) {
    printf("  test_trace_create_free...");

    CMLTrace* trace = cml_trace_create();
    REQUIRE(trace != NULL);
    REQUIRE(trace->num_entries == 0);
    REQUIRE(trace->num_slots == 0);
    REQUIRE(trace->is_recording == false);
    REQUIRE(trace->is_complete == false);

    cml_trace_free(trace);
    printf(" PASS\n");
}

static void test_trace_begin_end(void) {
    printf("  test_trace_begin_end...");

    CMLTrace* trace = cml_trace_create();
    REQUIRE(trace != NULL);

    uint64_t graph_hash = 0x12345678;
    int ret             = cml_trace_begin(trace, graph_hash);
    REQUIRE(ret == 0);
    REQUIRE(trace->is_recording == true);
    REQUIRE(trace->graph_hash == graph_hash);

    ret = cml_trace_end(trace);
    REQUIRE(ret == 0);
    REQUIRE(trace->is_recording == false);
    REQUIRE(trace->is_complete == true);

    cml_trace_free(trace);
    printf(" PASS\n");
}

static void test_trace_record_kernel(void) {
    printf("  test_trace_record_kernel...");

    CMLTrace* trace = cml_trace_create();
    REQUIRE(trace != NULL);

    int ret = cml_trace_begin(trace, 0xABCD);
    REQUIRE(ret == 0);

    /* Record a kernel entry */
    size_t grid[3]    = {64, 1, 1};
    size_t block[3]   = {256, 1, 1};
    int arg_indices[] = {0, 1, 2};
    void* fake_kernel = (void*)0xDEAD;

    ret = cml_trace_record_kernel(trace, 0x1111, fake_kernel, grid, block, arg_indices, 3);
    REQUIRE(ret == 0);
    REQUIRE(trace->num_entries == 1);

    /* Verify the recorded entry */
    REQUIRE(trace->entries[0].type == CML_TRACE_KERNEL);
    REQUIRE(trace->entries[0].kernel_hash == 0x1111);
    REQUIRE(trace->entries[0].compiled_kernel == fake_kernel);
    REQUIRE(trace->entries[0].grid[0] == 64);
    REQUIRE(trace->entries[0].block[0] == 256);
    REQUIRE(trace->entries[0].num_args == 3);
    REQUIRE(trace->entries[0].arg_indices[0] == 0);
    REQUIRE(trace->entries[0].arg_indices[1] == 1);
    REQUIRE(trace->entries[0].arg_indices[2] == 2);

    /* Record a second kernel */
    size_t grid2[3]    = {32, 32, 1};
    size_t block2[3]   = {16, 16, 1};
    int arg_indices2[] = {3, 4};

    ret = cml_trace_record_kernel(trace, 0x2222, (void*)0xBEEF, grid2, block2, arg_indices2, 2);
    REQUIRE(ret == 0);
    REQUIRE(trace->num_entries == 2);

    ret = cml_trace_end(trace);
    REQUIRE(ret == 0);
    REQUIRE(trace->is_complete == true);

    cml_trace_free(trace);
    printf(" PASS\n");
}

static void test_cache_create_free(void) {
    printf("  test_cache_create_free...");

    CMLTraceCache* cache = cml_trace_cache_create();
    REQUIRE(cache != NULL);
    REQUIRE(cache->count == 0);

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

static void test_cache_miss(void) {
    printf("  test_cache_miss...");

    CMLTraceCache* cache = cml_trace_cache_create();
    REQUIRE(cache != NULL);

    /* Lookup on empty cache should return NULL */
    CMLTrace* result = cml_trace_cache_lookup(cache, 0xDEADBEEF);
    REQUIRE(result == NULL);

    result = cml_trace_cache_lookup(cache, 0);
    REQUIRE(result == NULL);

    result = cml_trace_cache_lookup(cache, 999);
    REQUIRE(result == NULL);

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

static void test_cache_insert_lookup(void) {
    printf("  test_cache_insert_lookup...");

    CMLTraceCache* cache = cml_trace_cache_create();
    REQUIRE(cache != NULL);

    /* Create a trace and record some entries */
    CMLTrace* trace = cml_trace_create();
    REQUIRE(trace != NULL);

    uint64_t hash = 0xCAFE;
    int ret       = cml_trace_begin(trace, hash);
    REQUIRE(ret == 0);

    size_t grid[3]  = {128, 1, 1};
    size_t block[3] = {64, 1, 1};
    int args[]      = {0, 1};
    ret             = cml_trace_record_kernel(trace, 0x5555, (void*)0x1, grid, block, args, 2);
    REQUIRE(ret == 0);

    ret = cml_trace_end(trace);
    REQUIRE(ret == 0);

    /* Insert into cache */
    ret = cml_trace_cache_insert(cache, hash, trace);
    REQUIRE(ret == 0);
    REQUIRE(cache->count == 1);

    /* Lookup should find the trace */
    CMLTrace* found = cml_trace_cache_lookup(cache, hash);
    REQUIRE(found != NULL);
    REQUIRE(found == trace);
    REQUIRE(found->graph_hash == hash);
    REQUIRE(found->num_entries == 1);
    REQUIRE(found->is_complete == true);

    /* Lookup with different hash should miss */
    CMLTrace* not_found = cml_trace_cache_lookup(cache, 0xBEEF);
    REQUIRE(not_found == NULL);

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

static void test_cache_multiple_entries(void) {
    printf("  test_cache_multiple_entries...");

    CMLTraceCache* cache = cml_trace_cache_create();
    REQUIRE(cache != NULL);

    /* Insert multiple traces */
    for (int i = 0; i < 10; i++) {
        CMLTrace* trace = cml_trace_create();
        REQUIRE(trace != NULL);

        uint64_t hash = (uint64_t)(1000 + i);
        int ret       = cml_trace_begin(trace, hash);
        REQUIRE(ret == 0);
        ret = cml_trace_end(trace);
        REQUIRE(ret == 0);

        ret = cml_trace_cache_insert(cache, hash, trace);
        REQUIRE(ret == 0);
    }

    REQUIRE(cache->count == 10);

    /* Verify all can be found */
    for (int i = 0; i < 10; i++) {
        uint64_t hash   = (uint64_t)(1000 + i);
        CMLTrace* found = cml_trace_cache_lookup(cache, hash);
        REQUIRE(found != NULL);
        REQUIRE(found->graph_hash == hash);
    }

    cml_trace_cache_free(cache);
    printf(" PASS\n");
}

int main(void) {
    printf("Trace-and-Replay Tests\n");

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
